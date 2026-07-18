"""Customer-support scenario toolkit.

A realistic web-support agent built purely from tool registration (§1):
manual search, human escalation (records the full transcript), and an
artifact-style chart card. ``SupportBackend`` is an in-memory stand-in for
a real datastore (e.g. Firestore) and also aggregates the usage metrics a
dashboard would read.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from state_projection_loop import Registry, ToolContext

MANUALS_DIR = Path(__file__).parent / "manuals"

_EMAIL = re.compile(r"^[\w.+-]+@[\w-]+\.[\w.-]+$")


# ---------------------------------------------------------------------------
# Backend (in-memory stand-in for Firestore & the metrics dashboard)
# ---------------------------------------------------------------------------

@dataclass
class SupportBackend:
    tickets: list[dict[str, Any]] = field(default_factory=list)
    charts: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, int] = field(default_factory=dict)

    def bump(self, key: str) -> None:
        self.metrics[key] = self.metrics.get(key, 0) + 1

    def next_ticket_id(self) -> str:
        return f"TCK-{len(self.tickets) + 1:04d}"


# ---------------------------------------------------------------------------
# Manual loading & sectioning
# ---------------------------------------------------------------------------

def load_manuals(directory: Path = MANUALS_DIR) -> dict[str, list[dict[str, str]]]:
    """{product_id: [{heading, content}, ...]} split on '## ' headings."""
    manuals: dict[str, list[dict[str, str]]] = {}
    for path in sorted(directory.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        product = path.stem
        sections: list[dict[str, str]] = []
        heading = "(title)"
        buf: list[str] = []
        for line in text.splitlines():
            if line.startswith("## "):
                if buf:
                    sections.append({"heading": heading, "content": "\n".join(buf).strip()})
                heading = line[3:].strip()
                buf = []
            else:
                buf.append(line)
        if buf:
            sections.append({"heading": heading, "content": "\n".join(buf).strip()})
        manuals[product] = sections
    return manuals


def _score(query: str, text: str) -> int:
    terms = [t for t in re.split(r"[\s、。]+", query) if t]
    return sum(text.lower().count(t.lower()) for t in terms)


# ---------------------------------------------------------------------------
# Registry builder
# ---------------------------------------------------------------------------

def build_support_registry(backend: SupportBackend, manuals_dir: Path = MANUALS_DIR) -> Registry:
    manuals = load_manuals(manuals_dir)
    registry = Registry()

    def search_manual(query: str, product: Optional[str] = None, k: int = 3) -> Any:
        backend.bump(f"search:{product or 'all'}")
        hits: list[tuple[int, dict[str, str], str]] = []
        for pid, sections in manuals.items():
            if product and product != pid:
                continue
            for section in sections:
                score = _score(query, section["heading"] + "\n" + section["content"])
                if score > 0:
                    hits.append((score, section, pid))
        hits.sort(key=lambda h: h[0], reverse=True)
        if not hits:
            return f"No manual sections matched {query!r}. Products: {', '.join(manuals)}"
        return [
            {"product": pid, "heading": s["heading"], "content": s["content"]}
            for _, s, pid in hits[:k]
        ]

    def get_manual_section(product: str, heading: str) -> str:
        for section in manuals.get(product, []):
            if heading in section["heading"]:
                return section["content"]
        available = ", ".join(s["heading"] for s in manuals.get(product, []))
        return f"Section {heading!r} not found in {product}. Available: {available or '(no such product)'}"

    def escalate_to_human(ctx: ToolContext, email: str, phone: str, problem_summary: str) -> dict[str, Any]:
        if not _EMAIL.match(email):
            raise ValueError(f"invalid email address: {email!r} — ask the user again")
        transcript = [
            {"role": m.role, "text": m.text()}
            for m in ctx.session.conversation
            if m.role in ("user", "assistant") and m.text()
        ]
        ticket = {
            "id": backend.next_ticket_id(),
            "email": email,
            "phone": phone,
            "problem_summary": problem_summary,
            "transcript": transcript,
        }
        backend.tickets.append(ticket)
        backend.bump("escalation")
        return {"ticket_id": ticket["id"], "status": "sent_to_support",
                "message": "担当者から24時間以内にご連絡します。"}

    def render_chart(title: str, chart_type: str, data: dict, open: bool = True) -> dict[str, Any]:  # noqa: A002
        card = {
            "card_id": f"card-{len(backend.charts) + 1}",
            "title": title,
            "chart_type": chart_type,
            "data": data,
            "open": open,  # cards render expanded by default; user may collapse
        }
        backend.charts.append(card)
        backend.bump("chart")
        return {"card_id": card["card_id"], "status": "rendered", "open": open}

    registry.register({
        "name": "search_manual",
        "category": "support/manuals",
        "spec": {
            "description": "登録製品の説明書を全文検索し、該当セクションを返す。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "エラーコードや症状、知りたい操作"},
                    "product": {"type": ["string", "null"],
                                "description": f"製品ID ({', '.join(manuals)})。省略で全製品"},
                    "k": {"type": "integer", "default": 3, "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
            },
            "usage_notes": "回答は必ず説明書の記載に基づくこと。見つからなければ推測せずその旨を伝える。",
        },
        "discovery": {"embedding_text": "説明書 マニュアル 使い方 エラー 故障 手順 調べる トラブル 対処"},
        "execution": {"timeout_s": 10, "parallel_safe": True,
                      "output_policy": {"max_inline_tokens": 200, "preview": "head"}},
    }, handler=search_manual)

    registry.register({
        "name": "get_manual_section",
        "category": "support/manuals",
        "spec": {
            "description": "製品IDと見出しを指定して説明書の特定セクションを取得する。",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {"type": "string"},
                    "heading": {"type": "string"},
                },
                "required": ["product", "heading"],
            },
        },
        "discovery": {"embedding_text": "説明書 セクション 章 見出し 該当箇所"},
        "execution": {"timeout_s": 10, "parallel_safe": True},
    }, handler=get_manual_section)

    registry.register({
        "name": "escalate_to_human",
        "category": "support/tickets",
        "spec": {
            "description": (
                "人間のサポート担当者へ引き継ぐ。ユーザーから聞き取ったメールアドレス・電話番号・"
                "問題の要約に、これまでの会話全文を添えて送信する。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string", "description": "ユーザーの連絡先メールアドレス"},
                    "phone": {"type": "string", "description": "ユーザーの電話番号"},
                    "problem_summary": {"type": "string", "description": "問題と試した対応の要約"},
                },
                "required": ["email", "phone", "problem_summary"],
            },
            "usage_notes": "呼び出す前に必ずメールアドレスと電話番号をユーザーに確認すること。",
        },
        "discovery": {
            "require_spec": True,
            "embedding_text": "人間 担当者 オペレーター 引き継ぎ エスカレーション 解決しない 直接話したい",
        },
        "execution": {"timeout_s": 15},
    }, handler=escalate_to_human)

    registry.register({
        "name": "render_chart",
        "category": "support/artifacts",
        "spec": {
            "description": "グラフ/チャートカードを描画して表示する(artifacts相当)。カードは既定で開いた状態で表示され、ユーザーが閉じられる。",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "chart_type": {"enum": ["bar", "line", "pie"]},
                    "data": {"type": "object", "description": "{labels: [...], values: [...]}"},
                    "open": {"type": "boolean", "default": True},
                },
                "required": ["title", "chart_type", "data"],
            },
        },
        "discovery": {"embedding_text": "グラフ チャート 図 可視化 表示 比較 推移"},
        "execution": {"timeout_s": 10},
    }, handler=render_chart)

    return registry


SUPPORT_KERNEL = """あなたは家電メーカーのカスタマーサポートAIです。
- 回答は必ず search_manual / get_manual_section で説明書を確認してから行う。推測で答えない。
- 解決しない場合や人間の対応を求められた場合は、メールアドレスと電話番号を聞き取り、
  escalate_to_human で担当者へ引き継ぐ。
- 数値の比較や推移を示すときは render_chart でチャートカードを表示できる。
- 丁寧な日本語で簡潔に答える。"""
