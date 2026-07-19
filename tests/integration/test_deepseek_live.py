"""Live integration tests against any OpenAI-compatible API (DeepSeek by
default). These cost real API credits and run only when both are set:

    LLM_API_KEY=sk-...    (or DEEPSEEK_API_KEY)
    SPAL_RUN_LIVE=1

The adapter (``OpenAICompatAdapter``) lives in ``examples/llm_adapters.py``,
not in the package — see that module's docstring.
"""
from __future__ import annotations

import os

import pytest

from state_projection_loop import Config, Registry, Session
from state_projection_loop.policy import PolicyEngine

from examples.coding_agent.tools import CODING_KERNEL, build_coding_registry, seed_workspace
from examples.customer_support.tools import SUPPORT_KERNEL, SupportBackend, build_support_registry
from examples.llm_adapters import OpenAICompatAdapter

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not ((os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY"))
             and os.environ.get("SPAL_RUN_LIVE") == "1"),
        reason="live tests need LLM_API_KEY (or DEEPSEEK_API_KEY) and SPAL_RUN_LIVE=1",
    ),
]


def adapter(**kw) -> OpenAICompatAdapter:
    return OpenAICompatAdapter(
        model=os.environ.get("LLM_MODEL", "deepseek-v4-flash"),
        api_key=os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY"),
        base_url=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"),
        temperature=kw.pop("temperature", 0.0), **kw,
    )


def allow_all() -> PolicyEngine:
    return PolicyEngine(default_decision="allow")


class TestBasicChat:
    def test_plain_chat_reply(self):
        session = Session(adapter(), kernel="あなたは簡潔に日本語で答えるアシスタントです。")
        reply = session.send("1+1は?数字だけ答えて。")
        assert "2" in reply
        assert session.budget.prompt_tokens > 0  # provider usage recorded

    def test_multi_turn_context_kept(self):
        session = Session(adapter(), kernel="あなたは簡潔に日本語で答えるアシスタントです。")
        session.send("私の名前はモシモです。覚えてください。")
        reply = session.send("私の名前は?")
        assert "モシモ" in reply


class TestNativeToolCalling:
    def test_model_calls_registered_tool(self):
        called = {}

        def get_stock(product_id: str) -> dict:
            called["product_id"] = product_id
            return {"product_id": product_id, "stock": 42, "warehouse": "Tokyo-2"}

        reg = Registry()
        reg.register({
            "name": "inventory.stock.get",
            "category": "inventory",
            "spec": {
                "description": "商品IDの在庫数を返す。",
                "parameters": {"type": "object",
                               "properties": {"product_id": {"type": "string"}},
                               "required": ["product_id"]},
            },
            "discovery": {"embedding_text": "在庫 いくつ 残り 数 stock inventory"},
            "execution": {"retry_safety": "pure"},
            "effects": [{"kind": "none"}],
        }, handler=get_stock)

        session = Session(
            adapter(), kernel="あなたは在庫管理アシスタント。在庫は必ず inventory.stock.get で確認してから答える。",
            registry=reg, policy=allow_all(),
        )
        reply = session.send("商品 AP-100C の在庫はいくつ?")
        assert called.get("product_id") == "AP-100C"
        assert "42" in reply

    def test_self_repair_on_enum_violation(self):
        """Force a validation error via a constrained enum and confirm the
        model recovers using the attached spec (live)."""
        moods = []

        def set_mood(mood: str) -> str:
            moods.append(mood)
            return f"mood set to {mood}"

        reg = Registry()
        reg.register({
            "name": "ui.mood.set",
            "category": "ui",
            "spec": {
                "description": "アバターの気分を設定する。",
                "parameters": {"type": "object",
                               "properties": {"mood": {"enum": ["joyful", "melancholic", "serene"]}},
                               "required": ["mood"]},
            },
            "discovery": {"embedding_text": "気分 ムード 表情 感情 mood"},
            "execution": {"retry_safety": "idempotent"},
            "effects": [{"kind": "none"}],
        }, handler=set_mood)

        session = Session(
            adapter(), kernel="ユーザーの依頼に応じて ui.mood.set を呼ぶ。値はスキーマに従うこと。",
            registry=reg, policy=allow_all(),
        )
        session.send("アバターを『穏やか』な気分にして")
        assert moods and moods[-1] in ("joyful", "melancholic", "serene")


class TestLiveCustomerSupport:
    def test_error_code_answered_from_manual(self):
        backend = SupportBackend()
        session = Session(adapter(), kernel=SUPPORT_KERNEL,
                          registry=build_support_registry(backend), policy=allow_all())
        reply = session.send("浄水器 AquaPure AP-100 に E03 というエラーが出ています。どうすれば?")
        assert any(k in reply for k in ("カートリッジ", "AP-100C", "フィルター"))
        assert any(key.startswith("search:") for key in backend.metrics), \
            "the model should have consulted the manual"

    def test_live_escalation_collects_contacts(self):
        backend = SupportBackend()
        session = Session(adapter(), kernel=SUPPORT_KERNEL,
                          registry=build_support_registry(backend), policy=allow_all())
        session.send("何をしても直りません。人間の担当者に代わってください。"
                     "連絡先はメール taro@example.com、電話 090-1234-5678 です。"
                     "問題は SmartBrew SB-2 の B4 エラーが再発することです。")
        assert backend.tickets, "support.ticket.escalate should have been called"
        ticket = backend.tickets[0]
        assert ticket["email"] == "taro@example.com"
        assert "090-1234-5678" in ticket["phone"]
        assert ticket["transcript"], "the conversation transcript must be attached"


class TestLiveCodingAgent:
    def test_fixes_failing_test(self, tmp_path):
        seed_workspace(tmp_path)
        cfg = Config.from_dict({"mode": "job", "budget": {"max_steps": 12}})
        session = Session(adapter(), kernel=CODING_KERNEL,
                          registry=build_coding_registry(tmp_path), config=cfg, policy=allow_all())
        session.run_job("dev.tests.run を実行し、失敗しているテストを修正してください。"
                        "修正後は必ず dev.tests.run で確認し、finish(result) してください。")
        fixed = (tmp_path / "calculator.py").read_text(encoding="utf-8")
        assert "ValueError" in fixed, "the model should have added the zero-division guard"
        import subprocess, sys

        proc = subprocess.run([sys.executable, str(tmp_path / "test_calculator.py")],
                              capture_output=True, text=True, cwd=tmp_path)
        assert proc.returncode == 0, proc.stdout + proc.stderr


class TestLiveCompaction:
    def test_summarizer_contract_over_live_model(self):
        """Small window forces folding; the live model produces the delta."""
        cfg = Config.from_dict({"projection": {"window_tokens": 2500}})
        session = Session(adapter(), kernel="日本語で長めに丁寧に答えるアシスタント。", config=cfg)
        for q in ("日本の四季それぞれの魅力を語って", "その中で旅行に最適な季節は?",
                  "北海道でおすすめの街は?", "そこで食べるべきものは?"):
            session.send(q)
        assert not session.working_state.is_empty(), "the conversation should have been folded at least once"
        assert session.send("最初に私が聞いた話題は何だった?")  # continuity survives folding
