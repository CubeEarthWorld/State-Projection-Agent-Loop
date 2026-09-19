"""Long-horizon recall scenarios of different shapes, in English and
Japanese, for ``compression_eval``.

Each scenario is a kernel, a registry of fake tools, the user's turns, and
questions whose answers are graded by exact substring. The shapes differ in
what compression must preserve:

* ``records``  — facts buried in noisy tool output; some later corrected
* ``coding``   — test runs that fail with a specific error, then pass
* ``support``  — several tickets whose status changes across the thread
* ``game``     — inventory and flags that the user's own words change
* ``document`` — sections too large to inline (artifacts) read early on

Answers are identifiers or short words; the ``ja`` variants translate the
prose and keep the identifiers, so the two languages are graded alike.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

from state_projection_loop import Registry

T = {
    "en": {
        "kernel": ("You are a careful assistant. Use the tools when asked, then reply in ONE short sentence "
                   "that repeats the key figure, status or name. When asked a question, answer from this "
                   "conversation only, in one short sentence; if it was never stated, answer exactly: not stated."),
        "not_stated": "not stated",
        "lookup": "Look up record {rid}.",
        "correction": "Correction on {rid}: the amount is now {amount} EUR, please note it.",
        "manager": "By the way, our account manager is Petra Lindqvist and the site code is ZX-41.",
        "q_invoice": "What was the invoice number for {rid}?",
        "q_status": "What status did {rid} have?",
        "q_amount": "What is the current amount for {rid}?",
        "q_manager": "Who is our account manager?",
        "q_site": "What is the site code?",
        "q_driver": "What was the delivery driver's name for {rid}?",
        "run_tests": "Run the tests for {module}.",
        "fixed": "I fixed {module}; run its tests again.",
        "convention": "Remember: this repo uses tabs, and the config lives in settings/base.toml.",
        "q_error": "What error did the first test run of {module} report?",
        "q_config": "Where does the config live?",
        "q_indent": "Tabs or spaces in this repo?",
        "q_last": "Did the last run of {module} pass?",
        "q_reviewer": "Who reviewed the fix to {module}?",
        "pass": "pass",
        "open_ticket": "Open ticket {tid}.",
        "resolve": "Ticket {tid} is resolved now, note it.",
        "q_tstatus": "What is the current status of ticket {tid}?",
        "q_email": "What was the customer email on ticket {tid}?",
        "q_open": "Is ticket {tid} still open? Answer yes or no.",
        "q_phone": "What was the phone number on ticket {tid}?",
        "resolved": "resolved",
        "no": "no",
        "roll": "Roll for the {action}.",
        "take": "I pick up the {item}.",
        "drop": "I drop the {item}.",
        "q_have": "Do I still have the {item}? Answer yes or no.",
        "q_key": "Which key do I carry?",
        "q_first": "What was my very first roll for?",
        "q_gold": "How much gold do I have?",
        "yes": "yes",
        "read": "Read section {sid} of the contract.",
        "q_clause": "What is the notice period in section {sid}?",
        "q_penalty": "What is the penalty amount in section {sid}?",
        "q_venue": "Which court is named in section {sid}?",
    },
    "ja": {
        "kernel": ("あなたは慎重なアシスタントです。頼まれたらツールを使い、重要な数値・状態・名前を繰り返して1文で答えてください。"
                   "質問にはこの会話の内容だけから1文で答え、述べられていないことは正確に「不明」とだけ答えてください。"),
        "not_stated": "不明",
        "lookup": "記録 {rid} を照会して。",
        "correction": "{rid} の訂正：金額は {amount} EUR になりました。控えておいて。",
        "manager": "ところで、担当マネージャーは Petra Lindqvist で、拠点コードは ZX-41 です。",
        "q_invoice": "{rid} の請求書番号は何でしたか？",
        "q_status": "{rid} の状態は何でしたか？",
        "q_amount": "{rid} の現在の金額はいくらですか？",
        "q_manager": "担当マネージャーは誰ですか？",
        "q_site": "拠点コードは何ですか？",
        "q_driver": "{rid} の配達ドライバーの名前は？",
        "run_tests": "{module} のテストを実行して。",
        "fixed": "{module} を直しました。テストをもう一度実行して。",
        "convention": "覚えておいて：このリポジトリはタブを使い、設定は settings/base.toml にあります。",
        "q_error": "{module} の最初のテスト実行はどんなエラーを報告しましたか？",
        "q_config": "設定はどこにありますか？",
        "q_indent": "このリポジトリはタブとスペースのどちらですか？",
        "q_last": "{module} の最後の実行は成功しましたか？",
        "q_reviewer": "{module} の修正をレビューしたのは誰ですか？",
        "pass": "成功",
        "open_ticket": "チケット {tid} を開いて。",
        "resolve": "チケット {tid} は解決済みになりました。控えておいて。",
        "q_tstatus": "チケット {tid} の現在の状態は？",
        "q_email": "チケット {tid} の顧客メールアドレスは？",
        "q_open": "チケット {tid} はまだ未解決ですか？「はい」か「いいえ」で答えて。",
        "q_phone": "チケット {tid} の電話番号は？",
        "resolved": "解決",
        "no": "いいえ",
        "roll": "{action}のためにダイスを振って。",
        "take": "{item}を拾う。",
        "drop": "{item}を捨てる。",
        "q_have": "{item}はまだ持っていますか？「はい」か「いいえ」で答えて。",
        "q_key": "どの鍵を持っていますか？",
        "q_first": "最初のダイスは何のために振りましたか？",
        "q_gold": "所持金はいくらですか？",
        "yes": "はい",
        "read": "契約書の第 {sid} 条を読んで。",
        "q_clause": "第 {sid} 条の通知期間は？",
        "q_penalty": "第 {sid} 条の違約金の額は？",
        "q_venue": "第 {sid} 条で指定されている裁判所は？",
    },
}

NOISE = {
    "en": ["region: EU-WEST", "carrier: Nordfreight", "priority: normal", "sla: 48h", "checked_by: system",
           "warehouse: Rotterdam-3", "pallets: 12", "temperature: ambient", "customs: cleared", "docs: complete"],
    "ja": ["地域: 関東", "運送会社: 北陸フレイト", "優先度: 通常", "SLA: 48時間", "確認者: システム",
           "倉庫: 川崎第3", "パレット数: 12", "温度帯: 常温", "通関: 完了", "書類: 完備"],
}


@dataclass
class Scenario:
    kernel: str
    registry: Registry
    steps: list[str]
    questions: list[dict[str, Any]]
    builtins: tuple[str, ...] = ()


def _register(registry: Registry, name: str, description: str, prop: str, handler: Callable[..., Any],
              *, pinned: bool = True) -> None:
    registry.register({
        "name": name, "category": name.split(".")[0],
        "spec": {"description": description,
                 "parameters": {"type": "object", "properties": {prop: {"type": "string"}}, "required": [prop]}},
        "discovery": {"pinned": pinned},
        "execution": {"retry_safety": "pure"},
        "effects": [{"kind": "read", "resource": "fake:*"}],
    }, handler=handler)


def records(seed: int, turns: int, lang: str) -> Scenario:
    t, rng = T[lang], random.Random(seed)
    recs: dict[str, dict] = {}
    steps: list[str] = []
    for i in range(turns):
        rid = f"R{seed * 1000 + i:05d}"
        recs[rid] = {"amount": rng.randint(1000, 9999), "status": rng.choice(["delivered", "in transit", "held", "returned"]),
                     "invoice": f"INV-{rng.randint(10000, 99999)}"}
        steps.append(t["lookup"].format(rid=rid))
    updated = rng.sample(list(recs), 3)
    for n, rid in enumerate(updated):
        recs[rid]["updated"] = recs[rid]["amount"] + 500
        steps.insert(turns // 2 + n * 2, t["correction"].format(rid=rid, amount=recs[rid]["updated"]))
    steps.insert(2, t["manager"])
    early, late = list(recs)[1], list(recs)[-2]
    noise_rng = random.Random(seed + 7)

    def lookup(id: str) -> str:
        r = recs.get(id)
        if r is None:
            return f"record {id}: not found"
        lines = [f"record {id}", f"status: {r['status']}"] + [noise_rng.choice(NOISE[lang]) for _ in range(28)]
        lines.insert(noise_rng.randint(3, len(lines)), f"amount: {r['amount']} EUR")
        lines.insert(noise_rng.randint(3, len(lines)), f"invoice: {r['invoice']}")
        return "\n".join(lines)

    registry = Registry()
    _register(registry, "ops.record.lookup", "Look up a shipment record by id.", "id", lookup)
    return Scenario(t["kernel"], registry, steps, [
        {"ask": t["q_invoice"].format(rid=early), "answer": recs[early]["invoice"], "kind": "early_tool_fact"},
        {"ask": t["q_status"].format(rid=late), "answer": recs[late]["status"], "kind": "late_tool_fact"},
        {"ask": t["q_amount"].format(rid=updated[0]), "answer": str(recs[updated[0]]["updated"]), "kind": "update",
         "stale": str(recs[updated[0]]["amount"])},
        {"ask": t["q_manager"], "answer": "Lindqvist", "kind": "user_fact"},
        {"ask": t["q_site"], "answer": "ZX-41", "kind": "user_fact"},
        {"ask": t["q_driver"].format(rid=early), "answer": t["not_stated"], "kind": "abstain"},
    ])


def coding(seed: int, turns: int, lang: str) -> Scenario:
    t, rng = T[lang], random.Random(seed)
    modules = [f"pkg_{seed}_{i}" for i in range(max(3, turns // 3))]
    errors = {m: f"AssertionError: expected {rng.randint(1, 9)} got {rng.randint(10, 99)} in test_{m}_{rng.randint(1, 9)}"
              for m in modules}
    runs: dict[str, int] = {m: 0 for m in modules}
    steps: list[str] = []
    for i in range(turns):
        m = modules[i % len(modules)]
        steps.append(t["run_tests"].format(module=m) if i < len(modules) else t["fixed"].format(module=m))
    steps.insert(1, t["convention"])

    def run_tests(module: str) -> str:
        runs[module] = runs.get(module, 0) + 1
        lines = [f"collected {rng.randint(20, 40)} items"] + [f"test_{module}_{j} PASSED" for j in range(1, 25)]
        if runs[module] == 1:
            lines += [f"FAILED test_{module}: {errors[module]}", "1 failed, 24 passed", "exit=1"]
        else:
            lines += ["25 passed", "exit=0"]
        return "\n".join(lines)

    registry = Registry()
    _register(registry, "dev.tests.run", "Run the tests of a module; returns the pytest output.", "module", run_tests)
    first = modules[0]
    return Scenario(t["kernel"], registry, steps, [
        {"ask": t["q_error"].format(module=first), "answer": errors[first].split(":")[1].strip()[:12],
         "kind": "early_error"},
        {"ask": t["q_config"], "answer": "settings/base.toml", "kind": "user_fact"},
        {"ask": t["q_indent"], "answer": "tab" if lang == "en" else "タブ", "kind": "user_fact"},
        {"ask": t["q_last"].format(module=modules[-1]), "answer": t["pass"] if turns > len(modules) else "",
         "kind": "late_tool_fact"},
        {"ask": t["q_reviewer"].format(module=first), "answer": t["not_stated"], "kind": "abstain"},
    ])


def support(seed: int, turns: int, lang: str) -> Scenario:
    t, rng = T[lang], random.Random(seed)
    tickets = {f"T-{seed}{i:03d}": {"email": f"user{rng.randint(100, 999)}@example.com", "status": "open"}
               for i in range(turns)}
    ids = list(tickets)
    steps = [t["open_ticket"].format(tid=tid) for tid in ids]
    resolved = rng.sample(ids[: turns // 2], 3)
    for n, tid in enumerate(resolved):
        steps.insert(turns // 2 + n * 3, t["resolve"].format(tid=tid))

    def get(id: str) -> str:
        k = tickets.get(id)
        if k is None:
            return f"ticket {id}: not found"
        return "\n".join([f"ticket {id}", "status: open", f"customer_email: {k['email']}", "channel: web",
                          "product: SmartBrew SB-2"] + [rng.choice(NOISE[lang]) for _ in range(20)])

    registry = Registry()
    _register(registry, "crm.ticket.get", "Fetch a support ticket by id.", "id", get)
    return Scenario(t["kernel"], registry, steps, [
        {"ask": t["q_tstatus"].format(tid=resolved[0]), "answer": t["resolved"], "kind": "update"},
        {"ask": t["q_email"].format(tid=ids[1]), "answer": tickets[ids[1]]["email"], "kind": "early_tool_fact"},
        {"ask": t["q_open"].format(tid=resolved[1]), "answer": t["no"], "kind": "update"},
        {"ask": t["q_email"].format(tid=ids[-2]), "answer": tickets[ids[-2]]["email"], "kind": "late_tool_fact"},
        {"ask": t["q_phone"].format(tid=ids[1]), "answer": t["not_stated"], "kind": "abstain"},
    ])


def game(seed: int, turns: int, lang: str) -> Scenario:
    t, rng = T[lang], random.Random(seed)
    items = {"en": ["torch", "rope", "silver key", "map", "lantern"], "ja": ["松明", "ロープ", "銀の鍵", "地図", "ランタン"]}[lang]
    actions = {"en": ["lockpicking", "climbing", "persuasion", "stealth"], "ja": ["鍵開け", "登攀", "説得", "隠密"]}[lang]
    steps: list[str] = []
    first_action = actions[0]
    for i in range(turns):
        steps.append(t["roll"].format(action=first_action if i == 0 else rng.choice(actions)))
    steps.insert(2, t["take"].format(item=items[0]))
    steps.insert(4, t["take"].format(item=items[2]))
    steps.insert(turns // 2, t["drop"].format(item=items[0]))
    steps.insert(turns // 2 + 4, t["take"].format(item=items[3]))

    def roll(action: str) -> str:
        return "\n".join([f"roll for {action}: d20 = {rng.randint(1, 20)}", "modifier: +2", "result: success"]
                         + [rng.choice(NOISE[lang]) for _ in range(6)])

    registry = Registry()
    _register(registry, "game.dice.roll", "Roll a d20 for an action.", "action", roll)
    return Scenario(t["kernel"], registry, steps, [
        {"ask": t["q_have"].format(item=items[0]), "answer": t["no"], "kind": "update"},
        {"ask": t["q_key"], "answer": items[2], "kind": "user_fact"},
        {"ask": t["q_have"].format(item=items[3]), "answer": t["yes"], "kind": "user_fact"},
        {"ask": t["q_first"], "answer": first_action, "kind": "early_user_fact"},
        {"ask": t["q_gold"], "answer": t["not_stated"], "kind": "abstain"},
    ])


def document(seed: int, turns: int, lang: str) -> Scenario:
    t, rng = T[lang], random.Random(seed)
    sections = {str(i + 1): {"notice": f"{rng.choice([14, 30, 45, 60, 90])} days", "penalty": f"{rng.randint(2, 9)}000 EUR",
                             "venue": rng.choice(["Rotterdam", "Hamburg", "Vienna", "Lyon"])} for i in range(turns)}
    filler = {"en": "The parties agree that the obligations set out in this section apply in full.",
              "ja": "両当事者は、本条に定める義務が全面的に適用されることに合意する。"}[lang]

    def read(section: str) -> str:
        s = sections.get(section)
        if s is None:
            return f"section {section}: not found"
        body = [f"Section {section}"] + [filler] * 25
        body.insert(8, f"Notice period: {s['notice']}.")
        body.insert(16, f"Penalty: {s['penalty']}.")
        body.insert(24, f"Venue: the courts of {s['venue']}.")
        return "\n".join(body) + "\n" + "\n".join([filler] * 40)

    registry = Registry()
    _register(registry, "docs.section.read", "Read one section of the contract.", "section", read)
    steps = [t["read"].format(sid=str(i + 1)) for i in range(turns)]
    return Scenario(t["kernel"], registry, steps, [
        {"ask": t["q_clause"].format(sid="2"), "answer": sections["2"]["notice"].split()[0], "kind": "early_tool_fact"},
        {"ask": t["q_penalty"].format(sid=str(turns - 1)), "answer": sections[str(turns - 1)]["penalty"].split()[0],
         "kind": "late_tool_fact"},
        {"ask": t["q_venue"].format(sid="2"), "answer": sections["2"]["venue"], "kind": "early_tool_fact"},
        {"ask": t["q_venue"].format(sid=str(turns)), "answer": sections[str(turns)]["venue"], "kind": "late_tool_fact"},
    ], builtins=("meta",))


SCENARIOS: dict[str, Callable[[int, int, str], Scenario]] = {
    "records": records, "coding": coding, "support": support, "game": game, "document": document,
}
