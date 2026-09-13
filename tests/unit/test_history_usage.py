from state_projection_loop import Config, ScriptedLLM, Session
from state_projection_loop.messages import Decision, ToolCall, Usage
from state_projection_loop.tokens import estimate_tokens
import pytest


def test_long_reply_survives_ledger_and_next_projection():
    reply = "x" * 2100 + "IMPORTANT_END"
    llm = ScriptedLLM([reply, "ok"])
    session = Session(llm)
    assert session.send("one") == reply
    assert session.conversation[-1].content == reply
    session.send("continue")
    assert any(m.content == reply for m in llm.requests[1]["messages"])


@pytest.mark.parametrize("kind", ["arguments", "raw", "finish", "usage"])
def test_usage_counts_complete_request_and_output(kind):
    payload = "x" * 8000
    call = ToolCall(name="missing_tool", arguments={"text": payload})
    if kind == "raw":
        call.arguments = {}
        call.raw_arguments = '{"text":"' + payload
    decision = Decision(calls=[call])
    if kind == "finish":
        decision = ScriptedLLM.finish({"text": payload})
    if kind == "usage":
        decision.usage = Usage(prompt_tokens=11, completion_tokens=7)
    llm = ScriptedLLM([decision, Decision(text="ok", usage=Usage())])
    config = Config.from_dict({"budget": {"cost_per_1k_input": 1, "cost_per_1k_output": 2}})
    session = Session(llm, config=config)
    session.send("go")
    if kind == "usage":
        assert session.budget.prompt_tokens == 11
        assert session.budget.completion_tokens == 7
    else:
        request = llm.requests[0]
        assert session.budget.prompt_tokens == estimate_tokens(request["messages"]) + estimate_tokens(request["tools"])
        assert session.budget.completion_tokens >= estimate_tokens(payload)
    assert session.budget.cost == pytest.approx(
        session.budget.prompt_tokens / 1000 + session.budget.completion_tokens / 1000 * 2
    )
