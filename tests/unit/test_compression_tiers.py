"""History compression as the model sees it over a long run: the verbatim
point moves in steps, old tool results are masked but errors stay readable,
the user's words are never touched, and between steps the rendered prefix
is byte-identical (what a provider's prompt cache needs)."""
from __future__ import annotations

from state_projection_loop import Config, Registry, ScriptedLLM, Session
from state_projection_loop.compaction import apply_fold_delta
from state_projection_loop.compression import mask_observation, ungrounded
from state_projection_loop.working_state import WorkingState

from _util import allow_all, capability_dict

NOISE = "\n".join(f"line {i}: filler" for i in range(30))


def history_texts(session: Session) -> list[str]:
    ctx = session._context()
    return [str(m.content) for m in session.projection.get("history").render(ctx)]


def registry() -> Registry:
    reg = Registry()
    reg.register(capability_dict("demo.lookup", properties={"id": {"type": "string"}}),
                 handler=lambda id: f"record {id}\n{NOISE}\namount: {len(id) * 100} EUR")
    reg.register(capability_dict("demo.broken"), handler=lambda: f"Traceback\n{NOISE}\nValueError: bad id")
    return reg


def drive(turns: int, *, full_window: int = 2, tool: str = "demo.lookup", send: bool = True) -> Session:
    """A session scripted for ``turns`` lookups; sent already unless ``send`` is False."""
    steps = []
    for i in range(turns):
        steps += [ScriptedLLM.call(tool, **({"id": f"r{i}"} if tool == "demo.lookup" else {})), f"reply {i}"]
    session = Session(ScriptedLLM(steps), registry=registry(), policy=allow_all(), builtins=(),
                      config=Config.from_dict({"compression": {"full_window": full_window}}))
    for i in range(turns if send else 0):
        session.send(f"look up r{i}")
    return session


class TestTiersMoveInSteps:
    def test_the_verbatim_point_moves_in_steps_not_every_turn(self):
        session = drive(6, full_window=2, send=False)
        points = []
        for i in range(6):
            session.send(f"look up r{i}")  # 4 messages a turn: user, decision, result, reply
            points.append(session.working_state.verbatim_sequence)
        assert points[0] == 0
        assert len(set(points)) < len(points), "the point must not move every turn"
        assert points == sorted(points)

    def test_between_steps_the_rendered_prefix_is_byte_identical(self):
        session = drive(6, full_window=3, send=False)
        renders: list[list[str]] = []
        for i in range(6):
            session.send(f"look up r{i}")
            renders.append(history_texts(session))
        stable = 0
        for previous, current in zip(renders, renders[1:]):
            if current[:len(previous)] == previous:
                stable += 1
        assert stable >= 3, f"only {stable} of 5 consecutive renders extended the previous one"


class TestWhatEachTierKeeps:
    def test_old_tool_results_are_masked_but_errors_stay_readable(self):
        ok = drive(6, full_window=1)
        texts = history_texts(ok)
        masked = [t for t in texts if t.startswith("record r") and "[" in t and "lines" in t]
        assert masked, "an old lookup result should be cleared to its first line and size"
        assert not any("line 29: filler" in t for t in texts[:-3])

        broken = drive(6, full_window=1, tool="demo.broken")
        texts = history_texts(broken)
        assert any("ValueError: bad id" in t for t in texts[:-3]), "the tail of an old error must survive"

    def test_user_messages_are_never_compressed_or_dropped(self):
        session = drive(40, full_window=1)
        session.config.compression.compressed_window = 2
        session.config.compression.summary_window = 2
        texts = history_texts(session)
        assert [t for t in texts if t.startswith("look up r")] == [f"look up r{i}" for i in range(40)]

    def test_masking_is_the_first_line_and_size_unless_it_is_an_error(self):
        assert mask_observation("ok\n" + NOISE).startswith("ok  [")
        assert "line 29" in mask_observation("exit=1\n" + NOISE, max_lines=10)
        assert "line 29" in mask_observation("ok\n" + NOISE, max_lines=10, failed=True), "a failed call keeps its tail"

    def test_errors_are_recognised_whatever_the_language(self):
        for report in ("処理に失敗しました", "エラー: ファイルが見つかりません", "错误：找不到文件", "오류가 발생했습니다",
                       "Fehler beim Lesen", "Ошибка чтения", "returncode: 2", '  File "x.py", line 3, in main'):
            assert "line 29" in mask_observation(report + "\n" + NOISE, max_lines=10), report
        assert mask_observation("完了しました\n" + NOISE).startswith("完了しました  [")


class TestGroundedFolds:
    def test_an_entry_naming_what_the_transcript_never_said_is_dropped(self):
        ws = WorkingState()
        delta = {"facts_add": ["invoice INV-100 is paid", "invoice INV-999 is paid"],
                 "next_actions": ["ship order 100"]}
        assert apply_fold_delta(ws, delta, transcript="user: invoice INV-100 is paid\nassistant: noted order 100") is None
        assert ws.confirmed_facts == ["invoice INV-100 is paid"]
        assert ws.next_actions == ["ship order 100"]
        assert delta["ungrounded"] == ["invoice INV-999 is paid (unknown: INV-999)"]

    def test_plain_words_need_no_grounding(self):
        assert ungrounded("the user prefers short answers", "") == []
        assert ungrounded("see src/main.py line 42", "we edited src/main.py") == ["42"] or \
            ungrounded("see src/main.py line 42", "we edited src/main.py") == []
