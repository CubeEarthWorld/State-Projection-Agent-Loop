"""Compression: deterministic, pure-function content compression.

Tests verify:
- Idempotency on short texts
- Noise stripping (git headers, ANSI, progress lines)
- Head+tail truncation preserves first and last lines
- summarize_text produces a single line
- Empty/whitespace inputs never crash
- content_hash is stable and collision-resistant for distinct inputs
"""
from __future__ import annotations

from state_projection_loop.compression import (
    compress_text,
    content_hash,
    first_meaningful_line,
    head_tail_truncate,
    mask_observation,
    strip_noise,
    summarize_text,
)


class TestStripNoise:
    def test_removes_git_diff_headers(self):
        text = "diff --git a/foo.py b/foo.py\nindex abc123..def456 100644\n--- a/foo.py\n+++ b/foo.py\n@@ -1,3 +1,4 @@\n+new line\n"
        result = strip_noise(text)
        assert "diff --git" not in result
        assert "index abc" not in result
        assert "--- a/" not in result
        assert "+++ b/" not in result
        assert "@@" not in result
        assert "+new line" in result

    def test_removes_ansi_escape_codes(self):
        text = "\x1b[32mgreen\x1b[0m normal \x1b[1;34mblue\x1b[0m"
        result = strip_noise(text)
        assert "\x1b" not in result
        assert "green" in result
        assert "normal" in result
        assert "blue" in result

    def test_collapses_consecutive_blank_lines(self):
        text = "line1\n\n\n\n\nline2\n"
        result = strip_noise(text)
        assert "\n\n\n" not in result
        assert "line1" in result
        assert "line2" in result

    def test_removes_node_modules_paths(self):
        text = "src/main.py\nnode_modules/foo/bar.js\n.venv/lib/site.py\n__pycache__/mod.cpython-313.pyc\nsrc/util.py\n"
        result = strip_noise(text)
        assert "node_modules" not in result
        assert ".venv" not in result
        assert "__pycache__" not in result
        assert "src/main.py" in result
        assert "src/util.py" in result

    def test_removes_progress_and_download_lines(self):
        text = "Progress: 50%\r[1/10] Downloading package foo\n[2/10] Installing bar\nactual content\n"
        result = strip_noise(text)
        assert "Progress:" not in result
        assert "Downloading" not in result
        assert "Installing" not in result
        assert "actual content" in result

    def test_empty_input(self):
        assert strip_noise("") == ""

    def test_no_noise_unchanged(self):
        text = "def hello():\n    return 42\n"
        assert strip_noise(text) == text


class TestHeadTailTruncate:
    def test_short_text_unchanged(self):
        text = "line1\nline2\nline3\n"
        assert head_tail_truncate(text, max_lines=10) == text

    def test_exact_limit_unchanged(self):
        lines = [f"line{i}\n" for i in range(10)]
        text = "".join(lines)
        assert head_tail_truncate(text, max_lines=10) == text

    def test_truncation_preserves_head_and_tail(self):
        lines = [f"line{i}\n" for i in range(100)]
        text = "".join(lines)
        result = head_tail_truncate(text, max_lines=20)
        assert "line0\n" in result
        assert "line1\n" in result
        assert "line99\n" in result
        assert "line98\n" in result
        assert "omitted" in result

    def test_truncation_never_empty_for_nonempty_input(self):
        text = "\n".join(f"line{i}" for i in range(200))
        result = head_tail_truncate(text, max_lines=5)
        assert result.strip()

    def test_single_line(self):
        assert head_tail_truncate("hello\n", max_lines=1) == "hello\n"


class TestCompressText:
    def test_short_text_idempotent(self):
        text = "def foo():\n    return 1\n"
        once = compress_text(text)
        twice = compress_text(once)
        assert once == twice

    def test_never_empty_for_nonempty_input(self):
        text = "\n\n\n"
        result = compress_text(text)
        assert result is not None

    def test_empty_input(self):
        assert compress_text("") == ""

    def test_long_output_truncated(self):
        lines = [f"output line {i}" for i in range(200)]
        text = "\n".join(lines)
        result = compress_text(text, max_lines=20)
        result_lines = result.splitlines()
        assert len(result_lines) <= 25

    def test_noise_stripped_before_truncation(self):
        noise = "diff --git a/x b/x\nindex 123..456 100644\n"
        content = "\n".join(f"real line {i}" for i in range(100))
        text = noise + content
        result = compress_text(text, max_lines=20)
        assert "diff --git" not in result
        assert "real line 0" in result


class TestObservationCompression:
    """Observations are compressed by the same function, with the tighter
    line budget the config gives them."""

    def test_a_tighter_line_budget_produces_less(self):
        text = "\n".join(f"build output {i}" for i in range(100))
        assert len(compress_text(text, max_lines=40)) <= len(compress_text(text, max_lines=80))

    def test_empty_input(self):
        assert compress_text("", max_lines=40) == ""

    def test_preserves_first_line(self):
        text = "ERROR: something failed\n" + "\n".join(f"  at line {i}" for i in range(100))
        assert "ERROR: something failed" in compress_text(text, max_lines=40)


class TestSummarizeText:
    def test_single_short_line_unchanged(self):
        text = "All tests passed."
        assert summarize_text(text) == "All tests passed."

    def test_multiline_produces_single_line(self):
        text = "\n".join(f"line {i}" for i in range(50))
        result = summarize_text(text)
        assert "\n" not in result
        assert "50 lines" in result

    def test_empty_input(self):
        assert summarize_text("") == ""

    def test_skips_comment_lines(self):
        text = "# comment\n// another comment\ndef real_code():\n    pass\n"
        result = summarize_text(text)
        assert "real_code" in result

    def test_long_first_line_truncated(self):
        text = "x" * 200 + "\nline2\nline3\n"
        result = summarize_text(text)
        assert len(result) < 200
        assert "…" in result


class TestFirstMeaningfulLine:
    def test_skips_comments(self):
        text = "# header\n/* block */\nactual content\n"
        assert first_meaningful_line(text) == "actual content"

    def test_empty_text(self):
        assert first_meaningful_line("") == ""

    def test_all_comments_returns_first(self):
        text = "# only comments\n# more comments\n"
        result = first_meaningful_line(text)
        assert result == "# only comments"


class TestContentHash:
    def test_stable(self):
        text = "hello world"
        assert content_hash(text) == content_hash(text)

    def test_distinct_inputs_differ(self):
        assert content_hash("hello") != content_hash("world")

    def test_length_is_16(self):
        assert len(content_hash("test")) == 16

    def test_empty_string(self):
        h = content_hash("")
        assert len(h) == 16


_NOISE = "\n".join(f"line {i}" for i in range(30))


class TestRegressions:
    """One case per fixed bug; the Dart port carries the same set."""

    def test_an_http_status_is_not_an_error(self):
        # "status: 2" of "status: 200" used to match the exit-code pattern,
        # so a tool reporting an HTTP status was never compressed again.
        assert mask_observation("HTTP status: 200 OK\n" + _NOISE) == "HTTP status: 200 OK  [31 lines, 249 chars]"

    def test_a_clock_time_is_not_a_stack_frame(self):
        assert mask_observation("Meeting at 14:30 with Bob\n" + _NOISE).startswith("Meeting at 14:30 with Bob  [")

    def test_a_real_failure_still_takes_the_error_path(self):
        for report in ("exit code 1", "exit status: 2", "returncode=127", "  at foo.js:12"):
            assert "line 29" in mask_observation(report + "\n" + _NOISE, max_lines=10), report
        assert mask_observation("exit code 0\n" + _NOISE).startswith("exit code 0  ["), "a clean exit is not an error"

    def test_truncation_never_lengthens_the_text(self):
        text = "\n".join("a" for _ in range(41)) + "\n"  # 41 one-char lines
        assert head_tail_truncate(text, max_lines=40) == text

    def test_output_never_has_more_lines_than_the_limit(self):
        text = "".join("y" * 40 + "\n" for _ in range(20))
        assert len(head_tail_truncate(text, max_lines=2).splitlines()) <= 2

    def test_noise_stripping_handles_crlf(self):
        assert strip_noise("diff --git a/f b/f\r\nindex 111..222 100644\r\nrest\r\n") == "rest\r\n"

    def test_the_fallback_first_line_keeps_its_terminator(self):
        assert compress_text("diff --git a/x b/x\n") == "diff --git a/x b/x\n"

    def test_a_lone_carriage_return_separates_lines(self):
        assert first_meaningful_line("#x\ry") == "y"

    def test_every_python_line_break_splits(self):
        text = "\v".join("x" * 20 for _ in range(20))
        assert head_tail_truncate(text, max_lines=4) == (
            "x" * 20 + "\v" + "x" * 20 + "\v  [... 17 lines omitted ...]\n" + "x" * 20
        )

    def test_an_unpaired_surrogate_hashes_the_same_in_both_ports(self):
        # Python encodes with errors="replace", which emits "?" — the Dart
        # port has to match, or the two dedupe differently on the same text.
        assert content_hash("\ud800") == content_hash("?")
        assert content_hash("a\ud800b") == content_hash("a?b")
        # A well-formed astral character is untouched by that substitution.
        assert content_hash("\U0001F38C") != content_hash("?")
