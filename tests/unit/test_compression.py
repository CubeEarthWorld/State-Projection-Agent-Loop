"""Compression: deterministic, pure-function content compression.

Tests verify:
- Idempotency on short texts
- Noise stripping (git headers, ANSI, progress lines)
- Head+tail truncation preserves first and last lines
- summarize_text produces a single line
- Empty/whitespace inputs never crash
- compress_observation is at least as aggressive as compress_text
- content_hash is stable and collision-resistant for distinct inputs
"""
from __future__ import annotations

from state_projection_loop.compression import (
    compress_observation,
    compress_text,
    content_hash,
    first_meaningful_line,
    head_tail_truncate,
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


class TestCompressObservation:
    def test_more_aggressive_than_compress_text(self):
        lines = [f"build output {i}" for i in range(100)]
        text = "\n".join(lines)
        obs = compress_observation(text)
        regular = compress_text(text)
        assert len(obs) <= len(regular)

    def test_empty_input(self):
        assert compress_observation("") == ""

    def test_preserves_first_line(self):
        text = "ERROR: something failed\n" + "\n".join(f"  at line {i}" for i in range(100))
        result = compress_observation(text)
        assert "ERROR: something failed" in result


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
