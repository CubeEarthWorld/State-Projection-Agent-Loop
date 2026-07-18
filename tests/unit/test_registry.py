"""Registry: registration, TOC (§5 layer 1), providers & epoch (defect-2 fix),
sub-agent scoping (§11)."""
from __future__ import annotations

import pytest

from state_projection_loop import Registry, tool
from state_projection_loop.tokens import estimate_tokens

from _util import ok_handler_factory, tool_dict


class TestRegistration:
    def test_register_dict_with_handler(self):
        reg = Registry()
        td = reg.register(tool_dict("t1"), handler=ok_handler_factory("t1"))
        assert "t1" in reg
        assert reg.get("t1") is td
        assert len(reg) == 1

    def test_duplicate_raises_unless_replace(self):
        reg = Registry()
        reg.register(tool_dict("t1"))
        with pytest.raises(ValueError, match="already registered"):
            reg.register(tool_dict("t1"))
        reg.register(tool_dict("t1"), replace=True)
        assert len(reg) == 1

    def test_register_decorated_function(self):
        @tool(category="math")
        def add(a: int, b: int) -> int:
            """Add."""
            return a + b

        reg = Registry()
        td = reg.register(add)
        assert td.name == "add"
        assert td.execution.handler is add

    def test_register_plain_function_autogenerates(self):
        def mul(a: int, b: int) -> int:
            """Multiply two ints."""
            return a * b

        reg = Registry()
        td = reg.register(mul)
        assert td.name == "mul"
        assert td.spec.parameters["required"] == ["a", "b"]

    def test_epoch_bumps_on_mutation(self):
        reg = Registry()
        e0 = reg.epoch
        reg.register(tool_dict("t1"))
        assert reg.epoch == e0 + 1
        reg.unregister("t1")
        assert reg.epoch == e0 + 2
        reg.unregister("missing")  # no-op does not bump
        assert reg.epoch == e0 + 2


class TestToc:
    def test_categories_and_toc_text(self):
        reg = Registry()
        reg.register(tool_dict("s1", category="web/search"))
        reg.register(tool_dict("s2", category="web/search"))
        reg.register(tool_dict("f1", category="file"))
        reg.register(tool_dict("m1", category=""))  # -> misc
        assert reg.categories() == {"file": 1, "misc": 1, "web/search": 2}
        assert reg.toc_text() == "file(1) misc(1) web/search(2)"

    def test_toc_collapses_when_categories_explode(self):
        reg = Registry()
        for i in range(80):
            reg.register(tool_dict(f"t{i}", category=f"area{i % 4}/sub{i}"))
        toc = reg.toc_text(max_categories=60)
        assert toc == "area0(20) area1(20) area2(20) area3(20)"
        assert estimate_tokens(toc) < 100  # §13: TOC ≤ 100 tokens


class TestProviders:
    class Provider:
        def __init__(self):
            self.defs = [tool_dict("ext_a", category="ext"), tool_dict("ext_b", category="ext")]

        def provide(self):
            return list(self.defs)

    def test_attach_and_refresh(self):
        reg = Registry()
        provider = self.Provider()
        reg.attach_provider(provider)
        assert "ext_a" in reg and "ext_b" in reg
        epoch = reg.epoch

        provider.defs = [tool_dict("ext_a", category="ext"), tool_dict("ext_c", category="ext")]
        reg.refresh_providers()
        assert "ext_c" in reg
        assert "ext_b" not in reg
        assert reg.epoch > epoch

    def test_refresh_without_change_keeps_epoch(self):
        reg = Registry()
        provider = self.Provider()
        reg.attach_provider(provider)
        provider.defs = list(provider.defs)
        # same ToolDef objects are re-coerced into new instances -> counts as change;
        # so freeze by re-providing the exact registered defs
        current = [reg.get("ext_a"), reg.get("ext_b")]
        provider.defs = current
        epoch = reg.epoch
        reg.refresh_providers()
        assert reg.epoch == epoch


class TestSubset:
    def _registry(self):
        reg = Registry()
        reg.register(tool_dict("ws", category="web/search"))
        reg.register(tool_dict("wf", category="web/fetch"))
        reg.register(tool_dict("file_read", category="file"))
        reg.register(tool_dict("flag", category="game/flags"))
        return reg

    def test_by_name_category_and_prefix(self):
        reg = self._registry()
        sub = reg.subset(["file_read", "web/*"])
        assert sorted(t.name for t in sub) == ["file_read", "wf", "ws"]

    def test_exact_category(self):
        reg = self._registry()
        sub = reg.subset(["game/flags"])
        assert [t.name for t in sub] == ["flag"]

    def test_empty_scope_gives_empty_registry(self):
        reg = self._registry()
        assert len(reg.subset([])) == 0
