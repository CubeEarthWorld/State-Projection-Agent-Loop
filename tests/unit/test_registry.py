"""Registry: registration, TOC (layer 1), providers & epoch, versioned
lookup, sub-agent scoping."""
from __future__ import annotations

import pytest

from state_projection_loop import Registry, capability
from state_projection_loop.tokens import estimate_tokens

from _util import capability_dict, ok_handler_factory


class TestRegistration:
    def test_register_dict_with_handler(self):
        reg = Registry()
        cap = reg.register(capability_dict("demo.t1"), handler=ok_handler_factory("t1"))
        assert "demo.t1" in reg
        assert reg.get("demo.t1") is cap
        assert len(reg) == 1

    def test_duplicate_raises_unless_replace(self):
        reg = Registry()
        reg.register(capability_dict("demo.t1"))
        with pytest.raises(ValueError, match="already registered"):
            reg.register(capability_dict("demo.t1"))
        reg.register(capability_dict("demo.t1"), replace=True)
        assert len(reg) == 1

    def test_register_decorated_function(self):
        @capability(name="math.add", category="math")
        def add(a: int, b: int) -> int:
            """Add."""
            return a + b

        reg = Registry()
        cap = reg.register(add)
        assert cap.name == "math.add"
        assert cap.execution.handler is add

    def test_register_plain_function_autogenerates(self):
        def mul_two(a: int, b: int) -> int:
            """Multiply two ints."""
            return a * b

        reg = Registry()
        cap = reg.register(mul_two)
        assert cap.name == "mul.two"  # underscores become dots by default
        assert cap.spec.parameters["required"] == ["a", "b"]

    def test_epoch_bumps_on_mutation(self):
        reg = Registry()
        e0 = reg.epoch
        reg.register(capability_dict("demo.t1"))
        assert reg.epoch == e0 + 1
        reg.unregister("demo.t1")
        assert reg.epoch == e0 + 2
        reg.unregister("demo.missing")  # no-op does not bump
        assert reg.epoch == e0 + 2

    def test_invalid_name_rejected(self):
        bad = capability_dict("demo.t1")
        bad["name"] = "singleword"
        with pytest.raises(ValueError, match="dotted"):
            Registry().register(bad)


class TestVersioning:
    def test_bare_name_resolves_to_latest(self):
        reg = Registry()
        reg.register({**capability_dict("demo.thing"), "version": 1})
        reg.register({**capability_dict("demo.thing"), "version": 2})
        assert reg.get("demo.thing").version == 2
        assert reg.get("demo.thing@1").version == 1
        assert len(reg) == 1  # counts distinct names, not versions

    def test_unregister_by_bare_name_drops_all_versions(self):
        reg = Registry()
        reg.register({**capability_dict("demo.thing"), "version": 1})
        reg.register({**capability_dict("demo.thing"), "version": 2})
        reg.unregister("demo.thing")
        assert reg.get("demo.thing") is None
        assert reg.get("demo.thing@1") is None


class TestToc:
    def test_categories_and_toc_text(self):
        reg = Registry()
        reg.register(capability_dict("web.search.s1", category="web/search"))
        reg.register(capability_dict("web.search.s2", category="web/search"))
        reg.register(capability_dict("file.f1", category="file"))
        reg.register(capability_dict("misc.m1", category=""))  # -> misc
        assert reg.categories() == {"file": 1, "misc": 1, "web/search": 2}
        assert reg.toc_text() == "file(1) misc(1) web/search(2)"

    def test_toc_collapses_when_categories_explode(self):
        reg = Registry()
        for i in range(80):
            reg.register(capability_dict(f"area{i % 4}.sub.t{i}", category=f"area{i % 4}/sub{i}"))
        toc = reg.toc_text(max_categories=60)
        assert toc == "area0(20) area1(20) area2(20) area3(20)"
        assert estimate_tokens(toc) < 100


class TestProviders:
    class Provider:
        def __init__(self):
            self.defs = [capability_dict("ext.a", category="ext"), capability_dict("ext.b", category="ext")]

        def provide(self):
            return list(self.defs)

    def test_attach_and_refresh(self):
        reg = Registry()
        provider = self.Provider()
        reg.attach_provider(provider)
        assert "ext.a" in reg and "ext.b" in reg
        epoch = reg.epoch

        provider.defs = [capability_dict("ext.a", category="ext"), capability_dict("ext.c", category="ext")]
        reg.refresh_providers()
        assert "ext.c" in reg
        assert "ext.b" not in reg
        assert reg.epoch > epoch

    def test_refresh_without_change_keeps_epoch(self):
        reg = Registry()
        provider = self.Provider()
        reg.attach_provider(provider)
        current = [reg.get("ext.a"), reg.get("ext.b")]
        provider.defs = current
        epoch = reg.epoch
        reg.refresh_providers()
        assert reg.epoch == epoch


class TestSubset:
    def _registry(self):
        reg = Registry()
        reg.register(capability_dict("web.search.query", category="web/search"))
        reg.register(capability_dict("web.fetch.url", category="web/fetch"))
        reg.register(capability_dict("file.read", category="file"))
        reg.register(capability_dict("game.flags.set", category="game/flags"))
        return reg

    def test_by_name_category_and_prefix(self):
        reg = self._registry()
        sub = reg.subset(["file.read", "web/*"])
        assert sorted(c.name for c in sub) == ["file.read", "web.fetch.url", "web.search.query"]

    def test_exact_category(self):
        reg = self._registry()
        sub = reg.subset(["game/flags"])
        assert [c.name for c in sub] == ["game.flags.set"]

    def test_empty_scope_gives_empty_registry(self):
        reg = self._registry()
        assert len(reg.subset([])) == 0
