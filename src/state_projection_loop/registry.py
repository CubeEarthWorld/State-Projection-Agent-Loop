"""Tool registry (spec §2.3 "台帳", §5 layer 1 TOC, defect-2 fix: ToolProvider).

The registry is one of the three nouns. It owns every tool definition and
exposes:

* ``toc_text()`` — the layer-1 table of contents (category names + counts)
* ``epoch`` — bumped on any mutation so epoch-cached sections and search
  indexes know when to rebuild (cache_class="epoch")
* ``ToolProvider`` — a pluggable source of tool definitions (e.g. an MCP-like
  external server) synced via ``refresh_providers()``
* ``subset()`` — scoped views for sub-agents (§11 tool_scope)
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator, Optional, Protocol, runtime_checkable

from .tooldef import ToolDef


@runtime_checkable
class ToolProvider(Protocol):
    """External source of tool definitions (defect-2 fix, §17 reserve)."""

    def provide(self) -> Iterable[Any]:  # ToolDef | dict
        ...


class Registry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDef] = {}
        self._epoch = 0
        self._providers: list[ToolProvider] = []
        self._provider_tools: dict[int, set[str]] = {}

    # -- mutation -----------------------------------------------------------

    def register(
        self,
        tool: ToolDef | dict[str, Any] | Callable[..., Any],
        handler: Optional[Callable[..., Any]] = None,
        *,
        replace: bool = False,
    ) -> ToolDef:
        td = self._coerce(tool, handler)
        if td.name in self._tools and not replace:
            raise ValueError(f"Tool {td.name!r} is already registered (use replace=True)")
        self._tools[td.name] = td
        self._epoch += 1
        return td

    def register_many(self, tools: Iterable[Any]) -> list[ToolDef]:
        return [self.register(t) for t in tools]

    def unregister(self, name: str) -> None:
        if name in self._tools:
            del self._tools[name]
            self._epoch += 1

    @staticmethod
    def _coerce(tool: Any, handler: Optional[Callable[..., Any]] = None) -> ToolDef:
        if isinstance(tool, ToolDef):
            if handler is not None:
                tool.execution.handler = handler
            return tool
        if isinstance(tool, dict):
            return ToolDef.from_dict(tool, handler=handler)
        if callable(tool):
            td = getattr(tool, "__spal_tool__", None)
            if td is None:
                from .tooldef import build_tooldef_from_function

                td = build_tooldef_from_function(tool)
            return td
        raise TypeError(f"Cannot register {tool!r} as a tool")

    # -- providers (defect-2 fix) ------------------------------------------

    def attach_provider(self, provider: ToolProvider, *, refresh: bool = True) -> None:
        self._providers.append(provider)
        if refresh:
            self.refresh_providers()

    def refresh_providers(self) -> None:
        """Sync provider-supplied tools; adds/removes bump the epoch once."""
        changed = False
        for provider in self._providers:
            pid = id(provider)
            fresh = {td.name: td for td in (self._coerce(t) for t in provider.provide())}
            previous = self._provider_tools.get(pid, set())
            for name in previous - set(fresh):
                if name in self._tools:
                    del self._tools[name]
                    changed = True
            for name, td in fresh.items():
                if name not in self._tools or self._tools[name] is not td:
                    self._tools[name] = td
                    changed = True
            self._provider_tools[pid] = set(fresh)
        if changed:
            self._epoch += 1

    # -- lookup -------------------------------------------------------------

    @property
    def epoch(self) -> int:
        return self._epoch

    def get(self, name: str) -> Optional[ToolDef]:
        return self._tools.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[ToolDef]:
        return iter(self._tools.values())

    def all(self) -> list[ToolDef]:
        return list(self._tools.values())

    def pinned(self) -> list[ToolDef]:
        return [t for t in self._tools.values() if t.discovery.pinned]

    def categories(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for t in self._tools.values():
            cat = t.category or "misc"
            counts[cat] = counts.get(cat, 0) + 1
        return dict(sorted(counts.items()))

    def in_category(self, category: str) -> list[ToolDef]:
        return [
            t for t in self._tools.values()
            if (t.category or "misc") == category or (t.category or "").startswith(category.rstrip("/") + "/")
        ]

    # -- layer 1: table of contents (§5) ------------------------------------

    def toc_text(self, *, max_categories: int = 60) -> str:
        """Compact category index, e.g. ``web/search(2) file(12) game/flags(24)``.

        Above ``max_categories`` the index collapses to top-level categories
        only (§16: hierarchise when the TOC itself grows too large).
        """
        counts = self.categories()
        if len(counts) > max_categories:
            top: dict[str, int] = {}
            for cat, n in counts.items():
                root = cat.split("/", 1)[0]
                top[root] = top.get(root, 0) + n
            counts = dict(sorted(top.items()))
        return " ".join(f"{cat}({n})" for cat, n in counts.items())

    # -- scoped views for sub-agents (§11) -----------------------------------

    def subset(self, scope: Iterable[str]) -> "Registry":
        """New registry containing only the named tools/categories.

        Scope entries match a tool name exactly, a category exactly, or a
        category prefix written as ``"cat/*"``.
        """
        scope = list(scope)
        sub = Registry()
        for t in self._tools.values():
            cat = t.category or "misc"
            for entry in scope:
                if entry == t.name or entry == cat:
                    break
                if entry.endswith("/*") and cat.startswith(entry[:-1]):
                    break
            else:
                continue
            sub._tools[t.name] = t
        sub._epoch = 1
        return sub
