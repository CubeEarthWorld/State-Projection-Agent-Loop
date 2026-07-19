"""Capability registry (layer 1 TOC, ToolProvider sync).

One of the three nouns. Owns every :class:`Capability` and exposes:

* ``toc_text()`` — the layer-1 table of contents (category names + counts)
* ``epoch`` — bumped on any mutation so epoch-cached sections and search
  indexes know when to rebuild (cache_class="epoch")
* ``ToolProvider`` — a pluggable source of capabilities (e.g. an MCP-like
  external server) synced via ``refresh_providers()``
* ``subset()`` — scoped views for sub-agents (spawn tool_scope)

Capabilities are versioned (``name@version``); ``get(name)`` without a
version resolves to the highest registered version, so a call site written
against the bare name always gets the latest contract without touching
config.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Iterator, Optional, Protocol, runtime_checkable

from .capability import Capability, from_api_name


@runtime_checkable
class ToolProvider(Protocol):
    """External source of capability definitions."""

    def provide(self) -> Iterable[Any]:  # Capability | dict
        ...


class Registry:
    def __init__(self) -> None:
        self._capabilities: dict[str, Capability] = {}  # keyed by qualified_name
        self._latest: dict[str, str] = {}  # name -> qualified_name of highest version
        self._epoch = 0
        self._providers: list[ToolProvider] = []
        self._provider_tools: dict[int, set[str]] = {}

    # -- mutation -------------------------------------------------------------

    def register(
        self,
        capability: Capability | dict[str, Any] | Callable[..., Any],
        handler: Optional[Callable[..., Any]] = None,
        *,
        replace: bool = False,
    ) -> Capability:
        cap = self._coerce(capability, handler)
        if cap.qualified_name in self._capabilities and not replace:
            raise ValueError(f"Capability {cap.qualified_name!r} is already registered (use replace=True)")
        self._capabilities[cap.qualified_name] = cap
        current = self._latest.get(cap.name)
        if current is None or self._capabilities[current].version < cap.version:
            self._latest[cap.name] = cap.qualified_name
        self._epoch += 1
        return cap

    def register_many(self, capabilities: Iterable[Any]) -> list[Capability]:
        return [self.register(c) for c in capabilities]

    def unregister(self, name: str) -> None:
        """Remove by bare name (all versions) or exact ``name@version``."""
        if name in self._capabilities:
            del self._capabilities[name]
            self._recompute_latest()
            self._epoch += 1
            return
        removed = [q for q in self._capabilities if q.rsplit("@", 1)[0] == name]
        if removed:
            for q in removed:
                del self._capabilities[q]
            self._recompute_latest()
            self._epoch += 1

    def _recompute_latest(self) -> None:
        self._latest = {}
        for cap in self._capabilities.values():
            current = self._latest.get(cap.name)
            if current is None or self._capabilities[current].version < cap.version:
                self._latest[cap.name] = cap.qualified_name

    @staticmethod
    def _coerce(capability: Any, handler: Optional[Callable[..., Any]] = None) -> Capability:
        if isinstance(capability, Capability):
            if handler is not None:
                capability.execution.handler = handler
            return capability
        if isinstance(capability, dict):
            return Capability.from_dict(capability, handler=handler)
        if callable(capability):
            cap = getattr(capability, "__spal_capability__", None)
            if cap is None:
                from .capability import build_capability_from_function

                cap = build_capability_from_function(capability)
            return cap
        raise TypeError(f"Cannot register {capability!r} as a capability")

    # -- providers ------------------------------------------------------------

    def attach_provider(self, provider: ToolProvider, *, refresh: bool = True) -> None:
        self._providers.append(provider)
        if refresh:
            self.refresh_providers()

    def refresh_providers(self) -> None:
        """Sync provider-supplied capabilities; adds/removes bump the epoch once."""
        changed = False
        for provider in self._providers:
            pid = id(provider)
            fresh = {cap.qualified_name: cap for cap in (self._coerce(c) for c in provider.provide())}
            previous = self._provider_tools.get(pid, set())
            for qname in previous - set(fresh):
                if qname in self._capabilities:
                    del self._capabilities[qname]
                    changed = True
            for qname, cap in fresh.items():
                if qname not in self._capabilities or self._capabilities[qname] is not cap:
                    self._capabilities[qname] = cap
                    changed = True
            self._provider_tools[pid] = set(fresh)
        if changed:
            self._recompute_latest()
            self._epoch += 1

    # -- lookup ---------------------------------------------------------------

    @property
    def epoch(self) -> int:
        return self._epoch

    def get(self, name: str) -> Optional[Capability]:
        """Resolve by ``name@version`` (exact) or bare ``name`` (latest)."""
        if name in self._capabilities:
            return self._capabilities[name]
        qname = self._latest.get(name)
        return self._capabilities.get(qname) if qname else None

    def resolve_api_name(self, name: str) -> str:
        """Translate a provider-safe ``api_name`` (see ``Capability.api_name``)
        back to the registered dotted name, if it resolves to one.

        Names that already resolve directly (a bare or qualified dotted
        name) are returned unchanged; a name that doesn't resolve even
        after decoding is also returned unchanged, so the normal
        "unknown capability" error path still reports the name the model
        actually sent.
        """
        if name in self._capabilities or name in self._latest:
            return name
        dotted = from_api_name(name)
        if dotted in self._capabilities or dotted in self._latest:
            return dotted
        return name

    def __contains__(self, name: str) -> bool:
        return self.get(name) is not None

    def __len__(self) -> int:
        return len(self._latest)

    def __iter__(self) -> Iterator[Capability]:
        return (self._capabilities[q] for q in self._latest.values())

    def all(self) -> list[Capability]:
        return list(self)

    def pinned(self) -> list[Capability]:
        return [c for c in self if c.discovery.pinned]

    def categories(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in self:
            cat = c.category or "misc"
            counts[cat] = counts.get(cat, 0) + 1
        return dict(sorted(counts.items()))

    def categories_with_pinned(self) -> dict[str, tuple[int, int]]:
        totals: dict[str, int] = {}
        pinned: dict[str, int] = {}
        for c in self:
            cat = c.category or "misc"
            totals[cat] = totals.get(cat, 0) + 1
            if c.discovery.pinned:
                pinned[cat] = pinned.get(cat, 0) + 1
        return {cat: (totals[cat], pinned.get(cat, 0)) for cat in sorted(totals)}

    def in_category(self, category: str) -> list[Capability]:
        return [
            c for c in self
            if (c.category or "misc") == category or (c.category or "").startswith(category.rstrip("/") + "/")
        ]

    # -- layer 1: table of contents -------------------------------------------

    def toc_text(self, *, max_categories: int = 60) -> str:
        """Compact category index with pinned counts, e.g.::

            meta(2p) game/media(3) file(2, 1p)

        Above ``max_categories`` the index collapses to top-level categories
        only (hierarchise when the TOC itself grows too large).
        """
        cat_info = self.categories_with_pinned()
        if len(cat_info) > max_categories:
            top_totals: dict[str, int] = {}
            top_pinned: dict[str, int] = {}
            for cat, (total, p) in cat_info.items():
                root = cat.split("/", 1)[0]
                top_totals[root] = top_totals.get(root, 0) + total
                top_pinned[root] = top_pinned.get(root, 0) + p
            cat_info = {cat: (top_totals[cat], top_pinned.get(cat, 0))
                        for cat in sorted(top_totals)}

        parts: list[str] = []
        for cat, (total, p) in cat_info.items():
            if p == total and total > 0:
                parts.append(f"{cat}({total}p)")
            elif p > 0:
                parts.append(f"{cat}({total}, {p}p)")
            else:
                parts.append(f"{cat}({total})")
        return " ".join(parts)

    # -- scoped views for sub-agents -------------------------------------------

    def subset(self, scope: Iterable[str]) -> "Registry":
        """New registry containing only the named capabilities/categories.

        Scope entries match a capability name exactly, a category exactly,
        or a category prefix written as ``"cat/*"``.
        """
        scope = list(scope)
        sub = Registry()
        for c in self:
            cat = c.category or "misc"
            for entry in scope:
                if entry == c.name or entry == cat:
                    break
                if entry.endswith("/*") and cat.startswith(entry[:-1]):
                    break
            else:
                continue
            sub._capabilities[c.qualified_name] = c
        sub._recompute_latest()
        sub._epoch = 1
        return sub
