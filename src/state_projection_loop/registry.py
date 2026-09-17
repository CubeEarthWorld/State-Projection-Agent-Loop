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


def scope_matches(entry: str, name: str, category: str) -> bool:
    """One scope entry against one capability.

    An entry matches a capability name exactly, a category exactly, or a
    category prefix written as ``"cat/*"``. Shared by ``subset()`` (an
    allow-list for sub-agents) and ``disable()`` (a deny-list).
    """
    if entry == name or entry == category:
        return True
    return entry.endswith("/*") and category.startswith(entry[:-1])


class Registry:
    """Every capability in the system, and the single gate the model sees it through.

    Both ``__iter__`` and ``get()`` skip disabled capabilities, and every
    other surface — the TOC, pinned specs, native tool schemas, layer-2
    candidates, ``meta.tool.find``, and execution — derives from those two.
    Disabling therefore removes a capability from all of them at once.
    """

    def __init__(self, *, disabled: Iterable[str] = ()) -> None:
        self._capabilities: dict[str, Capability] = {}  # keyed by qualified_name
        self._latest: dict[str, str] = {}  # name -> qualified_name of highest version
        self._epoch = 0
        self._providers: list[ToolProvider] = []
        self._provider_tools: dict[int, set[str]] = {}
        # A deny-list of names/categories, not of registered objects: a
        # disabled name stays disabled however it is registered afterwards,
        # so a bundled pack installed later (install_builtins) cannot sneak
        # it back in.
        self._disabled: set[str] = set(disabled)
        self._pinned_cache: tuple[int, list[Capability]] = (-1, [])

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

    # -- disabling ------------------------------------------------------------

    @property
    def disabled(self) -> frozenset[str]:
        return frozenset(self._disabled)

    def disable(self, *names: str) -> None:
        """Hide capabilities from the model entirely.

        Entries follow :func:`scope_matches`. A disabled capability is gone
        from the tool index, pinned specs, native schemas, search and
        execution alike — the model can neither see it nor call it.
        """
        added = set(names) - self._disabled
        if added:
            self._disabled |= added
            self._epoch += 1

    def enable(self, *names: str) -> None:
        """Undo :meth:`disable` for the given entries."""
        removed = self._disabled & set(names)
        if removed:
            self._disabled -= removed
            self._epoch += 1

    @staticmethod
    def _category_of(capability: Capability) -> str:
        return capability.category or "misc"

    def _is_disabled(self, capability: Capability) -> bool:
        if not self._disabled:
            return False
        return any(scope_matches(e, capability.name, self._category_of(capability)) for e in self._disabled)

    # -- lookup ---------------------------------------------------------------

    @property
    def epoch(self) -> int:
        return self._epoch

    def get(self, name: str) -> Optional[Capability]:
        """Resolve by ``name@version`` (exact) or bare ``name`` (latest).

        Disabled capabilities resolve to ``None``, exactly like unregistered
        ones — that is what makes them unreachable from the runtime.
        """
        capability = self._capabilities.get(name)
        if capability is None:
            qname = self._latest.get(name)
            capability = self._capabilities.get(qname) if qname else None
        if capability is None or self._is_disabled(capability):
            return None
        return capability

    def resolve_api_name(self, name: str) -> str:
        """Translate a provider-safe ``api_name`` (see ``Capability.api_name``)
        back to the registered dotted name, if it resolves to one.

        Names that already resolve directly (a bare or qualified dotted
        name) are returned unchanged; a name that doesn't resolve even
        after decoding is also returned unchanged, so the normal
        "unknown capability" error path still reports the name the model
        actually sent.
        """
        if self.get(name) is not None:
            return name
        dotted = from_api_name(name)
        if self.get(dotted) is not None:
            return dotted
        return name

    def __contains__(self, name: str) -> bool:
        return self.get(name) is not None

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __iter__(self) -> Iterator[Capability]:
        return (
            c for c in (self._capabilities[q] for q in self._latest.values())
            if not self._is_disabled(c)
        )

    def all(self) -> list[Capability]:
        return list(self)

    def pinned(self) -> list[Capability]:
        """Cached by epoch: this is read several times per turn (native
        schemas, the kernel section, the layer-2 exclusion set)."""
        epoch, cached = self._pinned_cache
        if epoch != self._epoch:
            cached = [c for c in self if c.discovery.pinned]
            self._pinned_cache = (self._epoch, cached)
        return cached

    def categories(self) -> dict[str, tuple[int, int]]:
        """Sorted ``category -> (total, pinned)`` counts."""
        totals: dict[str, int] = {}
        pinned: dict[str, int] = {}
        for c in self:
            cat = self._category_of(c)
            totals[cat] = totals.get(cat, 0) + 1
            if c.discovery.pinned:
                pinned[cat] = pinned.get(cat, 0) + 1
        return {cat: (totals[cat], pinned.get(cat, 0)) for cat in sorted(totals)}

    # -- layer 1: table of contents -------------------------------------------

    def toc_text(self, *, max_categories: int = 60) -> str:
        """Compact category index with pinned counts, e.g.::

            meta(2p) game/media(3) file(2, 1p)

        Above ``max_categories`` the index collapses to top-level categories
        only (hierarchise when the TOC itself grows too large).
        """
        cat_info = self.categories()
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

        Scope entries follow :func:`scope_matches`. Capabilities disabled
        here are already invisible to the iteration, and the deny-list
        carries over so they stay disabled in the child.
        """
        scope = list(scope)
        sub = Registry(disabled=self._disabled)
        for c in self:
            if any(scope_matches(entry, c.name, self._category_of(c)) for entry in scope):
                sub._capabilities[c.qualified_name] = c
        sub._recompute_latest()
        sub._epoch = 1
        return sub
