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

import copy
from typing import Any, Callable, Iterable, Iterator, Optional, Protocol

from .capability import Capability, build_capability_from_function, from_api_name


class ToolProvider(Protocol):
    """External source of capability definitions."""

    def provide(self) -> Iterable[Any]:  # Capability | dict
        ...


def scope_matches(entry: str, name: str, category: str) -> bool:
    """One scope entry against one capability.

    An entry matches a capability name exactly, a category exactly, the
    wildcard ``"*"`` (everything), or a category prefix written as
    ``"cat/*"`` — which covers the category itself as well as every
    sub-category under it, so ``"meta/*"`` is never an empty scope while
    ``"meta"`` is not. Shared by ``subset()`` (an allow-list for sub-agents)
    and ``disable()`` (a deny-list).
    """
    if entry == "*" or entry == name or entry == category:
        return True
    if not entry.endswith("/*"):
        return False
    prefix = entry[:-2]
    return category == prefix or category.startswith(prefix + "/")


def _copy_capability(cap: Capability) -> Capability:
    """An independent copy sharing nothing a registry mutates.

    ``@capability`` caches one :class:`Capability` on the decorated function
    and ``subset()`` hands the parent's objects to the child, so without this
    two registries would share one object — and attaching a handler in one
    would silently rewire the other.
    """
    new = copy.copy(cap)
    new.card = copy.copy(cap.card)
    new.spec = copy.copy(cap.spec)
    new.discovery = copy.copy(cap.discovery)
    new.execution = copy.copy(cap.execution)
    new.effects = list(cap.effects)
    return new


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
        # Qualified names that came in through register(), not a provider:
        # refresh_providers() must never delete one of these.
        self._hand_registered: set[str] = set()
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
        cap = self._coerce(capability, handler, copy_shared=True)
        if cap.qualified_name in self._capabilities and not replace:
            raise ValueError(f"Capability {cap.qualified_name!r} is already registered (use replace=True)")
        self._capabilities[cap.qualified_name] = cap
        self._hand_registered.add(cap.qualified_name)
        self._track_latest(cap)
        self._epoch += 1
        return cap

    def unregister(self, name: str) -> None:
        """Remove by bare name (all versions) or exact ``name@version``."""
        if name in self._capabilities:
            del self._capabilities[name]
            self._hand_registered.discard(name)
            self._recompute_latest()
            self._epoch += 1
            return
        removed = [q for q in self._capabilities if q.rsplit("@", 1)[0] == name]
        if removed:
            for q in removed:
                del self._capabilities[q]
                self._hand_registered.discard(q)
            self._recompute_latest()
            self._epoch += 1

    def _recompute_latest(self) -> None:
        self._latest = {}
        for cap in self._capabilities.values():
            self._track_latest(cap)

    def _track_latest(self, cap: Capability) -> None:
        current = self._latest.get(cap.name)
        if current is None or self._capabilities[current].version < cap.version:
            self._latest[cap.name] = cap.qualified_name

    @staticmethod
    def _coerce(capability: Any, handler: Optional[Callable[..., Any]] = None,
                *, copy_shared: bool = False) -> Capability:
        """Normalise to a Capability, honouring ``handler`` whatever form the
        definition took.

        ``copy_shared`` copies an already-built Capability (a bare object, or
        the one ``@capability`` caches on the function) so the registry owns
        it outright; ``refresh_providers`` leaves it off, because it compares
        provider output by identity to decide whether anything changed.
        """
        if isinstance(capability, dict):
            return Capability.from_dict(capability, handler=handler)
        if isinstance(capability, Capability):
            cap = capability
        elif callable(capability):
            cap = getattr(capability, "__spal_capability__", None)
            if cap is None:
                # Freshly built for this call: already unshared.
                cap = build_capability_from_function(capability)
                copy_shared = False
        else:
            raise TypeError(f"Cannot register {capability!r} as a capability")
        if copy_shared:
            cap = _copy_capability(cap)
        if handler is not None:
            cap.execution.handler = handler
        return cap

    # -- providers ------------------------------------------------------------

    def attach_provider(self, provider: ToolProvider, *, refresh: bool = True) -> None:
        self._providers.append(provider)
        if refresh:
            self.refresh_providers()

    def refresh_providers(self) -> None:
        """Sync provider-supplied capabilities; adds/removes bump the epoch once.

        A name one provider stops offering only goes away when *nothing*
        still provides it: the removal set is per-provider but the registry
        is shared, so deleting on the per-provider set alone made the
        outcome depend on the order the providers were attached in.
        """
        changed = False
        fresh_by_pid = {
            id(provider): {c.qualified_name: c for c in (self._coerce(x) for x in provider.provide())}
            for provider in self._providers
        }
        still_offered = {q for fresh in fresh_by_pid.values() for q in fresh}
        for pid, fresh in fresh_by_pid.items():
            for qname in self._provider_tools.get(pid, set()) - still_offered:
                if qname in self._capabilities and qname not in self._hand_registered:
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

    def has_definition(self, name: str) -> bool:
        """Is the slot taken, disabled or not?

        ``__contains__`` answers "can the model reach it", which is ``False``
        for a disabled capability. An installer needs this question instead,
        or a disabled name looks unregistered and gets overwritten.
        """
        return name in self._capabilities or name in self._latest

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

    @staticmethod
    def _sorted_counts(totals: dict[str, int], pinned: dict[str, int]) -> dict[str, tuple[int, int]]:
        return {cat: (totals[cat], pinned.get(cat, 0)) for cat in sorted(totals)}

    def categories(self) -> dict[str, tuple[int, int]]:
        """Sorted ``category -> (total, pinned)`` counts."""
        totals: dict[str, int] = {}
        pinned: dict[str, int] = {}
        for c in self:
            cat = self._category_of(c)
            totals[cat] = totals.get(cat, 0) + 1
            if c.discovery.pinned:
                pinned[cat] = pinned.get(cat, 0) + 1
        return self._sorted_counts(totals, pinned)

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
            cat_info = self._sorted_counts(top_totals, top_pinned)

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
