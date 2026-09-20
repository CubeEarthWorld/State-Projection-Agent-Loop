"""Bundled tool packs, and the one way to install them.

A pack is a JSON definition set (``defs/<pack>.json``, shared byte for byte
with the Dart package) plus a handler per tool. Handlers are the only part
written per language.
"""
from __future__ import annotations

from typing import Callable, Iterable

from ..registry import Registry
from .ask import ASK_HANDLERS
from .checklist import CHECKLIST_HANDLERS
from .defs import load
from .memory import MEMORY_HANDLERS
from .meta import META_HANDLERS, SPAWN_HANDLERS
from .state import STATE_HANDLERS

# Packs a bare ``Session(llm)`` installs.
DEFAULT_BUILTINS: tuple[str, ...] = ("meta", "checklist")

_PACKS: dict[str, dict[str, Callable]] = {
    "meta": META_HANDLERS,
    "checklist": CHECKLIST_HANDLERS,
    "state": STATE_HANDLERS,
    "spawn": SPAWN_HANDLERS,
    "ask": ASK_HANDLERS,
    "memory": MEMORY_HANDLERS,
}

# Every pack name `install_builtins` accepts.
BUILTIN_PACKS: tuple[str, ...] = tuple(_PACKS)


def install_builtins(registry: Registry, packs: Iterable[str]) -> None:
    """Install the named packs into ``registry``.

    Idempotent: a name the registry already resolves is left alone (a
    developer's own definition wins). A name on the registry's deny-list is
    registered but stays hidden — ``disable`` is the per-tool switch,
    ``packs`` the per-pack one.
    """
    for pack in packs:
        handlers = _PACKS.get(pack)
        if handlers is None:
            raise ValueError(f"Unknown builtin pack {pack!r}; expected one of {sorted(_PACKS)}")
        _install(registry, load(pack), handlers)


def _install(registry: Registry, definitions: list[dict], handlers: dict[str, Callable]) -> None:
    """Register each definition with its handler, unless the registry already
    resolves that name."""
    for definition in definitions:
        name = definition["name"]
        # has_definition, not `in registry`: `get()` hides a *disabled*
        # capability, so a developer's own definition that happens to be
        # switched off would otherwise look unregistered and be replaced.
        if not registry.has_definition(name):
            registry.register(definition, handler=handlers[name], replace=True)
