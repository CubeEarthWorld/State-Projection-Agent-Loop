"""Bundled capability definitions, as data.

The definitions live in JSON, not in Python or Dart literals, because this
package is developed alongside a Dart port and every definition used to be
written out by hand twice. The same files are copied into the Dart
repository under ``spec/tools/`` and compiled into a string constant there,
so a definition can only be changed in one place.

Handlers stay in code — they are the part that genuinely differs per
language.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def _load(name: str) -> str:
    return (_DIR / f"{name}.json").read_text(encoding="utf-8")


def load(name: str) -> Any:
    """Return a fresh copy of the named definition set."""
    return json.loads(_load(name))
