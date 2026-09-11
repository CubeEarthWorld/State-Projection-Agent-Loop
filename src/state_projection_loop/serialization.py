"""The one way this package turns a value into JSON.

Every JSON string the model sees or the ledger stores goes through here:
tool schemas, capability specs, artifact bodies, the working-state ``extra``
line, ledger rows. Having a single definition is what keeps the Python and
Dart ports byte-identical — the two drifted apart on separator spacing
alone, which silently changed every token estimate.

* compact separators — the spaces in Python's default ``", "``/``": "``
  cost tokens and buy nothing
* ``ensure_ascii=False`` — a Japanese description should not become
  ``\\uXXXX`` escapes, which cost several tokens per character
* unencodable values fall back to ``str`` rather than raising: ``extra`` is
  documented as a free-form escape hatch, and a ledger append must not fail
  because something in it was not JSON
"""
from __future__ import annotations

import json
from typing import Any

_SEPARATORS = (",", ":")


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=_SEPARATORS, default=str)
