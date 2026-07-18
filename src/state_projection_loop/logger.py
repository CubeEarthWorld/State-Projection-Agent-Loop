"""Structured session logging (spec §8.1): every projection, decision and
execution result is recorded machine-readably for replay and debugging.

Events are kept in memory (``events``) and, when a path is configured,
appended as JSON Lines.
"""
from __future__ import annotations

import json
import time
from typing import Any, Optional


class SessionLogger:
    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self.events: list[dict[str, Any]] = []

    def log(self, event: str, **data: Any) -> None:
        record = {"ts": time.time(), "event": event, **data}
        self.events.append(record)
        if self.path:
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            except OSError:
                pass  # logging must never break the loop

    def of_type(self, event: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["event"] == event]
