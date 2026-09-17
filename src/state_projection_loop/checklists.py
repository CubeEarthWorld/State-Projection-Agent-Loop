"""Versioned, JSON-portable plans. Mutations validate a copy before committing."""
from __future__ import annotations

import copy
import re
from typing import Any

from .ids import new_ulid
from .serialization import dumps

STATUSES = ("pending", "in_progress", "blocked", "completed", "cancelled")
CONTEXT_MODES = ("name", "summary", "full")
_ID = re.compile(r"[0-7][0-9A-HJKMNP-TV-Z]{25}\Z")


def _text(value: Any, field: str, limit: int, *, empty: bool = False) -> str:
    if not isinstance(value, str) or len(value) > limit or (not empty and not value.strip()):
        raise ValueError(f"{field} must be a {'possibly empty ' if empty else 'nonempty '}string of at most {limit} characters")
    return value


def _keys(data: Any, allowed: set[str]) -> None:
    if not isinstance(data, dict) or set(data) - allowed:
        raise ValueError(f"Expected an object with only these fields: {sorted(allowed)}")


def _identifier(value: Any) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError("id must be a canonical 26-character ULID")
    return value


def _item(data: Any, *, generate: bool = False) -> dict[str, Any]:
    _keys(data, {"id", "text", "status", "notes"})
    return {
        "id": _identifier(data.get("id", new_ulid() if generate else None)),
        "text": _text(data.get("text"), "text", 500),
        "status": _choice(data.get("status", "pending"), STATUSES, "status"),
        "notes": _text(data.get("notes", ""), "notes", 2000, empty=True),
    }


def _choice(value: Any, choices: tuple, name: str) -> Any:
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}")
    return value


def _checklist(data: Any) -> dict[str, Any]:
    _keys(data, {"id", "name", "include_in_context", "context_mode", "revision", "items"})
    if type(data.get("include_in_context")) is not bool:
        raise ValueError("include_in_context must be a boolean")
    if type(data.get("revision")) is not int or data["revision"] < 1:
        raise ValueError("revision must be a positive integer")
    items = data.get("items")
    if not isinstance(items, list) or len(items) > 200:
        raise ValueError("items must be an array with at most 200 entries")
    items = [_item(x) for x in items]
    if len({x["id"] for x in items}) != len(items):
        raise ValueError("duplicate item id")
    if sum(x["status"] == "in_progress" for x in items) > 1:
        raise ValueError("At most one item per checklist may be in_progress; update items atomically to switch")
    return {
        "id": _identifier(data.get("id")), "name": _text(data.get("name"), "name", 200),
        "include_in_context": data["include_in_context"],
        "context_mode": _choice(data.get("context_mode"), CONTEXT_MODES, "context_mode"),
        "revision": data["revision"], "items": items,
    }


def _view(data: dict, mode: str = "full") -> dict[str, Any]:
    result = {k: copy.deepcopy(v) for k, v in data.items() if k != "items"}
    if mode == "name":
        return {"id": data["id"], "name": data["name"]}
    counts = {s: sum(x["status"] == s for x in data["items"]) for s in STATUSES}
    total = len(data["items"])
    remaining = total - counts["completed"] - counts["cancelled"]
    status = ("pending" if total == 0 else "cancelled" if counts["cancelled"] == total
              else "completed" if remaining == 0 else "in_progress" if counts["in_progress"]
              else "blocked" if counts["blocked"] else "pending")
    result.update(status=status, progress={"total": total, **counts, "remaining": remaining,
                  "fraction": counts["completed"] / (total - counts["cancelled"]) if total > counts["cancelled"] else 0.0})
    if mode == "full":
        result["items"] = copy.deepcopy(data["items"])
    return result


class ChecklistStore:
    """Session-local plans. Returned dictionaries never alias stored state.

    Use ``Session.invoke('planning.checklist.manage', ...)`` for recorded, policy-checked edits.
    Direct ``execute`` is intended for host setup and offline interchange.
    """

    def __init__(self) -> None:
        self._lists: dict[str, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self._lists)

    def to_dict(self) -> dict[str, Any]:
        return {"version": 1, "checklists": copy.deepcopy(list(self._lists.values()))}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChecklistStore":
        _keys(data, {"version", "checklists"})
        if type(data.get("version")) is not int or data["version"] != 1:
            raise ValueError("Unsupported checklist format version")
        values = data.get("checklists")
        if not isinstance(values, list) or len(values) > 100:
            raise ValueError("checklists must be an array with at most 100 entries")
        store = cls()
        for value in values:
            value = _checklist(value)
            if value["id"] in store._lists:
                raise ValueError("duplicate checklist id")
            store._lists[value["id"]] = value
        return store

    def execute(self, action: str, **args: Any) -> Any:
        allowed = {
            "list": {"mode"}, "get": {"id", "mode"}, "export": {"id"},
            "create": {"name", "items", "include_in_context", "context_mode"},
            "import": {"document"}, "delete": {"id", "expected_revision"},
            "update": {"id", "expected_revision", "name", "items", "include_in_context", "context_mode"},
            "add_item": {"id", "expected_revision", "item"},
            "update_item": {"id", "expected_revision", "item_id", "item"},
            "delete_item": {"id", "expected_revision", "item_id"},
        }
        if action not in allowed:
            raise ValueError(f"Unknown checklist action: {action}")
        _keys(args, allowed[action])
        mode = _choice(args.get("mode", "summary" if action == "list" else "full"), CONTEXT_MODES, "mode")
        if action == "list":
            return [_view(v, mode) for v in self._lists.values()]
        if action == "export":
            if "id" not in args:
                return self.to_dict()
            return {"version": 1, "checklists": [copy.deepcopy(self._get(args["id"]))]}
        if action == "import":
            incoming = self.from_dict(args.get("document"))
            if set(incoming._lists) & set(self._lists):
                raise ValueError("Checklist id already exists; import never overwrites local plans")
            if len(self._lists) + len(incoming._lists) > 100:
                raise ValueError("At most 100 checklists are allowed")
            self._lists.update(incoming._lists)
            return [_view(v) for v in incoming._lists.values()]
        if action == "create":
            if len(self._lists) >= 100:
                raise ValueError("At most 100 checklists are allowed")
            raw_items = args.get("items", [])
            if not isinstance(raw_items, list):
                raise ValueError("items must be an array")
            value = _checklist({"id": new_ulid(), "revision": 1, "name": args.get("name"),
                "include_in_context": args.get("include_in_context", True),
                "context_mode": args.get("context_mode", "summary"),
                "items": [_item(x, generate=True) for x in raw_items]})
        else:
            current = self._get(args.get("id"))
            if action == "get":
                return _view(current, mode)
            revision = args.get("expected_revision")
            if type(revision) is not int or revision != current["revision"]:
                raise ValueError(f"Revision conflict: read checklist first; expected_revision must be {current['revision']}")
            if action == "delete":
                del self._lists[current["id"]]
                return {"deleted": current["id"]}
            value = copy.deepcopy(current)
            if action == "update":
                for key in ("name", "include_in_context", "context_mode", "items"):
                    if key in args:
                        value[key] = args[key]
                if not isinstance(value["items"], list):
                    raise ValueError("items must be an array")
                value["items"] = [_item(x, generate=True) for x in value["items"]]
            elif action == "add_item":
                value["items"].append(_item(args.get("item"), generate=True))
            else:
                item_id = _identifier(args.get("item_id"))
                index = next((i for i, x in enumerate(value["items"]) if x["id"] == item_id), None)
                if index is None:
                    raise ValueError(f"Unknown item id: {item_id}")
                if action == "delete_item":
                    value["items"].pop(index)
                else:
                    patch = args.get("item")
                    _keys(patch, {"text", "status", "notes"})
                    value["items"][index].update(patch)
            value["revision"] += 1
            value = _checklist(value)
        self._lists[value["id"]] = value
        return _view(value)

    def _get(self, identifier: Any) -> dict:
        identifier = _identifier(identifier)
        if identifier not in self._lists:
            raise ValueError(f"Unknown checklist id: {identifier}")
        return self._lists[identifier]

    def render(self, *, max_chars: int = 6000) -> str:
        """Bounded projection; visibility affects this section, not history/access."""
        if max_chars < 100:
            return ""
        lines: list[str] = []
        visible = [v for v in self._lists.values() if v["include_in_context"]]
        for value in visible:
            line = dumps(_view(value, value["context_mode"]))
            if sum(len(x) + 1 for x in lines) + len(line) > max_chars - 100:
                line = dumps(_view(value, "summary"))
            if sum(len(x) + 1 for x in lines) + len(line) > max_chars - 100:
                break
            lines.append(line)
        if len(lines) < len(visible):
            lines.append(f"[{len(visible) - len(lines)} more checklists omitted; use planning.checklist.manage.]")
        return "\n".join(lines)
