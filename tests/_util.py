"""Test helpers: quick capability-definition builders."""
from __future__ import annotations

from typing import Any, Callable, Optional


def echo_handler(text: str = "") -> str:
    return f"echo: {text}"


def capability_dict(
    name: str,
    *,
    category: str = "misc",
    description: str = "",
    summary: Optional[str] = None,
    tags: Optional[list[str]] = None,
    embedding_text: str = "",
    pinned: bool = False,
    no_embed: bool = False,
    require_spec: bool = False,
    properties: Optional[dict[str, Any]] = None,
    required: Optional[list[str]] = None,
    retry_safety: str = "never_retry",
    retries: int = 0,
    timeout_s: float = 30.0,
    effects: Optional[list[tuple[str, str]]] = None,
    max_inline_tokens: Optional[int] = None,
    overflow: str = "artifact",
) -> dict[str, Any]:
    """Build a capability definition dict. ``name`` should already be dotted
    (e.g. ``"demo.echo.say"``); a bare name is namespaced under ``test.``."""
    if "." not in name:
        name = f"test.{name}"
    parameters: dict[str, Any] = {"type": "object", "properties": properties or {}}
    if required:
        parameters["required"] = required
    d: dict[str, Any] = {
        "name": name,
        "category": category,
        "spec": {"description": description or f"{name} capability.", "parameters": parameters},
        "discovery": {
            "pinned": pinned,
            "no_embed": no_embed,
            "require_spec": require_spec,
            "embedding_text": embedding_text,
        },
        "execution": {
            "timeout_s": timeout_s,
            "retries": retries,
            "retry_safety": retry_safety,
            "output_policy": {"max_inline_tokens": max_inline_tokens, "overflow": overflow},
        },
        "effects": [{"kind": k, "resource": r} for k, r in (effects or [("none", "*")])],
    }
    if summary is not None or tags is not None:
        d["card"] = {"summary": summary or "", "tags": tags or []}
    return d


def ok_handler_factory(name: str) -> Callable[..., str]:
    def handler(**kwargs: Any) -> str:
        return f"{name} ok"

    return handler
