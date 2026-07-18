"""Test helpers: quick tool-definition builders."""
from __future__ import annotations

from typing import Any, Callable, Optional


def echo_handler(text: str = "") -> str:
    return f"echo: {text}"


def tool_dict(
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
    parallel_safe: bool = False,
    timeout_s: float = 30.0,
    retries: int = 0,
    max_inline_tokens: Optional[int] = None,
    overflow: str = "handle",
) -> dict[str, Any]:
    parameters: dict[str, Any] = {"type": "object", "properties": properties or {}}
    if required:
        parameters["required"] = required
    d: dict[str, Any] = {
        "name": name,
        "category": category,
        "spec": {"description": description or f"{name} tool.", "parameters": parameters},
        "discovery": {
            "pinned": pinned,
            "no_embed": no_embed,
            "require_spec": require_spec,
            "embedding_text": embedding_text,
        },
        "execution": {
            "timeout_s": timeout_s,
            "retries": retries,
            "parallel_safe": parallel_safe,
            "output_policy": {"max_inline_tokens": max_inline_tokens, "overflow": overflow},
        },
    }
    if summary is not None or tags is not None:
        d["card"] = {"summary": summary or "", "tags": tags or []}
    return d


def ok_handler_factory(name: str) -> Callable[..., str]:
    def handler(**kwargs: Any) -> str:
        return f"{name} ok"

    return handler
