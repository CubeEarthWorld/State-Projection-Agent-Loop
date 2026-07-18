"""Tool definitions (spec §4).

Metadata is declared as JSON/dicts; the handler is a Python callable.
Registration paths:

* ``registry.register({...json...}, handler=fn)`` — explicit JSON (§4.1)
* ``@tool(...)`` decorator — metadata auto-generated from the signature,
  type hints and docstring (§4, discretionary)

Cards are auto-derived from the spec when omitted (§4.3).
"""
from __future__ import annotations

import importlib
import inspect
import re
import types
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Tool context (injected into handlers that declare a `ctx` parameter)
# ---------------------------------------------------------------------------

@dataclass
class ToolContext:
    """Runtime services available to tool handlers.

    A handler opts in by declaring a first parameter named ``ctx`` (or
    annotated with ``ToolContext``); it is excluded from the JSON schema and
    injected by the runtime.
    """

    session: Any = None
    registry: Any = None
    store: Any = None
    state: dict = field(default_factory=dict)
    config: Any = None
    search: Any = None
    logger: Any = None


# ---------------------------------------------------------------------------
# Dataclasses mirroring the §4.1 JSON schema
# ---------------------------------------------------------------------------

@dataclass
class ToolCard:
    summary: str = ""
    signature: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class ToolSpec:
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    returns: Optional[dict[str, Any]] = None
    usage_notes: str = ""
    examples: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ToolDiscovery:
    pinned: bool = False
    require_spec: bool = False
    embedding_text: str = ""
    no_embed: bool = False


@dataclass
class OutputPolicy:
    max_inline_tokens: Optional[int] = None  # None -> config.handles.inline_threshold_tokens
    overflow: str = "handle"  # "handle" | "truncate"
    preview: str = "head"  # "head" | "tail"


@dataclass
class ToolExecution:
    handler: Optional[Callable[..., Any]] = None
    handler_ref: str = ""
    timeout_s: float = 30.0
    retries: int = 0
    parallel_safe: bool = False
    resolve_handles: bool = True  # False for tools that take $hN ids literally (e.g. peek)
    output_policy: OutputPolicy = field(default_factory=OutputPolicy)


_JSON_TO_PY = {
    "string": "str", "integer": "int", "number": "float", "boolean": "bool",
    "array": "list", "object": "dict", "null": "None",
}


def _type_str(schema: dict[str, Any]) -> str:
    t = schema.get("type")
    if isinstance(t, list):
        return " | ".join(_JSON_TO_PY.get(x, str(x)) for x in t)
    if isinstance(t, str):
        return _JSON_TO_PY.get(t, t)
    if "enum" in schema:
        return "Literal[" + ", ".join(repr(v) for v in schema["enum"]) + "]"
    return "Any"


def synthesize_signature(name: str, parameters: dict[str, Any], returns: Optional[dict] = None) -> str:
    """Build a python-ish signature string from a JSON Schema (§4.3)."""
    props = parameters.get("properties", {}) or {}
    required = set(parameters.get("required", []) or [])
    parts = []
    for pname, sch in props.items():
        piece = f"{pname}: {_type_str(sch if isinstance(sch, dict) else {})}"
        if pname not in required:
            if isinstance(sch, dict) and "default" in sch:
                piece += f" = {sch['default']!r}"
            else:
                piece += " = None"
        parts.append(piece)
    ret = _type_str(returns) if isinstance(returns, dict) else "Any"
    return f"{name}({', '.join(parts)}) -> {ret}"


def _first_sentence(text: str) -> str:
    text = (text or "").strip().split("\n", 1)[0]
    for sep in ("。", ". "):
        if sep in text:
            return text.split(sep, 1)[0] + (sep.strip())
    return text


@dataclass
class ToolDef:
    name: str
    category: str = ""
    card: ToolCard = field(default_factory=ToolCard)
    spec: ToolSpec = field(default_factory=ToolSpec)
    discovery: ToolDiscovery = field(default_factory=ToolDiscovery)
    execution: ToolExecution = field(default_factory=ToolExecution)
    wants_ctx: bool = False

    # -- construction -------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any], handler: Optional[Callable[..., Any]] = None) -> "ToolDef":
        if not data.get("name"):
            raise ValueError("Tool definition requires a 'name'")
        spec_d = dict(data.get("spec") or {})
        spec = ToolSpec(
            description=spec_d.get("description", ""),
            parameters=spec_d.get("parameters") or {"type": "object", "properties": {}},
            returns=spec_d.get("returns"),
            usage_notes=spec_d.get("usage_notes", ""),
            examples=list(spec_d.get("examples") or []),
        )
        card_d = dict(data.get("card") or {})
        card = ToolCard(
            summary=card_d.get("summary", ""),
            signature=card_d.get("signature", ""),
            tags=list(card_d.get("tags") or []),
        )
        disc_d = dict(data.get("discovery") or {})
        discovery = ToolDiscovery(
            pinned=bool(disc_d.get("pinned", False)),
            require_spec=bool(disc_d.get("require_spec", False)),
            embedding_text=disc_d.get("embedding_text", ""),
            no_embed=bool(disc_d.get("no_embed", False)),
        )
        exe_d = dict(data.get("execution") or {})
        op_d = dict(exe_d.get("output_policy") or {})
        execution = ToolExecution(
            handler=handler,
            handler_ref=exe_d.get("handler", "") if isinstance(exe_d.get("handler"), str) else "",
            timeout_s=float(exe_d.get("timeout_s", 30.0)),
            retries=int(exe_d.get("retries", 0)),
            parallel_safe=bool(exe_d.get("parallel_safe", False)),
            resolve_handles=bool(exe_d.get("resolve_handles", True)),
            output_policy=OutputPolicy(
                max_inline_tokens=op_d.get("max_inline_tokens"),
                overflow=op_d.get("overflow", "handle"),
                preview=op_d.get("preview", "head"),
            ),
        )
        if execution.handler is None and callable(exe_d.get("handler")):
            execution.handler = exe_d["handler"]
        if execution.handler is None and execution.handler_ref:
            execution.handler = _resolve_handler(execution.handler_ref)
        td = cls(
            name=data["name"],
            category=data.get("category", ""),
            card=card,
            spec=spec,
            discovery=discovery,
            execution=execution,
        )
        td.derive_card()
        td.wants_ctx = _handler_wants_ctx(td.execution.handler)
        return td

    def derive_card(self) -> None:
        """Fill missing card fields from the spec (§4.3)."""
        if not self.card.summary:
            self.card.summary = _first_sentence(self.spec.description) or self.name
        if not self.card.signature:
            self.card.signature = synthesize_signature(self.name, self.spec.parameters, self.spec.returns)

    # -- projections of this definition ------------------------------------

    def card_text(self) -> str:
        """~30-token one-liner: enough to call the tool directly (§4.2, §6)."""
        sig = self.card.signature or self.name
        return f"- {sig} — {self.card.summary}"

    def spec_text(self) -> str:
        """Full spec rendering, attached to validation errors (§6)."""
        import json as _json

        lines = [f"### {self.name}", self.card.signature]
        if self.spec.description:
            lines.append(self.spec.description)
        lines.append("Parameters (JSON Schema): " + _json.dumps(self.spec.parameters, ensure_ascii=False))
        if self.spec.returns:
            lines.append("Returns: " + _json.dumps(self.spec.returns, ensure_ascii=False))
        if self.spec.usage_notes:
            lines.append("Usage notes: " + self.spec.usage_notes)
        for ex in self.spec.examples:
            call = _json.dumps(ex.get("call", {}), ensure_ascii=False)
            note = ex.get("note", "")
            lines.append(f"Example: {self.name}({call})" + (f" — {note}" if note else ""))
        return "\n".join(lines)

    def api_schema(self) -> dict[str, Any]:
        """OpenAI-style function schema for native tool calling."""
        description = self.spec.description or self.card.summary
        if self.spec.usage_notes:
            description = f"{description}\nUsage: {self.spec.usage_notes}"
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": description,
                "parameters": self.spec.parameters,
            },
        }

    def embedding_source(self) -> str:
        """Text embedded for layer-2 discovery (§4.2 discovery.embedding_text)."""
        if self.discovery.embedding_text:
            return self.discovery.embedding_text
        parts = [self.card.summary] + list(self.card.tags)
        return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Handler helpers
# ---------------------------------------------------------------------------

def _resolve_handler(ref: str) -> Callable[..., Any]:
    module_name, _, attr = ref.rpartition(".")
    if not module_name:
        raise ValueError(f"Handler reference {ref!r} must be 'module.attr'")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def _handler_wants_ctx(handler: Optional[Callable[..., Any]]) -> bool:
    if handler is None:
        return False
    try:
        sig = inspect.signature(handler)
    except (TypeError, ValueError):
        return False
    for param in sig.parameters.values():
        if param.name == "ctx":
            return True
        ann = param.annotation
        if ann is ToolContext or (isinstance(ann, str) and ann.endswith("ToolContext")):
            return True
        break  # only the first parameter may be the context
    return False


# ---------------------------------------------------------------------------
# @tool decorator: metadata from signature + docstring
# ---------------------------------------------------------------------------

def _hint_to_schema(hint: Any) -> dict[str, Any]:
    if hint is inspect.Parameter.empty or hint is Any:
        return {}
    if hint is type(None):
        return {"type": "null"}
    simple = {str: "string", int: "integer", float: "number", bool: "boolean",
              list: "array", dict: "object"}
    if hint in simple:
        return {"type": simple[hint]}
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin in (list, tuple, set):
        schema: dict[str, Any] = {"type": "array"}
        if args and args[0] is not Any:
            item = _hint_to_schema(args[0])
            if item:
                schema["items"] = item
        return schema
    if origin is dict:
        return {"type": "object"}
    if origin is typing.Literal:
        return {"enum": list(args)}
    if origin in (typing.Union, types.UnionType):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            base = _hint_to_schema(non_none[0])
            t = base.get("type")
            if isinstance(t, str):
                base["type"] = [t, "null"]
            elif t is None and "enum" not in base:
                base = {"type": ["string", "null"], **base} if not base else base
            return base
        return {"anyOf": [_hint_to_schema(a) for a in args]}
    return {}


_ARGS_SECTION = re.compile(r"^\s*(Args|Arguments|Parameters|引数)\s*:\s*$", re.IGNORECASE)
_SECTION_END = re.compile(r"^\s*(Returns|Raises|Yields|Examples?|Notes?|戻り値)\s*:\s*$", re.IGNORECASE)
_PARAM_LINE = re.compile(r"^\s+(\w+)\s*(?:\([^)]*\))?\s*:\s*(.+)$")


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    if not doc:
        return "", {}
    lines = inspect.cleandoc(doc).split("\n")
    description_lines: list[str] = []
    param_docs: dict[str, str] = {}
    in_args = False
    for line in lines:
        if _ARGS_SECTION.match(line):
            in_args = True
            continue
        if _SECTION_END.match(line):
            in_args = False
            continue
        if in_args:
            m = _PARAM_LINE.match("  " + line.strip()) if line.strip() else None
            if m:
                param_docs[m.group(1)] = m.group(2).strip()
            continue
        description_lines.append(line)
    description = "\n".join(description_lines).strip()
    return description, param_docs


def build_tooldef_from_function(
    fn: Callable[..., Any],
    *,
    name: Optional[str] = None,
    category: str = "",
    summary: Optional[str] = None,
    tags: Optional[list[str]] = None,
    pinned: bool = False,
    require_spec: bool = False,
    embedding_text: str = "",
    no_embed: bool = False,
    timeout_s: float = 30.0,
    retries: int = 0,
    parallel_safe: bool = False,
    max_inline_tokens: Optional[int] = None,
    overflow: str = "handle",
    preview: str = "head",
    usage_notes: str = "",
    examples: Optional[list[dict[str, Any]]] = None,
) -> ToolDef:
    description, param_docs = _parse_docstring(fn.__doc__ or "")
    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        hints = {}
    sig = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for i, (pname, param) in enumerate(sig.parameters.items()):
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        ann = hints.get(pname, param.annotation)
        if i == 0 and (pname == "ctx" or ann is ToolContext):
            continue
        schema = _hint_to_schema(ann)
        if pname in param_docs:
            schema = {**schema, "description": param_docs[pname]}
        if param.default is inspect.Parameter.empty:
            required.append(pname)
        elif param.default is not None:
            schema = {**schema, "default": param.default}
        properties[pname] = schema or {"description": param_docs.get(pname, "")}
    parameters: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        parameters["required"] = required

    data: dict[str, Any] = {
        "name": name or fn.__name__,
        "category": category,
        "card": {"summary": summary or _first_sentence(description), "tags": tags or []},
        "spec": {
            "description": description,
            "parameters": parameters,
            "usage_notes": usage_notes,
            "examples": examples or [],
        },
        "discovery": {
            "pinned": pinned,
            "require_spec": require_spec,
            "embedding_text": embedding_text,
            "no_embed": no_embed,
        },
        "execution": {
            "timeout_s": timeout_s,
            "retries": retries,
            "parallel_safe": parallel_safe,
            "output_policy": {
                "max_inline_tokens": max_inline_tokens,
                "overflow": overflow,
                "preview": preview,
            },
        },
    }
    return ToolDef.from_dict(data, handler=fn)


def tool(fn: Optional[Callable[..., Any]] = None, /, **kwargs: Any):
    """Decorator: attach an auto-generated :class:`ToolDef` to a function.

    Usable bare (``@tool``) or with options (``@tool(category="web", ...)``).
    The decorated function itself is returned unchanged and can be passed
    straight to ``registry.register``.
    """

    def wrap(f: Callable[..., Any]) -> Callable[..., Any]:
        f.__spal_tool__ = build_tooldef_from_function(f, **kwargs)  # type: ignore[attr-defined]
        return f

    if fn is not None:
        return wrap(fn)
    return wrap
