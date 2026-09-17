"""Capabilities: versioned execution contracts (replaces the old ``ToolDef``).

A Capability is not just a function signature — it is a full contract the
runtime and policy engine can reason about *without* running the handler:
what it touches (``effects``) and whether it is safe to retry after a
timeout (``retry_safety``). The LLM only ever sees the projected card/spec
text; it never gets to assert any of these properties itself.

Naming: capabilities live in a dotted namespace 3-5 levels deep, mirroring a
stable service/resource/operation shape rather than an org chart, e.g.::

    filesystem.file.read@1
    github.pull_request.create@1

``name`` is the dotted path; ``version`` is a plain integer. The qualified id
(``name@version``) is what the registry indexes on, so two versions of the
same capability can coexist during a rollout.
"""
from __future__ import annotations

import inspect
import re
import types
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .serialization import dumps

# ---------------------------------------------------------------------------
# Effects & retry safety — the vocabulary the policy engine and runtime share
# ---------------------------------------------------------------------------

EFFECT_KINDS = ("none", "read", "write", "external")
RETRY_SAFETY = ("pure", "idempotent", "check_then_retry", "never_retry")


@dataclass(frozen=True)
class Effect:
    """One declared side effect: what kind, and which resource it touches.

    ``resource`` is a free-form pattern the policy engine matches against
    rules (e.g. ``"workspace:*"``, ``"network:api.github.com"``,
    ``"secrets:*"``). Declaration is self-reported by the capability author;
    it is the *planned* effect, not a runtime guarantee — see the Sandbox
    note in the module docstring of :mod:`state_projection_loop.policy`.
    """

    kind: str  # one of EFFECT_KINDS
    resource: str = "*"

    def __post_init__(self) -> None:
        if self.kind not in EFFECT_KINDS:
            raise ValueError(f"Effect.kind must be one of {EFFECT_KINDS}, got {self.kind!r}")


# ---------------------------------------------------------------------------
# Tool context (injected into handlers that declare a `ctx` parameter)
# ---------------------------------------------------------------------------

@dataclass
class ToolContext:
    """What a tool handler receives.

    A handler opts in by declaring a first parameter named ``ctx`` (or
    annotated with ``ToolContext``); it is excluded from the JSON schema and
    injected by the runtime with ``command_id`` set. ``command_id`` is stable
    across retries of the *same* logical attempt and is the correct
    idempotency key to hand to an external API. Fields are typed ``Any`` only
    to avoid circular imports; they hold the session's Config, Registry,
    EventLedger, Run, WorkingState, ArtifactStore and ToolSearch.

    Sections render from the superset :class:`~state_projection_loop.projection.TurnContext`;
    the runtime narrows it with :meth:`for_command` before a handler runs, so
    projection state never reaches a tool.
    """

    config: Any = None
    registry: Any = None
    ledger: Any = None
    run: Any = None
    working_state: Any = None
    session: Any = None
    store: Any = None
    search: Any = None
    command_id: str = ""

    @property
    def run_id(self) -> str:
        return self.run.id if self.run is not None else ""

    def for_command(self, command_id: str) -> "ToolContext":
        """The handler-facing view of this context for one command."""
        return ToolContext(
            config=self.config, registry=self.registry, ledger=self.ledger, run=self.run,
            working_state=self.working_state, session=self.session, store=self.store,
            search=self.search, command_id=command_id,
        )


# ---------------------------------------------------------------------------
# Dataclasses mirroring the projected shape of a capability
# ---------------------------------------------------------------------------

@dataclass
class CapabilityCard:
    summary: str = ""
    tags: list[str] = field(default_factory=list)
    signature: str = ""  # derived by Capability.derive_card(); never authored


@dataclass
class CapabilitySpec:
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    returns: Optional[dict[str, Any]] = None
    usage_notes: str = ""
    examples: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CapabilityDiscovery:
    pinned: bool = False
    require_spec: bool = False
    embedding_text: str = ""
    no_embed: bool = False
    # One standing sentence for the kernel's "[Runtime notes]", shown while
    # the capability is pinned and reachable. Pinned only: the pin set is the
    # developer's own bound on kernel size.
    kernel_note: str = ""


@dataclass
class OutputPolicy:
    max_inline_tokens: Optional[int] = None  # None -> config.artifacts.inline_threshold_tokens
    overflow: str = "artifact"  # "artifact" | "truncate"
    preview: str = "head"  # "head" | "tail"


@dataclass
class CapabilityExecution:
    handler: Optional[Callable[..., Any]] = None
    timeout_s: float = 30.0
    retries: int = 0
    retry_safety: str = "never_retry"  # one of RETRY_SAFETY
    resolve_handles: bool = True  # False for tools that take artifact refs literally (e.g. peek)
    output_policy: OutputPolicy = field(default_factory=OutputPolicy)

    def __post_init__(self) -> None:
        if self.retry_safety not in RETRY_SAFETY:
            raise ValueError(f"retry_safety must be one of {RETRY_SAFETY}, got {self.retry_safety!r}")
        if self.retries > 0 and self.retry_safety not in ("pure", "idempotent"):
            raise ValueError(
                f"retries={self.retries} is unsafe for retry_safety={self.retry_safety!r}; "
                "only 'pure' or 'idempotent' capabilities may set retries > 0"
            )


def _type_str(schema: dict[str, Any]) -> str:
    """Render a parameter's type for the signature line.

    Deliberately the JSON Schema vocabulary, not a language's: the model
    sees the same type names here and in the full spec below, and the Python
    and Dart ports render one identical string instead of two dialects.
    """
    t = schema.get("type")
    if isinstance(t, list):
        return " | ".join(str(x) for x in t)
    if isinstance(t, str):
        return t
    if "enum" in schema:
        return "Literal[" + ", ".join(dumps(v) for v in schema["enum"]) + "]"
    return "any"


def synthesize_signature(name: str, parameters: dict[str, Any], returns: Optional[dict] = None) -> str:
    """Build a signature string from a JSON Schema."""
    props = parameters.get("properties", {}) or {}
    required = set(parameters.get("required", []) or [])
    parts = []
    for pname, sch in props.items():
        piece = f"{pname}: {_type_str(sch if isinstance(sch, dict) else {})}"
        if pname not in required:
            piece += f" = {dumps(sch['default'])}" if isinstance(sch, dict) and "default" in sch else " = null"
        parts.append(piece)
    ret = _type_str(returns) if isinstance(returns, dict) else "any"
    return f"{name}({', '.join(parts)}) -> {ret}"


def _first_sentence(text: str) -> str:
    text = (text or "").strip().split("\n", 1)[0]
    for sep in ("。", ". "):
        if sep in text:
            return text.split(sep, 1)[0] + (sep.strip())
    return text


_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*){1,4}$")

# Many native-function-calling providers (OpenAI included) reject "." in a
# function name (they require ``^[a-zA-Z0-9_-]+$``). The dotted name is the
# capability's real, canonical identity everywhere else (registry, ledger,
# policy patterns); `api_name` is only a wire-safe encoding of it for the
# tool schema sent to the provider. "__" is reserved as that encoding's
# segment separator, so a name may not contain it — otherwise decoding an
# api_name back to a dotted name would be ambiguous.
API_NAME_SEPARATOR = "__"


def validate_capability_name(name: str) -> None:
    """Enforce the 2-5 level dotted namespace convention (service.resource.op)."""
    if not _NAME_RE.match(name):
        raise ValueError(
            f"Capability name {name!r} must be 2-5 lowercase dotted segments, "
            "e.g. 'filesystem.file.read' or 'github.pull_request.create'"
        )
    if API_NAME_SEPARATOR in name:
        raise ValueError(
            f"Capability name {name!r} must not contain {API_NAME_SEPARATOR!r} "
            "(reserved for the provider-safe api_name encoding)"
        )


def to_api_name(name: str) -> str:
    """Dotted capability name -> provider-safe function name."""
    return name.replace(".", API_NAME_SEPARATOR)


def from_api_name(api_name: str) -> str:
    """Provider-safe function name -> dotted capability name."""
    return api_name.replace(API_NAME_SEPARATOR, ".")


@dataclass
class Capability:
    name: str
    version: int = 1
    category: str = ""
    card: CapabilityCard = field(default_factory=CapabilityCard)
    spec: CapabilitySpec = field(default_factory=CapabilitySpec)
    discovery: CapabilityDiscovery = field(default_factory=CapabilityDiscovery)
    execution: CapabilityExecution = field(default_factory=CapabilityExecution)
    effects: list[Effect] = field(default_factory=list)
    wants_ctx: bool = False

    def __post_init__(self) -> None:
        validate_capability_name(self.name)

    @property
    def qualified_name(self) -> str:
        return f"{self.name}@{self.version}"

    @property
    def api_name(self) -> str:
        """Provider-safe function name for native tool-calling schemas."""
        return to_api_name(self.name)

    # -- construction ---------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any], handler: Optional[Callable[..., Any]] = None) -> "Capability":
        if not data.get("name"):
            raise ValueError("Capability definition requires a 'name'")
        spec_d = dict(data.get("spec") or {})
        spec = CapabilitySpec(
            description=spec_d.get("description", ""),
            parameters=spec_d.get("parameters") or {"type": "object", "properties": {}},
            returns=spec_d.get("returns"),
            usage_notes=spec_d.get("usage_notes", ""),
            examples=list(spec_d.get("examples") or []),
        )
        card_d = dict(data.get("card") or {})
        if "signature" in card_d:
            raise ValueError(
                f"Capability {data['name']!r}: card.signature is derived from the name and "
                "parameters, not authored — remove it from the definition"
            )
        card = CapabilityCard(summary=card_d.get("summary", ""), tags=list(card_d.get("tags") or []))
        disc_d = dict(data.get("discovery") or {})
        discovery = CapabilityDiscovery(
            pinned=bool(disc_d.get("pinned", False)),
            require_spec=bool(disc_d.get("require_spec", False)),
            embedding_text=disc_d.get("embedding_text", ""),
            no_embed=bool(disc_d.get("no_embed", False)),
            kernel_note=disc_d.get("kernel_note", ""),
        )
        exe_d = dict(data.get("execution") or {})
        op_d = dict(exe_d.get("output_policy") or {})
        execution = CapabilityExecution(
            handler=handler,
            timeout_s=float(exe_d.get("timeout_s", 30.0)),
            retries=int(exe_d.get("retries", 0)),
            retry_safety=exe_d.get("retry_safety", "never_retry"),
            resolve_handles=bool(exe_d.get("resolve_handles", True)),
            output_policy=OutputPolicy(
                max_inline_tokens=op_d.get("max_inline_tokens"),
                overflow=op_d.get("overflow", "artifact"),
                preview=op_d.get("preview", "head"),
            ),
        )
        if execution.handler is None and callable(exe_d.get("handler")):
            execution.handler = exe_d["handler"]
        effects = [
            Effect(kind=e.get("kind", "none"), resource=e.get("resource", "*"))
            for e in (data.get("effects") or [])
        ]
        cap = cls(
            name=data["name"],
            version=int(data.get("version", 1)),
            category=data.get("category", ""),
            card=card,
            spec=spec,
            discovery=discovery,
            execution=execution,
            effects=effects,
        )
        cap.derive_card()
        cap.wants_ctx = _handler_wants_ctx(cap.execution.handler)
        return cap

    def derive_card(self) -> None:
        if not self.card.summary:
            self.card.summary = _first_sentence(self.spec.description) or self.name
        # The signature is always derived, never authored: it is the one
        # line telling the model how to call this capability, and a
        # hand-written one drifts from the real name and parameters.
        self.card.signature = synthesize_signature(self.name, self.spec.parameters, self.spec.returns)

    # -- projections ------------------------------------------------------

    def card_text(self) -> str:
        """~30-token one-liner: enough to call the capability directly."""
        sig = self.card.signature or self.name
        return f"- {sig} — {self.card.summary}"

    def spec_text(self) -> str:
        import json as _json

        lines = [f"### {self.qualified_name}", self.card.signature]
        if self.spec.description:
            lines.append(self.spec.description)
        lines.append("Parameters (JSON Schema): " + dumps(self.spec.parameters))
        if self.spec.returns:
            lines.append("Returns: " + dumps(self.spec.returns))
        if self.effects:
            lines.append("Effects: " + ", ".join(f"{e.kind}:{e.resource}" for e in self.effects))
        if self.spec.usage_notes:
            lines.append("Usage notes: " + self.spec.usage_notes)
        for ex in self.spec.examples:
            call = dumps(ex.get("call", {}))
            note = ex.get("note", "")
            lines.append(f"Example: {self.name}({call})" + (f" — {note}" if note else ""))
        return "\n".join(lines)

    def api_schema(self) -> dict[str, Any]:
        """OpenAI-style function schema for native tool calling.

        Uses ``api_name`` (dots encoded as ``__``), not the dotted ``name``
        directly — most native-function-calling providers, OpenAI included,
        reject "." in a function name. Callers translate the name back with
        :func:`from_api_name` (see ``Registry.resolve_api_name``) before
        the call reaches the registry.
        """
        description = self.spec.description or self.card.summary
        if self.spec.usage_notes:
            description = f"{description}\nUsage: {self.spec.usage_notes}"
        return {
            "type": "function",
            "function": {
                "name": self.api_name,
                "description": description,
                "parameters": self.spec.parameters,
            },
        }

    def embedding_source(self) -> str:
        if self.discovery.embedding_text:
            return self.discovery.embedding_text
        parts = [self.card.summary] + list(self.card.tags)
        return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Handler helpers
# ---------------------------------------------------------------------------

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
# @capability decorator: metadata from signature + docstring
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


def build_capability_from_function(
    fn: Callable[..., Any],
    *,
    name: Optional[str] = None,
    version: int = 1,
    category: str = "",
    summary: Optional[str] = None,
    tags: Optional[list[str]] = None,
    pinned: bool = False,
    require_spec: bool = False,
    embedding_text: str = "",
    no_embed: bool = False,
    kernel_note: str = "",
    timeout_s: float = 30.0,
    retries: int = 0,
    retry_safety: str = "never_retry",
    effects: Optional[list[tuple[str, str]]] = None,
    max_inline_tokens: Optional[int] = None,
    overflow: str = "artifact",
    preview: str = "head",
    usage_notes: str = "",
    examples: Optional[list[dict[str, Any]]] = None,
) -> Capability:
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
        "name": name or fn.__name__.replace("_", "."),
        "version": version,
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
            "kernel_note": kernel_note,
        },
        "execution": {
            "timeout_s": timeout_s,
            "retries": retries,
            "retry_safety": retry_safety,
            "output_policy": {
                "max_inline_tokens": max_inline_tokens,
                "overflow": overflow,
                "preview": preview,
            },
        },
        "effects": [{"kind": k, "resource": r} for k, r in (effects or [])],
    }
    return Capability.from_dict(data, handler=fn)


def capability(fn: Optional[Callable[..., Any]] = None, /, **kwargs: Any):
    """Decorator: attach an auto-generated :class:`Capability` to a function.

    Usable bare (``@capability``) or with options
    (``@capability(category="web", retry_safety="idempotent", ...)``). The
    decorated function itself is returned unchanged and can be passed
    straight to ``registry.register``.
    """

    def wrap(f: Callable[..., Any]) -> Callable[..., Any]:
        f.__spal_capability__ = build_capability_from_function(f, **kwargs)  # type: ignore[attr-defined]
        return f

    if fn is not None:
        return wrap(fn)
    return wrap
