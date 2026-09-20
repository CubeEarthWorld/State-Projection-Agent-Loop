"""Policy engine: the sole owner of execution permission.

The LLM proposes; it never decides. Every planned effect of a capability
call is evaluated here, in a fixed layer order, before the runtime is
allowed to execute anything:

    absolute > admin > developer > workspace/user > session > llm

A ``deny`` at any layer can never be relaxed by a layer below it — this is
enforced structurally by taking the *most restrictive* verdict across all
matching layers, not by "last write wins". The LLM's own layer is the
lowest priority and, depending on ``llm_safety_mode``, is either ignored
entirely, advisory-only (recorded but never changes the outcome), or capped
at ``require_approval`` — it can never single-handedly grant ``allow`` or
issue a final ``deny``.

Declared effects (:class:`~state_projection_loop.capability.Effect`) are
self-reported by the capability author. This engine is the *policy*
boundary, not the *sandbox* boundary — pairing it with OS/process-level
restrictions on network, filesystem and credentials is the caller's
responsibility.
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .capability import Capability, Effect

LAYER_ORDER = ("absolute", "admin", "developer", "workspace", "session", "llm")
DECISIONS = ("allow", "deny", "require_approval")
_SEVERITY = {"allow": 1, "require_approval": 2, "deny": 3}

# Convenience scopes mapped onto effect-kind + resource patterns.
SCOPES: dict[str, tuple[Optional[str], str]] = {
    "workspace_read": ("read", "workspace:*"),
    "workspace_write": ("write", "workspace:*"),
    "sandbox_command": (None, "sandbox:*"),
    "network_access": (None, "network:*"),
    "external_mutation": ("external", "*"),
    "secrets_access": (None, "secrets:*"),
    "host_access": (None, "host:*"),
}

# Writes confined to the session's own working state never leave the process,
# so the auto presets allow them; they are declared as writes so the runtime
# keeps them in the model's stated order.
_LOCAL_STATE = (
    dict(decision="allow", capability_pattern="planning.checklist.manage", effect_kind="write",
         resource_pattern="working_state:checklists", reason="preset:local_checklists"),
    dict(decision="allow", capability_pattern="meta.user.ask", effect_kind="external",
         resource_pattern="user:*", reason="preset:ask_user"),
    # No effect_kind: reads of the working state are covered too (a read is
    # strictly less dangerous than the writes right beside it), exactly like
    # the memory.* rule below.
    dict(decision="allow", capability_pattern="state.*",
         resource_pattern="working_state:*", reason="preset:local_working_state"),
    dict(decision="allow", capability_pattern="memory.*", resource_pattern="memory:*",
         reason="preset:local_memory"),
)

# Each preset is the rules it installs, in order; a rule without a reason gets
# "preset:<name>". Specs rather than Rule objects, so no two engines share one.
PRESETS: dict[str, tuple[dict[str, Any], ...]] = {
    "deny_all": (dict(decision="deny"),),
    "approve_all_effects": (
        dict(decision="allow", effect_kind="none"),
        dict(decision="require_approval"),
    ),
    "auto_safe": _LOCAL_STATE + (
        dict(decision="allow", effect_kind="none"),
        dict(decision="allow", effect_kind="read", resource_pattern="workspace:*"),
        dict(decision="require_approval"),
    ),
    "auto_workspace_dev": _LOCAL_STATE + (
        dict(decision="allow", effect_kind="none"),
        dict(decision="allow", resource_pattern="workspace:*"),
        dict(decision="allow", resource_pattern="sandbox:*"),
        dict(decision="require_approval"),
    ),
}


# Case-sensitive on every platform (fnmatch.fnmatch folds case on Windows),
# and the function the shared spec fixtures pin against the Dart port.
glob_match = fnmatch.fnmatchcase


@dataclass
class Rule:
    decision: str  # one of DECISIONS
    capability_pattern: str = "*"
    effect_kind: Optional[str] = None  # None matches any effect kind
    resource_pattern: str = "*"
    arg_predicate: Optional[Callable[[dict[str, Any]], bool]] = None
    reason: str = ""

    @property
    def is_catch_all(self) -> bool:
        """Matches every call: the layer's fallback, consulted after every
        rule that names something (see :meth:`PolicyEngine._match_layer`)."""
        return (self.capability_pattern == "*" and self.effect_kind is None
                and self.resource_pattern == "*" and self.arg_predicate is None)

    def matches(self, capability: Capability, effect: Effect, arguments: dict[str, Any]) -> bool:
        if not glob_match(capability.name, self.capability_pattern):
            return False
        if self.effect_kind is not None and effect.kind != self.effect_kind:
            return False
        if not glob_match(effect.resource, self.resource_pattern):
            return False
        if self.arg_predicate is not None and not self.arg_predicate(arguments):
            return False
        return True


@dataclass
class PolicyDecision:
    decision: str
    reason: str
    layer: str = ""


class PolicyEngine:
    def __init__(self, *, default_decision: str = "require_approval",
                 on_change: Optional[Callable[[str], None]] = None) -> None:
        if default_decision not in DECISIONS:
            raise ValueError(f"default_decision must be one of {DECISIONS}")
        self.default_decision = default_decision
        self.llm_safety_mode = "disabled"  # disabled | advisory | approval_routing
        self.layers: dict[str, list[Rule]] = {name: [] for name in LAYER_ORDER}
        self.revision = 0
        self._on_change = on_change

    # -- mutation (each bumps the revision; a stale ApprovalRequest is
    #    detected by comparing revisions — see Run.resolve_approval) --------

    def _changed(self, description: str) -> None:
        self.revision += 1
        if self._on_change is not None:
            self._on_change(description)

    def add_rule(self, layer: str, rule: Rule) -> None:
        if layer not in LAYER_ORDER:
            raise ValueError(f"Unknown policy layer {layer!r}; expected one of {LAYER_ORDER}")
        self.layers[layer].append(rule)
        self._changed(f"add_rule layer={layer} pattern={rule.capability_pattern} decision={rule.decision}")

    def clear_layer(self, layer: str) -> None:
        self.layers[layer] = []
        self._changed(f"clear_layer layer={layer}")

    def set_scope(self, scope: str, decision: str, *, layer: str = "workspace") -> None:
        """Grant/deny/gate one of the named scopes, e.g.
        ``set_scope("network_access", "deny")``."""
        if scope not in SCOPES:
            raise ValueError(f"Unknown scope {scope!r}; expected one of {sorted(SCOPES)}")
        effect_kind, resource_pattern = SCOPES[scope]
        self.add_rule(layer, Rule(decision=decision, capability_pattern="*",
                                   effect_kind=effect_kind, resource_pattern=resource_pattern,
                                   reason=f"scope:{scope}"))

    def apply_preset(self, preset: str, *, layer: str = "workspace") -> None:
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset {preset!r}; expected one of {tuple(PRESETS)}")
        self.clear_layer(layer)
        for spec in PRESETS[preset]:
            self.add_rule(layer, Rule(**{"reason": f"preset:{preset}", **spec}))

    def set_llm_safety_mode(self, mode: str) -> None:
        if mode not in ("disabled", "advisory", "approval_routing"):
            raise ValueError("llm_safety_mode must be disabled|advisory|approval_routing")
        self.llm_safety_mode = mode
        self._changed(f"set_llm_safety_mode {mode}")

    # -- evaluation -----------------------------------------------------------

    def _match_layer(self, layer: str, capability: Capability, effect: Effect,
                      arguments: dict[str, Any]) -> Optional[Rule]:
        """The first matching rule in the layer. A rule that matches
        everything (a preset's closing ``require_approval``) is the layer's
        fallback and is consulted last, so a grant added after
        ``apply_preset`` on the same layer takes effect instead of being
        shadowed by it."""
        matched = [rule for rule in self.layers[layer] if rule.matches(capability, effect, arguments)]
        return next((rule for rule in matched if not rule.is_catch_all), matched[0] if matched else None)

    def _evaluate_effect(self, capability: Capability, effect: Effect,
                          arguments: dict[str, Any]) -> tuple[str, str, str]:
        # `best` tracks the most restrictive verdict among layers that
        # actually matched a rule. The engine's `default_decision` is a
        # fallback used ONLY when no layer matched anything — it must never
        # compete in the severity race, or a real "allow" rule could never
        # beat a default that happens to be stricter (and vice versa,
        # defeating "most restrictive real rule wins").
        best: Optional[tuple[int, str, str, str]] = None  # (severity, decision, layer, reason)
        for layer in LAYER_ORDER:
            if layer == "llm" and self.llm_safety_mode == "disabled":
                continue
            rule = self._match_layer(layer, capability, effect, arguments)
            if rule is None:
                continue
            decision = rule.decision
            if layer == "llm":
                if self.llm_safety_mode == "advisory":
                    continue  # recorded by caller via decision reason text, never changes outcome
                # approval_routing: LLM may only escalate toward approval, never grant
                # allow on its own and never issue the final deny by itself.
                decision = "require_approval" if decision != "allow" else self.default_decision
            severity = _SEVERITY[decision]
            if best is None or severity > best[0]:
                best = (severity, decision, layer, rule.reason)
        if best is None:
            return self.default_decision, "default", "no matching rule"
        return best[1], best[2], best[3]

    def evaluate(self, capability: Capability, arguments: dict[str, Any]) -> PolicyDecision:
        # planned_effects always yields at least one effect (an undeclared
        # capability gets a synthesized "external" one), so there is no
        # empty case to seed. `max` keeps the first of equal severities,
        # which is the first-listed effect.
        decision, layer, reason = max(
            (self._evaluate_effect(capability, e, arguments) for e in capability.planned_effects),
            key=lambda r: _SEVERITY[r[0]],
        )
        return PolicyDecision(decision=decision, reason=reason, layer=layer)


