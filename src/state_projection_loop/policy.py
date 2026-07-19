"""Policy engine: the sole owner of execution permission (P1-2, §7).

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
responsibility (§7.4).
"""
from __future__ import annotations

import fnmatch
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .capability import Capability, Effect

LAYER_ORDER = ("absolute", "admin", "developer", "workspace", "session", "llm")
DECISIONS = ("allow", "deny", "require_approval")
_SEVERITY = {"allow": 1, "require_approval": 2, "deny": 3}

# Convenience scopes mapped onto effect-kind + resource patterns (§7.2).
SCOPES: dict[str, tuple[Optional[str], str]] = {
    "workspace_read": ("read", "workspace:*"),
    "workspace_write": ("write", "workspace:*"),
    "sandbox_command": (None, "sandbox:*"),
    "network_access": (None, "network:*"),
    "external_mutation": ("external", "*"),
    "secrets_access": (None, "secrets:*"),
    "host_access": (None, "host:*"),
}

PRESETS = ("deny_all", "approve_all_effects", "auto_safe", "auto_workspace_dev")


@dataclass
class Rule:
    decision: str  # one of DECISIONS
    capability_pattern: str = "*"
    effect_kind: Optional[str] = None  # None matches any effect kind
    resource_pattern: str = "*"
    arg_predicate: Optional[Callable[[dict[str, Any]], bool]] = None
    reason: str = ""

    def matches(self, capability: Capability, effect: Effect, arguments: dict[str, Any]) -> bool:
        if not fnmatch.fnmatch(capability.name, self.capability_pattern):
            return False
        if self.effect_kind is not None and effect.kind != self.effect_kind:
            return False
        if not fnmatch.fnmatch(effect.resource, self.resource_pattern):
            return False
        if self.arg_predicate is not None and not self.arg_predicate(arguments):
            return False
        return True


@dataclass
class PolicyDecision:
    decision: str
    reason: str
    layer: str = ""
    per_effect: list[tuple[Effect, str, str]] = field(default_factory=list)  # (effect, decision, layer)


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
        """Grant/deny/gate one of the named scopes (§7.2), e.g.
        ``set_scope("network_access", "deny")``."""
        if scope not in SCOPES:
            raise ValueError(f"Unknown scope {scope!r}; expected one of {sorted(SCOPES)}")
        effect_kind, resource_pattern = SCOPES[scope]
        self.add_rule(layer, Rule(decision=decision, capability_pattern="*",
                                   effect_kind=effect_kind, resource_pattern=resource_pattern,
                                   reason=f"scope:{scope}"))

    def apply_preset(self, preset: str, *, layer: str = "workspace") -> None:
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset {preset!r}; expected one of {PRESETS}")
        self.clear_layer(layer)
        if preset == "deny_all":
            self.add_rule(layer, Rule(decision="deny", reason="preset:deny_all"))
        elif preset == "approve_all_effects":
            self.add_rule(layer, Rule(decision="allow", effect_kind="none", reason="preset:approve_all_effects"))
            self.add_rule(layer, Rule(decision="require_approval", reason="preset:approve_all_effects"))
        elif preset == "auto_safe":
            self.add_rule(layer, Rule(decision="allow", effect_kind="none", reason="preset:auto_safe"))
            self.add_rule(layer, Rule(decision="allow", effect_kind="read", resource_pattern="workspace:*",
                                       reason="preset:auto_safe"))
            self.add_rule(layer, Rule(decision="require_approval", reason="preset:auto_safe"))
        elif preset == "auto_workspace_dev":
            self.add_rule(layer, Rule(decision="allow", effect_kind="none", reason="preset:auto_workspace_dev"))
            self.add_rule(layer, Rule(decision="allow", resource_pattern="workspace:*",
                                       reason="preset:auto_workspace_dev"))
            self.add_rule(layer, Rule(decision="allow", resource_pattern="sandbox:*",
                                       reason="preset:auto_workspace_dev"))
            self.add_rule(layer, Rule(decision="require_approval", reason="preset:auto_workspace_dev"))

    def set_llm_safety_mode(self, mode: str) -> None:
        if mode not in ("disabled", "advisory", "approval_routing"):
            raise ValueError("llm_safety_mode must be disabled|advisory|approval_routing")
        self.llm_safety_mode = mode
        self._changed(f"set_llm_safety_mode {mode}")

    # -- evaluation -----------------------------------------------------------

    def _match_layer(self, layer: str, capability: Capability, effect: Effect,
                      arguments: dict[str, Any]) -> Optional[Rule]:
        for rule in self.layers[layer]:
            if rule.matches(capability, effect, arguments):
                return rule
        return None

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
        # A capability that declares no effects at all is NOT assumed safe —
        # that would reward an author who simply forgot to declare effects
        # with maximum trust. Treat undeclared effects as the most
        # restrictive kind so the default posture stays conservative.
        effects = capability.effects or [Effect(kind="external", resource="undeclared:*")]
        per_effect: list[tuple[Effect, str, str]] = []
        worst_decision, worst_layer, worst_reason = "allow", "default", "no effects"
        worst_severity = 0
        for effect in effects:
            decision, layer, reason = self._evaluate_effect(capability, effect, arguments)
            per_effect.append((effect, decision, layer))
            severity = _SEVERITY[decision]
            if severity > worst_severity:
                worst_decision, worst_layer, worst_reason, worst_severity = decision, layer, reason, severity
        return PolicyDecision(decision=worst_decision, reason=worst_reason, layer=worst_layer,
                               per_effect=per_effect)


@dataclass
class ApprovalExpiry:
    """Small helper so callers don't hardcode a bare number of seconds."""

    seconds: float = 3600.0

    def at(self, *, now: Optional[float] = None) -> float:
        return (now if now is not None else time.time()) + self.seconds
