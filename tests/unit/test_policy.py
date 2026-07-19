"""Policy engine: layer priority (a deny at any layer can't be relaxed by
a lower layer), scopes/presets, undeclared-effects default, LLM safety
modes never granting a unilateral allow or the final deny."""
from __future__ import annotations

from state_projection_loop.capability import Capability, Effect
from state_projection_loop.policy import PolicyEngine, Rule


def cap(name="demo.thing", effects=None):
    return Capability(name=name, effects=effects if effects is not None else [Effect(kind="write", resource="workspace:*")])


class TestDefaultAndUndeclared:
    def test_default_decision_used_when_nothing_matches(self):
        engine = PolicyEngine(default_decision="deny")
        decision = engine.evaluate(cap(), {})
        assert decision.decision == "deny"
        assert decision.layer == "default"

    def test_undeclared_effects_are_not_treated_as_safe(self):
        engine = PolicyEngine(default_decision="require_approval")
        engine.apply_preset("auto_safe")  # only allows effect_kind="none" and workspace reads
        no_effects_cap = Capability(name="demo.mystery")  # effects=[] declared
        decision = engine.evaluate(no_effects_cap, {})
        # A forgotten effects declaration must NOT be rewarded with "allow":
        # undeclared effects synthesize as "external", which auto_safe's
        # effect_kind="none" rule does not match — only its catch-all
        # require_approval rule does.
        assert decision.decision == "require_approval"


class TestLayering:
    def test_deny_at_higher_layer_cannot_be_relaxed_by_lower(self):
        engine = PolicyEngine(default_decision="allow")
        engine.add_rule("admin", Rule(decision="deny", capability_pattern="demo.*"))
        engine.add_rule("session", Rule(decision="allow", capability_pattern="demo.*"))
        decision = engine.evaluate(cap(), {})
        assert decision.decision == "deny"
        assert decision.layer == "admin"

    def test_most_restrictive_real_rule_wins_even_if_less_severe_than_default(self):
        # default is "deny" (most restrictive), but an actual matching rule
        # (even an "allow") must still be honored over the synthetic default.
        engine = PolicyEngine(default_decision="deny")
        engine.add_rule("workspace", Rule(decision="allow", effect_kind="none"))
        decision = engine.evaluate(cap(effects=[Effect(kind="none")]), {})
        assert decision.decision == "allow"
        assert decision.layer == "workspace"


class TestScopesAndPresets:
    def test_auto_safe_allows_pure_and_workspace_read(self):
        engine = PolicyEngine(default_decision="require_approval")
        engine.apply_preset("auto_safe")
        pure = engine.evaluate(cap(effects=[Effect(kind="none")]), {})
        read = engine.evaluate(cap(effects=[Effect(kind="read", resource="workspace:*")]), {})
        write = engine.evaluate(cap(effects=[Effect(kind="write", resource="workspace:*")]), {})
        assert pure.decision == "allow"
        assert read.decision == "allow"
        assert write.decision == "require_approval"

    def test_deny_all_preset(self):
        engine = PolicyEngine()
        engine.apply_preset("deny_all")
        decision = engine.evaluate(cap(effects=[Effect(kind="none")]), {})
        assert decision.decision == "deny"

    def test_set_scope_network(self):
        engine = PolicyEngine(default_decision="allow")
        engine.set_scope("network_access", "deny")
        decision = engine.evaluate(cap(effects=[Effect(kind="external", resource="network:api.example.com")]), {})
        assert decision.decision == "deny"


class TestRevision:
    def test_mutation_bumps_revision(self):
        engine = PolicyEngine()
        r0 = engine.revision
        engine.set_scope("network_access", "deny")
        assert engine.revision == r0 + 1

    def test_on_change_callback_invoked(self):
        calls = []
        engine = PolicyEngine(on_change=calls.append)
        engine.set_scope("network_access", "deny")
        assert len(calls) == 1


class TestLlmSafetyMode:
    def test_disabled_ignores_llm_layer(self):
        engine = PolicyEngine(default_decision="require_approval")
        engine.add_rule("llm", Rule(decision="allow"))
        decision = engine.evaluate(cap(effects=[Effect(kind="write", resource="workspace:*")]), {})
        assert decision.decision == "require_approval"  # llm layer never consulted

    def test_advisory_never_changes_outcome(self):
        engine = PolicyEngine(default_decision="require_approval")
        engine.set_llm_safety_mode("advisory")
        engine.add_rule("llm", Rule(decision="deny"))
        decision = engine.evaluate(cap(effects=[Effect(kind="write", resource="workspace:*")]), {})
        assert decision.decision == "require_approval"

    def test_approval_routing_cannot_grant_bare_allow(self):
        engine = PolicyEngine(default_decision="deny")
        engine.set_llm_safety_mode("approval_routing")
        engine.add_rule("llm", Rule(decision="allow"))
        decision = engine.evaluate(cap(effects=[Effect(kind="write", resource="workspace:*")]), {})
        # LLM saying "allow" must never itself grant allow; falls back to default.
        assert decision.decision == "deny"

    def test_approval_routing_can_escalate_to_require_approval(self):
        engine = PolicyEngine(default_decision="allow")
        engine.set_llm_safety_mode("approval_routing")
        engine.add_rule("llm", Rule(decision="deny"))
        decision = engine.evaluate(cap(effects=[Effect(kind="write", resource="workspace:*")]), {})
        assert decision.decision == "require_approval"
