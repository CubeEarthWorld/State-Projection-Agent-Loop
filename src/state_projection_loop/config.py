"""Configuration.

Everything works with ``Config()`` untouched; features are enabled
additively.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ProjectionConfig:
    # "toc" is a separate epoch-cached section: the kernel stays immutable
    # while the tool index may change mid-session.
    # "working_state" and "candidates" are both volatile (may change every
    # turn) and must stay last, in that order, after the append-only
    # conversation section.
    sections: list[str] = field(
        default_factory=lambda: ["kernel", "toc", "history", "working_state", "checklists", "candidates"]
    )
    window_tokens: int = 30000
    # Reserved so the model always has room to answer; counted against the
    # window budget alongside messages and native tool schemas.
    reserved_output_tokens: int = 1024
    # Provider-side fixed overhead not visible in the message list itself
    # (e.g. a vendor's per-request wrapping tokens); 0 is a safe default.
    provider_overhead_tokens: int = 0
    # When native tool schemas are sent to the provider, the candidates
    # section only needs the one-line signature, not the full card
    # description a second time (dedup).
    dedupe_candidate_cards_against_schemas: bool = True


@dataclass
class DiscoveryConfig:
    vector: str = "auto"  # "auto" | "on" | "off"
    k: int = 8
    toc: bool = True
    # Recently used non-pinned tools whose native schemas are re-sent each turn.
    active_tools: int = 48
    query_sources: list[str] = field(
        default_factory=lambda: ["last_user_message", "last_model_thought", "goal_if_exists"]
    )


@dataclass
class CompressionConfig:
    # History renders in tiers measured from a verbatim point that moves in
    # steps (Session._advance_tiers): the newest `full_window` messages are
    # verbatim once the tail grows past four times that; before the point, the
    # next `compressed_window` messages are compressed (tool results masked
    # to one line unless they report an error, assistant text head+tail),
    # the next `summary_window` are one-line summaries, older ones are
    # dropped. The user's own messages are never compressed or dropped.
    full_window: int = 6
    compressed_window: int = 24
    summary_window: int = 60
    compressed_max_lines: int = 80
    observation_max_lines: int = 40


@dataclass
class BudgetConfig:
    # None means no step limit, as for the other caps below; the runtime
    # already tests `is not None`, and the Dart port already types it so.
    max_steps: Optional[int] = 50
    max_tokens: Optional[int] = None
    max_cost: Optional[float] = None
    max_seconds: Optional[float] = None
    # Needed only when max_cost is set and the adapter reports usage.
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0


@dataclass
class ArtifactsConfig:
    inline_threshold_tokens: int = 800
    preview_tokens: int = 120
    # When set, ArtifactStore persists large payloads to disk under this
    # directory (namespaced by run id) so a resumed run can recover them.
    directory: Optional[str] = None


@dataclass
class LimitsConfig:
    max_validation_retries: int = 2
    # Job mode: consecutive text-only (no tool call, no finish) turns
    # tolerated before the runtime nudges the model to call finish(result).
    max_idle_turns: int = 3
    # Default approval TTL; None means requests never expire on their own.
    approval_expires_s: Optional[float] = 3600.0
    # Loop guard: an identical call repeated this many times inside the last
    # repeat_window calls (all failing, or all returning the same result) is
    # not executed again; 0 disables the guard.
    max_repeats: int = 3
    repeat_window: int = 8


@dataclass
class PersistenceConfig:
    # Directory for the JSONL event ledger + snapshots. None keeps the
    # ledger in-memory only (no cross-process resume).
    ledger_directory: Optional[str] = None


@dataclass
class CompactionConfig:
    # When the rendered prompt exceeds this fraction of the window, one extra
    # model call folds old history into the working state (see compaction.py).
    # 0 disables compaction; deterministic compression always stays on.
    # Fold when the prompt exceeds this share of the room the render has
    # (window less reserved output), at the next step of the verbatim
    # point. 0 turns the fold off; see docs/compression.md for the cost.
    trigger_ratio: float = 0.75


@dataclass
class ModelConfig:
    # One model call: how long to wait, how often to retry a failed call
    # (any exception, including the timeout), and the pause between tries
    # (multiplied by the attempt number). Every failed attempt is a
    # `model_call_failed` ledger event; the last one also raises.
    timeout_s: Optional[float] = None
    retries: int = 0
    backoff_s: float = 1.0


@dataclass
class Config:
    mode: str = "chat"  # "chat" | "job"
    # Job mode: JSON Schema finish(result) must satisfy; a failing result is
    # bounced back to the model like an argument error.
    result_schema: Optional[dict[str, Any]] = None
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    artifacts: ArtifactsConfig = field(default_factory=ArtifactsConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    compaction: CompactionConfig = field(default_factory=CompactionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        cfg = cls()
        for key, value in data.items():
            if not hasattr(cfg, key):
                raise ValueError(f"Unknown config key: {key!r}")
            current = getattr(cfg, key)
            if dataclasses.is_dataclass(current):
                if not isinstance(value, dict):
                    raise ValueError(f'Config key "{key}" expects a map')
                for sub_key, sub_value in value.items():
                    if not hasattr(current, sub_key):
                        raise ValueError(f"Unknown config key: {key}.{sub_key}")
                    setattr(current, sub_key, sub_value)
            else:
                setattr(cfg, key, value)
        return cfg
