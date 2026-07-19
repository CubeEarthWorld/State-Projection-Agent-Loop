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
        default_factory=lambda: ["kernel", "toc", "conversation", "working_state", "candidates"]
    )
    window_tokens: int = 30000
    # Reserved so the model always has room to answer; counted against the
    # window budget alongside messages and native tool schemas (P0-5).
    reserved_output_tokens: int = 1024
    # Provider-side fixed overhead not visible in the message list itself
    # (e.g. a vendor's per-request wrapping tokens); 0 is a safe default.
    provider_overhead_tokens: int = 0
    # When native tool schemas are sent to the provider, the candidates
    # section only needs the one-line signature, not the full card
    # description a second time (P0-5 dedup).
    dedupe_candidate_cards_against_schemas: bool = True


@dataclass
class DiscoveryConfig:
    vector: str = "auto"  # "auto" | "on" | "off"
    k: int = 8
    toc: bool = True
    query_sources: list[str] = field(
        default_factory=lambda: ["last_user_message", "last_model_thought", "goal_if_exists"]
    )


@dataclass
class CompactionConfig:
    trigger_ratio: float = 0.8
    model: str = "same"  # "same" | "none" (deterministic fallback) | model name
    contract: str = "v2"
    max_summary_ratio: float = 0.1  # legacy free-text fallback cap, used only when model="none"


@dataclass
class BudgetConfig:
    max_steps: int = 50
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


@dataclass
class PersistenceConfig:
    # Directory for the JSONL event ledger + snapshots. None keeps the
    # ledger in-memory only (no cross-process resume).
    ledger_directory: Optional[str] = None
    snapshot_every_n_events: int = 20


@dataclass
class Config:
    mode: str = "chat"  # "chat" | "job"
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    compaction: CompactionConfig = field(default_factory=CompactionConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    artifacts: ArtifactsConfig = field(default_factory=ArtifactsConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        cfg = cls()
        for key, value in data.items():
            if not hasattr(cfg, key):
                raise ValueError(f"Unknown config key: {key!r}")
            current = getattr(cfg, key)
            if dataclasses.is_dataclass(current) and isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if not hasattr(current, sub_key):
                        raise ValueError(f"Unknown config key: {key}.{sub_key}")
                    setattr(current, sub_key, sub_value)
            else:
                setattr(cfg, key, value)
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
