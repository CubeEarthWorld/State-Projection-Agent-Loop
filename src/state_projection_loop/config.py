"""Configuration with the spec §13 defaults.

Everything works with ``Config()`` untouched (invariant I11); features are
enabled additively.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ProjectionConfig:
    # "toc" is a separate epoch-cached section (defect-2 fix): the kernel
    # stays immutable (I4) while the tool index may change mid-session.
    sections: list[str] = field(
        default_factory=lambda: ["kernel", "toc", "summary", "conversation", "candidates"]
    )
    window_tokens: int = 30000


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
    contract: str = "v1"
    max_summary_ratio: float = 0.1  # summary length cap: folded tokens × ratio (§10.2)


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
class HandlesConfig:
    inline_threshold_tokens: int = 800
    preview_tokens: int = 120


@dataclass
class LimitsConfig:
    max_validation_retries: int = 2
    # Job mode: consecutive text-only (no tool call, no done) turns tolerated
    # before the runtime nudges the model to call done(result).
    max_idle_turns: int = 3


@dataclass
class Config:
    mode: str = "chat"  # "chat" | "job"
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    compaction: CompactionConfig = field(default_factory=CompactionConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    handles: HandlesConfig = field(default_factory=HandlesConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    log_path: Optional[str] = None

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
