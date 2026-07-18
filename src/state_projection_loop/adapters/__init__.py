"""Model adapters (spec §15: no vendor lock-in)."""

from .openai_compat import DeepSeekAdapter, OpenAICompatAdapter

__all__ = ["OpenAICompatAdapter", "DeepSeekAdapter"]
