"""state-projection-loop — State-Projection Agent Loop.

Truth lives outside the context; every turn projects a minimal disposable
view (spec §2.2). Three nouns — Registry, Projection, Runtime — and four
verbs: render → decide → execute → commit.
"""

from .config import Config
from .discovery import ScoredTool, ToolSearch
from .embeddings import EmbeddingBackend, HashingEmbedding, LlamaCppEmbedding
from .handles import ValueStore
from .hooks import HookBlock, Hooks
from .llm import LLMAdapter, ScriptedLLM, parse_text_tool_calls
from .messages import Decision, Message, ToolCall, Usage
from .projection import (
    CandidatesSection,
    ConversationSection,
    KernelSection,
    Projection,
    ProjectionError,
    Section,
    SummarySection,
    TocSection,
    TurnContext,
)
from .registry import Registry, ToolProvider
from .runtime import BudgetState, Runtime, ToolResult, validate_args
from .session import Session
from .tooldef import ToolContext, ToolDef, tool
from .builtin.meta import install_spawn
from .builtin.state import StateViewSection, install_state

__version__ = "0.1.0"

__all__ = [
    "Config",
    "Session",
    "Registry",
    "ToolProvider",
    "ToolDef",
    "ToolContext",
    "tool",
    "Projection",
    "ProjectionError",
    "Section",
    "TurnContext",
    "KernelSection",
    "TocSection",
    "SummarySection",
    "ConversationSection",
    "CandidatesSection",
    "StateViewSection",
    "Runtime",
    "ToolResult",
    "BudgetState",
    "validate_args",
    "ValueStore",
    "Hooks",
    "HookBlock",
    "ToolSearch",
    "ScoredTool",
    "EmbeddingBackend",
    "HashingEmbedding",
    "LlamaCppEmbedding",
    "LLMAdapter",
    "ScriptedLLM",
    "parse_text_tool_calls",
    "Decision",
    "Message",
    "ToolCall",
    "Usage",
    "install_state",
    "install_spawn",
    "__version__",
]
