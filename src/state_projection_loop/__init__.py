"""state-projection-loop — State-Projection Agent Loop.

Truth lives in the append-only Event Ledger, never in the model's context;
every turn renders a minimal disposable Projection derived from it. The
loop: Project → Decide → Validate → Plan Effects → Authorize → Execute →
Record → Continue/Wait/Complete.
"""

from .artifacts import ArtifactStore, ref as artifact_ref
from .capability import Capability, ToolContext, capability
from .config import Config
from .discovery import ScoredTool, ToolSearch
from .embeddings import EmbeddingBackend, HashingEmbedding
from .events import Event, EventLedger, InMemoryLedger, JsonlLedger, Snapshot
from .llm import LLMAdapter, ScriptedLLM, extract_finish, parse_text_tool_calls
from .messages import Decision, Message, ToolCall, Usage
from .policy import PolicyEngine, PolicyDecision, Rule
from .projection import (
    CandidatesSection,
    ConversationSection,
    KernelSection,
    Projection,
    ProjectionError,
    Section,
    TocSection,
    TurnContext,
)
from .registry import Registry, ToolProvider
from .run import ApprovalRequest, Command, Run, RunStateError
from .runtime import BudgetState, ExecuteBatchResult, Runtime, ToolResult, validate_args
from .session import ConcurrencyError, Session
from .working_state import RecordedDecision, WorkingState, WorkingStateSection
from .builtin.meta import install_spawn
from .builtin.state import install_state

__version__ = "0.2.0"

__all__ = [
    "Config",
    "Session",
    "ConcurrencyError",
    "Registry",
    "ToolProvider",
    "Capability",
    "ToolContext",
    "capability",
    "Projection",
    "ProjectionError",
    "Section",
    "TurnContext",
    "KernelSection",
    "TocSection",
    "ConversationSection",
    "CandidatesSection",
    "WorkingState",
    "WorkingStateSection",
    "RecordedDecision",
    "Runtime",
    "ToolResult",
    "ExecuteBatchResult",
    "BudgetState",
    "validate_args",
    "ArtifactStore",
    "artifact_ref",
    "PolicyEngine",
    "PolicyDecision",
    "Rule",
    "Run",
    "RunStateError",
    "Command",
    "ApprovalRequest",
    "Event",
    "EventLedger",
    "InMemoryLedger",
    "JsonlLedger",
    "Snapshot",
    "ToolSearch",
    "ScoredTool",
    "EmbeddingBackend",
    "HashingEmbedding",
    "LLMAdapter",
    "ScriptedLLM",
    "extract_finish",
    "parse_text_tool_calls",
    "Decision",
    "Message",
    "ToolCall",
    "Usage",
    "install_state",
    "install_spawn",
    "__version__",
]
