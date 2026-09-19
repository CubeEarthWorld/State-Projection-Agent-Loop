"""state-projection-loop — State-Projection Agent Loop.

Truth lives in the append-only Event Ledger, never in the model's context;
every turn renders a minimal disposable Projection derived from it with
fidelity-graded compression. The loop: Project → Decide → Validate →
Authorize → Execute → Record → Continue/Wait/Complete.
"""

from .artifacts import ArtifactStore, ref as artifact_ref
from .builtin import BUILTIN_PACKS, DEFAULT_BUILTINS, install_builtins
from .builtin.skills import skill_capability
from .builtin.toolkits import install_toolkits
from .compaction import FOLD_INSTRUCTIONS, FOLD_SCHEMA, apply_fold_delta, parse_fold_reply
from .capability import Capability, capability
from .context import ToolContext, TurnContext
from .compression import compress_text, summarize_text, content_hash
from .config import Config
from .checklists import ChecklistStore
from .discovery import ScoredTool, ToolSearch
from .embeddings import EmbeddingBackend, HashingEmbedding
from .events import Event, EventLedger, InMemoryLedger, JsonlLedger, ObservedLedger, Snapshot, event_to_message
from .llm import FallbackAdapter, LLMAdapter, ScriptedLLM, extract_finish, parse_text_tool_calls
from .messages import Decision, Message, ToolCall, Usage
from .policy import PolicyEngine, PolicyDecision, Rule
from .projection import (
    CandidatesSection,
    ChecklistSection,
    HistorySection,
    KernelSection,
    Projection,
    Section,
    TocSection,
    WorkingStateSection,
    runtime_notes,
)
from .registry import Registry, ToolProvider
from .run import ApprovalRequest, Command, PendingQuestion, Question, Run, RunStateError
from .json_schema import validate_args, validate_value
from .runtime import BudgetState, ExecuteBatchResult, Hooks, Runtime, ToolResult
from .session import ConcurrencyError, Session
from .working_state import RecordedDecision, WorkingState

__version__ = "0.5.0"

__all__ = [
    "Config",
    "ChecklistStore",
    "ChecklistSection",
    "Session",
    "ConcurrencyError",
    "Registry",
    "ToolProvider",
    "Capability",
    "ToolContext",
    "capability",
    "Projection",
    "Section",
    "TurnContext",
    "KernelSection",
    "TocSection",
    "HistorySection",
    "CandidatesSection",
    "runtime_notes",
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
    "Question",
    "PendingQuestion",
    "ObservedLedger",
    "validate_value",
    "skill_capability",
    "install_toolkits",
    "FOLD_SCHEMA",
    "FOLD_INSTRUCTIONS",
    "parse_fold_reply",
    "apply_fold_delta",
    "Event",
    "EventLedger",
    "InMemoryLedger",
    "JsonlLedger",
    "Snapshot",
    "event_to_message",
    "ToolSearch",
    "ScoredTool",
    "EmbeddingBackend",
    "HashingEmbedding",
    "LLMAdapter",
    "FallbackAdapter",
    "Hooks",
    "ScriptedLLM",
    "extract_finish",
    "parse_text_tool_calls",
    "Decision",
    "Message",
    "ToolCall",
    "Usage",
    "compress_text",
    "summarize_text",
    "content_hash",
    "install_builtins",
    "DEFAULT_BUILTINS",
    "BUILTIN_PACKS",
    "__version__",
]
