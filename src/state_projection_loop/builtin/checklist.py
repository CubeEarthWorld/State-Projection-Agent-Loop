"""Resident checklist tool; all operations use the session's working state."""
from ..capability import ToolContext
from ..checklists import ChecklistStore
from .defs import load



async def _checklist(ctx: ToolContext, action: str, **arguments):
    if action in ("list", "get", "export"):
        return ctx.working_state.checklists.execute(action, **arguments)
    updated = ChecklistStore.from_dict(ctx.working_state.checklists.to_dict())
    result = updated.execute(action, **arguments)
    if ctx.ledger is not None and ctx.run is not None:
        ctx.ledger.append(ctx.run.id, "checklists_changed", {
            "action": action, "command_id": ctx.command_id, "checklists": updated.to_dict(),
        })
    ctx.working_state.checklists = updated
    return result


def ensure_checklist_tool(registry):
    if "planning.checklist.manage" not in registry:
        registry.register(load("checklist"), handler=_checklist)
