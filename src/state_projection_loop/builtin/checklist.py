"""Handler of the ``checklist`` pack; all operations use the session's working state."""
from ..context import ToolContext
from ..checklists import ChecklistStore


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


CHECKLIST_HANDLERS = {"planning.checklist.manage": _checklist}
