"""Resident checklist tool; all operations use the session's working state."""
import json

from ..capability import ToolContext
from ..checklists import ChecklistStore

CHECKLIST_DEF = json.loads(r'''
{
  "name": "planning.checklist.manage",
  "category": "planning",
  "card": {
    "summary": "Create, inspect, revise and hand off versioned checklists.",
    "signature": "planning.checklist.manage(action, id?, expected_revision?, ...) -> checklist"
  },
  "spec": {
    "description": "Manage durable plans. create requires name; list/get inspect progress; update replaces supplied fields (items is an ordered full replacement). add_item requires item.text; update_item patches text/status/notes by item_id; delete_item removes by item_id. All edits and delete require id and expected_revision from get. export takes optional id; import takes document. Only use arguments relevant to the action. Keep at most one item in_progress per checklist. Mark completed only after verification, blocked with notes explaining why. include_in_context controls automatic projection, not access or history. Export/import copies plans without shared state; imported IDs must not exist locally.",
    "parameters": {
      "type": "object",
      "properties": {
        "action": {
          "type": "string",
          "enum": [
            "create",
            "list",
            "get",
            "update",
            "delete",
            "add_item",
            "update_item",
            "delete_item",
            "export",
            "import"
          ]
        },
        "id": {
          "type": "string",
          "description": "Checklist ULID returned by create/list."
        },
        "expected_revision": {
          "type": "integer",
          "minimum": 1,
          "description": "Required for every edit/delete. Read get first; stale revisions are rejected."
        },
        "name": {
          "type": "string",
          "minLength": 1,
          "maxLength": 200
        },
        "include_in_context": {
          "type": "boolean"
        },
        "context_mode": {
          "type": "string",
          "enum": [
            "name",
            "summary",
            "full"
          ]
        },
        "mode": {
          "type": "string",
          "enum": [
            "name",
            "summary",
            "full"
          ]
        },
        "items": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "id": {
                "type": "string"
              },
              "text": {
                "type": "string",
                "minLength": 1,
                "maxLength": 500
              },
              "status": {
                "type": "string",
                "enum": [
                  "pending",
                  "in_progress",
                  "blocked",
                  "completed",
                  "cancelled"
                ]
              },
              "notes": {
                "type": "string",
                "maxLength": 2000
              }
            },
            "additionalProperties": false
          },
          "description": "Ordered full replacement on update. Preserve existing item IDs. At most 200 items; at most one in_progress."
        },
        "item": {
          "type": "object",
          "properties": {
            "id": {
              "type": "string"
            },
            "text": {
              "type": "string",
              "minLength": 1,
              "maxLength": 500
            },
            "status": {
              "type": "string",
              "enum": [
                "pending",
                "in_progress",
                "blocked",
                "completed",
                "cancelled"
              ]
            },
            "notes": {
              "type": "string",
              "maxLength": 2000
            }
          },
          "additionalProperties": false
        },
        "item_id": {
          "type": "string"
        },
        "document": {
          "type": "object",
          "description": "Version 1 document from export; imports preserve IDs and reject collisions."
        }
      },
      "required": [
        "action"
      ],
      "additionalProperties": false
    }
  },
  "discovery": {
    "pinned": true,
    "no_embed": true
  },
  "execution": {
    "timeout_s": 5,
    "retry_safety": "never_retry",
    "resolve_handles": false
  },
  "effects": [
    {
      "kind": "write",
      "resource": "working_state:checklists"
    }
  ]
}
''')


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
        registry.register(CHECKLIST_DEF, handler=_checklist)
