"""A minimal MCP server over stdio for the McpProvider tests: two tools,
one read-only and one destructive, plus an error case."""
from __future__ import annotations

import json
import sys

TOOLS = [
    {"name": "echo", "description": "Echo the text back.",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
     "annotations": {"readOnlyHint": True, "idempotentHint": True}},
    {"name": "delete-all", "description": "Delete everything.",
     "inputSchema": {"type": "object", "properties": {}}},
]


def main() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        message = json.loads(line)
        method, request_id = message.get("method"), message.get("id")
        if request_id is None:
            continue  # a notification
        if method == "initialize":
            result = {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                      "serverInfo": {"name": "fake", "version": "0"}}
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            name, arguments = message["params"]["name"], message["params"].get("arguments") or {}
            if name == "echo":
                if arguments.get("text") == "boom":
                    result = {"content": [{"type": "text", "text": "echo refused"}], "isError": True}
                else:
                    result = {"content": [{"type": "text", "text": "echo: " + arguments["text"]}]}
            else:
                result = {"content": [{"type": "text", "text": "deleted"}]}
        else:
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": request_id,
                                         "error": {"code": -32601, "message": "unknown method"}}) + "\n")
            sys.stdout.flush()
            continue
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
