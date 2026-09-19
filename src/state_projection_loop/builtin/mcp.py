"""An MCP client as a :class:`~state_projection_loop.registry.ToolProvider`.

One server over stdio; every tool it lists becomes a capability
``mcp.<server>.<tool>``. MCP's tool annotations map onto the contract the
policy engine and the runtime reason about, and a tool that declares
nothing lands in the most restrictive class — an external, never-retried
effect — exactly as an undeclared local capability would.

Standard library only (the wire format is newline-delimited JSON-RPC).
"""
from __future__ import annotations

import json
import re
import subprocess
import threading
from typing import Any, Optional

from ..capability import Capability

PROTOCOL_VERSION = "2025-06-18"
_SEGMENT = re.compile(r"[^a-z0-9_]+")


def _segment(name: str) -> str:
    """An MCP tool name as one dotted-name segment."""
    seg = _SEGMENT.sub("_", name.lower()).strip("_") or "tool"
    return seg if seg[0].isalpha() else "t_" + seg


def _contract(annotations: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    """MCP annotations -> (effects, retry_safety). Absent hints are read
    the conservative way round: not read-only, destructive, not idempotent."""
    read_only = bool(annotations.get("readOnlyHint", False))
    destructive = bool(annotations.get("destructiveHint", True))
    idempotent = bool(annotations.get("idempotentHint", False))
    if read_only:
        return [{"kind": "read", "resource": "mcp:*"}], "idempotent" if idempotent else "check_then_retry"
    kind = "external" if destructive else "write"
    return [{"kind": kind, "resource": "mcp:*"}], "idempotent" if idempotent else "never_retry"


class McpProvider:
    """``registry.attach_provider(McpProvider("fs", ["npx", "-y", "@modelcontextprotocol/server-filesystem", "."]))``.

    The subprocess starts on first use and lives until :meth:`close`.
    """

    def __init__(self, name: str, command: list[str], *, env: Optional[dict[str, str]] = None,
                 timeout_s: float = 30.0) -> None:
        self.name = _segment(name)
        self.command = list(command)
        self.env = env
        self.timeout_s = timeout_s
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._next_id = 0

    # -- transport ------------------------------------------------------------

    def _start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        self._proc = subprocess.Popen(
            self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=self.env,
            text=True, encoding="utf-8", bufsize=1,
        )
        self._request("initialize", {
            "protocolVersion": PROTOCOL_VERSION, "capabilities": {},
            "clientInfo": {"name": "state-projection-loop", "version": "0.5.0"},
        })
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _send(self, message: dict[str, Any]) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(message) + "\n")
        self._proc.stdin.flush()

    def _request(self, method: str, params: dict[str, Any]) -> Any:
        assert self._proc is not None and self._proc.stdout is not None
        self._next_id += 1
        request_id = self._next_id
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError(f"MCP server {self.name!r} closed the connection during {method}")
            reply = json.loads(line)
            if reply.get("id") != request_id:
                continue  # a notification or an unrelated message
            if "error" in reply:
                raise RuntimeError(f"MCP {method} failed: {reply['error'].get('message', reply['error'])}")
            return reply.get("result")

    def call(self, method: str, params: dict[str, Any]) -> Any:
        with self._lock:
            self._start()
            return self._request(method, params)

    def close(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            self._proc = None

    # -- ToolProvider ---------------------------------------------------------

    def provide(self) -> list[Capability]:
        tools = self.call("tools/list", {}).get("tools", [])
        return [self._capability(tool) for tool in tools]

    def _capability(self, tool: dict[str, Any]) -> Capability:
        effects, retry_safety = _contract(tool.get("annotations") or {})
        mcp_name = tool["name"]
        description = tool.get("description") or f"MCP tool {mcp_name} of server {self.name}"

        def handler(**arguments: Any) -> Any:
            result = self.call("tools/call", {"name": mcp_name, "arguments": arguments})
            texts = [c.get("text", "") for c in result.get("content", []) if c.get("type") == "text"]
            body: Any = result.get("structuredContent") or ("\n".join(texts) if texts else result.get("content"))
            if result.get("isError"):
                raise RuntimeError("\n".join(texts) or "MCP tool reported an error")
            return body

        return Capability.from_dict({
            "name": f"mcp.{self.name}.{_segment(mcp_name)}",
            "category": f"mcp/{self.name}",
            "spec": {"description": description,
                     "parameters": tool.get("inputSchema") or {"type": "object", "properties": {}}},
            "execution": {"timeout_s": self.timeout_s, "retry_safety": retry_safety},
            "effects": effects,
        }, handler=handler)
