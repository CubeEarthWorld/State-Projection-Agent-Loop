"""Trace a run with OpenTelemetry from the event stream alone.

    pip install opentelemetry-sdk
    python -m examples.otel_tracing      # prints the spans to the console

One span per run, one per model call (ended at ``model_response``, started
``latency_ms`` earlier), one per tool command (``command_started`` to its
outcome event, error status unless it completed). Folds, hook
interventions, approvals and notices are span events on the run. Every
attribute comes from event data, so a replayed ledger traces exactly like
a live one; swap ``ConsoleSpanExporter`` for an OTLP exporter to ship them.
"""
from __future__ import annotations

from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.trace import Span, Status, StatusCode, Tracer, set_span_in_context

from state_projection_loop import Event, PolicyEngine, Registry, ScriptedLLM, Session
from state_projection_loop.run import TERMINAL_STATES

_OUTCOMES = ("command_completed", "command_failed", "command_outcome_unknown")
_NOTABLE = ("state_folded", "hook_intervened", "approval_requested", "approval_resolved", "notice",
            "model_call_failed", "rewound", "branch_created")


class OtelObserver:
    """An ``on_event`` observer that turns a run's events into spans. A chat
    session never reaches a terminal state; call :meth:`close` when done."""

    def __init__(self, tracer: Optional[Tracer] = None) -> None:
        self.tracer = tracer or trace.get_tracer("state_projection_loop")
        self.run: Optional[Span] = None
        self.commands: dict[str, Span] = {}

    def __call__(self, event: Event) -> None:
        data, now = event.data, int(event.ts * 1e9)
        if self.run is None:
            self.run = self.tracer.start_span("run", start_time=now, attributes={"run.id": event.run_id})
        ctx = set_span_in_context(self.run)
        if event.type == "model_response":
            usage = data.get("usage") or {}
            span = self.tracer.start_span(
                "model_call", context=ctx, start_time=now - int(data.get("latency_ms") or 0) * 1_000_000,
                attributes={"llm.tool_calls": len(data["calls"]), "llm.finish": bool(data["finish"]),
                            **{f"llm.usage.{k}": v for k, v in usage.items() if v is not None}},
            )
            span.end(end_time=now)
        elif event.type == "command_started":
            self.commands[data["command_id"]] = self.tracer.start_span(
                data["capability"], context=ctx, start_time=now, attributes={"command.id": data["command_id"]},
            )
        elif event.type in _OUTCOMES:
            span = self.commands.pop(data["command_id"], None)
            if span is not None:
                if event.type != "command_completed":
                    span.set_status(Status(StatusCode.ERROR, data.get("error") or event.type))
                span.end(end_time=now)
        elif event.type in _NOTABLE:
            self.run.add_event(event.type, timestamp=now)
        elif event.type == "run_state_changed" and data.get("to") in TERMINAL_STATES:
            self.run.set_attribute("run.state", data["to"])
            self.close()

    def close(self) -> None:
        for span in self.commands.values():
            span.end()
        self.commands.clear()
        if self.run is not None:
            self.run.end()
            self.run = None


if __name__ == "__main__":
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)

    registry = Registry()
    registry.register({
        "name": "demo.echo", "category": "demo",
        "spec": {"description": "Echo text in upper case.",
                 "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
        "discovery": {"pinned": True}, "execution": {"retry_safety": "pure"},
        "effects": [{"kind": "read", "resource": "demo:*"}],
    }, handler=lambda text: text.upper())
    observer = OtelObserver()
    session = Session(ScriptedLLM([ScriptedLLM.call("demo.echo", text="hi"), "It said HI."]),
                      registry=registry, on_event=observer, policy=PolicyEngine(default_decision="allow"))
    print(session.send("shout hi"))
    observer.close()
