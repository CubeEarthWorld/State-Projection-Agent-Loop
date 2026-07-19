"""Full stack live test: an OpenAI-compatible chat model (decide) +
embeddinggemma GGUF (layer-2 vector discovery) driving the customer-support
scenario end to end. Both adapters live in ``examples/llm_adapters.py``.
"""
from __future__ import annotations

import os

import pytest

from state_projection_loop import Session
from state_projection_loop.policy import PolicyEngine

from examples.customer_support.tools import SUPPORT_KERNEL, SupportBackend, build_support_registry
from examples.llm_adapters import OpenAICompatAdapter

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(
        not ((os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY"))
             and os.environ.get("SPAL_RUN_LIVE") == "1"),
        reason="live tests need LLM_API_KEY (or DEEPSEEK_API_KEY) and SPAL_RUN_LIVE=1",
    ),
]

llama_cpp = pytest.importorskip("llama_cpp", reason="llama-cpp-python not installed")


def test_vector_candidates_feed_the_live_model():
    from examples.llm_adapters import LlamaCppEmbedding

    backend = SupportBackend()
    session = Session(
        OpenAICompatAdapter(
            model=os.environ.get("LLM_MODEL", "deepseek-v4-flash"),
            api_key=os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY"),
            base_url=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"),
            temperature=0.0,
        ),
        kernel=SUPPORT_KERNEL,
        registry=build_support_registry(backend),
        embedder=LlamaCppEmbedding(),
        policy=PolicyEngine(default_decision="allow"),
    )
    reply = session.send("コーヒーメーカー SmartBrew SB-2 の B2 表示が消えません。どうしたらいい?")
    assert any(k in reply for k in ("石灰", "デスケール", "クエン酸"))
    # layer-2 vector discovery actually surfaced tools this turn
    render_events = [e for e in session.ledger.iter_run(session.run.id) if e.type == "projection_compiled"]
    assert "support.manual.search" in render_events[0].data["candidates"]
