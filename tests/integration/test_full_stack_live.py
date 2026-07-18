"""Full stack live test: DeepSeek (decide) + embeddinggemma GGUF (layer-2
vector discovery) driving the customer-support scenario end to end."""
from __future__ import annotations

import os

import pytest

from state_projection_loop import Session
from state_projection_loop.adapters import DeepSeekAdapter

from examples.customer_support.tools import SUPPORT_KERNEL, SupportBackend, build_support_registry

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.skipif(
        not (os.environ.get("DEEPSEEK_API_KEY") and os.environ.get("SPAL_RUN_LIVE") == "1"),
        reason="live tests need DEEPSEEK_API_KEY and SPAL_RUN_LIVE=1",
    ),
]

llama_cpp = pytest.importorskip("llama_cpp", reason="llama-cpp-python not installed")


def test_vector_candidates_feed_the_live_model():
    from state_projection_loop.embeddings import LlamaCppEmbedding

    backend = SupportBackend()
    session = Session(
        DeepSeekAdapter(temperature=0.0),
        kernel=SUPPORT_KERNEL,
        registry=build_support_registry(backend),
        embedder=LlamaCppEmbedding(),
    )
    reply = session.send("コーヒーメーカー SmartBrew SB-2 の B2 表示が消えません。どうしたらいい?")
    assert any(k in reply for k in ("石灰", "デスケール", "クエン酸"))
    # layer-2 vector discovery actually surfaced tools this turn
    render = session.logger.of_type("render")[0]
    assert "search_manual" in render["candidates"]
