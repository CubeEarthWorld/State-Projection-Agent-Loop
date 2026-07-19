"""Minimal quickstart: one decorated capability + any OpenAI-compatible API
(OpenAI, DeepSeek, a local server — set via env vars below).

The LLM adapter (``OpenAICompatAdapter``) lives in ``examples/llm_adapters.py``,
not in the ``state_projection_loop`` package — the package only defines the
``LLMAdapter`` Protocol; talking to a real provider is entirely up to you.

    python -m examples.quickstart "東京の在庫を教えて"
"""
from __future__ import annotations

import os
import sys

from state_projection_loop import Session, capability

from examples.llm_adapters import OpenAICompatAdapter

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


@capability(name="inventory.stock.get", category="inventory", embedding_text="在庫 いくつ 残り stock",
            retry_safety="pure", effects=[("none", "*")])
def get_stock(warehouse: str) -> dict:
    """倉庫の在庫数を返す。

    Args:
        warehouse: 倉庫名(例: 東京, 大阪)
    """
    return {"warehouse": warehouse, "stock": 42}


def main() -> None:
    llm = OpenAICompatAdapter(
        model=os.environ.get("LLM_MODEL", "deepseek-v4-flash"),
        api_key=os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY"),
        base_url=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"),
    )
    session = Session(
        llm,
        kernel="あなたは在庫管理アシスタント。在庫の質問には必ず inventory.stock.get を使う。答え終えたら finish(result) を呼ぶ。",
    )
    session.registry.register(get_stock)
    question = sys.argv[1] if len(sys.argv) > 1 else "東京倉庫の在庫はいくつ?"
    print("user:", question)
    print("assistant:", session.run_job(question))


if __name__ == "__main__":
    main()
