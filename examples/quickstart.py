"""Minimal quickstart: one decorated tool + DeepSeek (or any OpenAI-compatible API).

    python examples/quickstart.py "東京の在庫を教えて"
"""
from __future__ import annotations

import sys

from state_projection_loop import Session, tool
from state_projection_loop.adapters import DeepSeekAdapter

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


@tool(category="inventory", embedding_text="在庫 いくつ 残り stock")
def get_stock(warehouse: str) -> dict:
    """倉庫の在庫数を返す。

    Args:
        warehouse: 倉庫名(例: 東京, 大阪)
    """
    return {"warehouse": warehouse, "stock": 42}


def main() -> None:
    session = Session(
        DeepSeekAdapter(),
        kernel="あなたは在庫管理アシスタント。在庫の質問には必ず get_stock を使う。",
    )
    session.registry.register(get_stock)
    question = sys.argv[1] if len(sys.argv) > 1 else "東京倉庫の在庫はいくつ?"
    print("user:", question)
    print("assistant:", session.send(question))


if __name__ == "__main__":
    main()
