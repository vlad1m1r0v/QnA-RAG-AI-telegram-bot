"""
Generate a PNG diagram of the LangGraph and save it to docs/graph.png.

Usage:
    poetry run python scripts/plot_graph.py
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.llm.graph import build_graph
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    out_path = ROOT / "docs" / "graph.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    graph = build_graph(checkpointer=None)
    png_bytes = graph.get_graph().draw_mermaid_png()
    out_path.write_bytes(png_bytes)
    logger.info(f"Graph saved to {out_path}")


if __name__ == "__main__":
    main()