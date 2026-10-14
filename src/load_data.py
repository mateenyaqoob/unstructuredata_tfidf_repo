"""Utilities for loading the classroom dataset."""
import os
import sys
from typing import List, Tuple

# Make the repo root importable when this file is run directly (python src/load_data.py)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.documents import TITLES, DOCUMENTS


def load_documents() -> Tuple[List[str], List[str]]:
    """Return (titles, documents)."""
    return TITLES, DOCUMENTS


if __name__ == "__main__":
    titles, docs = load_documents()
    print(f"Loaded {len(docs)} documents.")
    for t, d in zip(titles, docs):
        print(f"- {t}: {len(d.split())} words")
