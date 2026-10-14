"""Utilities for loading the classroom dataset."""
import os
import sys
from typing import List, Tuple

# Ensure the repo root is importable so `data.documents` resolves regardless
# of the current working directory or how this module is invoked.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.documents import TITLES, DOCUMENTS  # noqa: E402


def load_documents() -> Tuple[List[str], List[str]]:
    """Return (titles, documents)."""
    return TITLES, DOCUMENTS


# Module-level exports so `from src.load_data import titles, documents` works.
titles: List[str] = list(TITLES)
documents: List[str] = list(DOCUMENTS)

__all__ = ["load_documents", "titles", "documents"]


if __name__ == "__main__":
    print(f"Loaded {len(documents)} documents.")
    for t, d in zip(titles, documents):
        print(f"- {t}: {len(d.split())} words")
