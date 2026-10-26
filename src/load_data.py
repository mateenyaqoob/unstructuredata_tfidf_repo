"""Utilities for loading the classroom dataset."""
from typing import List, Tuple

from unstructuredata_tfidf_repo.data.documents import TITLES, DOCUMENTS


def load_documents() -> Tuple[List[str], List[str]]:
    """Return (titles, documents)."""
    return TITLES, DOCUMENTS


if __name__ == "__main__":
    titles, docs = load_documents()
    print(f"Loaded {len(docs)} documents.")
    for t, d in zip(titles, docs):
        print(f"- {t}: {len(d.split())} words")