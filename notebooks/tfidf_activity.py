"""
TF-IDF Class Activity
==================================

Run from the repository root:
    python notebooks/tfidf_activity.py

Or run interactively cell-by-cell in a REPL / IDE by copying the
`# %%`-separated blocks below.
"""

# %% ----------------------------------------------------------------
# Setup: make sure the repo root is on sys.path so we can import src/ and data/
import sys
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.load_data import titles, documents

print(f"{len(documents)} documents loaded.")
for t, d in zip(titles, documents):
    print(f"  - {t}: {len(d.split())} words")

# %% [markdown]
# ## Step 1 — Build the TF-IDF matrix
#
# TODO: import `TfidfVectorizer` and fit it on `documents`.

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(documents)
print("Matrix shape:", X.shape)              # (10, n_unique_terms)
print("Vocabulary size:", len(vectorizer.get_feature_names_out()))

# %% [markdown]
# ## Step 2 — Inspect the matrix as a DataFrame
#
# TODO: convert to a pandas DataFrame and print the top-5 TF-IDF terms per document.

# %%
import pandas as pd

df = pd.DataFrame(
    X.toarray(),
    columns=vectorizer.get_feature_names_out(),
    index=titles,
)

print("\nSample of the TF-IDF matrix (first 3 docs × 10 terms):")
print(df.iloc[:3, :10].round(3))

print("\nTop 5 TF-IDF terms per document:")
for t in titles:
    top5 = df.loc[t].sort_values(ascending=False).head(5)
    print(f"\n{t}:")
    for term, score in top5.items():
        print(f"  {term:<20s} {score:.3f}")

# %% [markdown]
# ## Step 3 — Query similarity challenge
#
# TODO: rank documents by cosine similarity for a given query.

# %%
from sklearn.metrics.pairwise import cosine_similarity


def rank_documents(query: str, verbose: bool = True):
    """Return [(title, score)] ranked by cosine similarity to the query (desc)."""
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, X).flatten()
    ranking = sorted(zip(titles, scores), key=lambda x: x[1], reverse=True)
    if verbose:
        print(f"\n=== Query: {query!r} ===")
        for title, score in ranking:
            print(f"  {title:<25s} {score:.3f}")
    return ranking


# %%
expected_top = {
    "renewable energy and climate policy": "Climate Change",
    "password security and phishing": "Cybersecurity",
    "healthy sleep exercise and diet": "Public Health",
}
for q, expected in expected_top.items():
    top_title = rank_documents(q)[0][0]
    assert top_title == expected, f"{q!r}: expected {expected!r}, got {top_title!r}"
    print(f"  -> OK: {expected!r} ranked first")

# %% [markdown]
# ## Challenge
#
# Try the queries below and explain the rankings:
# - "password security and phishing"
# - "healthy sleep exercise and diet"
#
# Then re-run the vectorizer with:
#     TfidfVectorizer(stop_words="english",
#                     ngram_range=(1, 2),
#                     max_df=0.8,
#                     min_df=2)
# How does the vocabulary size change? Do the top terms become more meaningful?

# %%
vectorizer_v2 = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_df=0.8,
    min_df=2,
)
X_v2 = vectorizer_v2.fit_transform(documents)
print("\nv2 Matrix shape:", X_v2.shape)
print("v1 Vocabulary size:", len(vectorizer.get_feature_names_out()))
print("v2 Vocabulary size:", len(vectorizer_v2.get_feature_names_out()))

df_v2 = pd.DataFrame(
    X_v2.toarray(),
    columns=vectorizer_v2.get_feature_names_out(),
    index=titles,
)
print("\nTop 5 v2 (unigram + bigram) terms per document:")
for t in titles:
    top5 = df_v2.loc[t].sort_values(ascending=False).head(5)
    print(f"  {t}: {', '.join(top5.index)}")

# %% [markdown]
# ## Exit ticket
# 1. What does a high TF-IDF score mean for a term in one document?
# 2. Why does IDF reduce the weight of common words like "the" or "data"?
# 3. Name one real task where TF-IDF is useful.

# %%
if __name__ == "__main__":
    print("\nActivity complete.")
