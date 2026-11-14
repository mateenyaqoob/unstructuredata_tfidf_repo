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

from src.load_data import load_documents

titles, documents = load_documents()
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


def rank_documents(query: str):
    """Print documents ranked by cosine similarity to the query."""
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, X).flatten()
    ranking = sorted(zip(titles, scores), key=lambda x: x[1], reverse=True)
    print(f"\n=== Query: {query!r} ===")
    for title, score in ranking:
        print(f"  {title:<25s} {score:.3f}")


# %%
queries = [
    "renewable energy and climate policy",
    "password security and phishing",
    "healthy sleep exercise and diet",
]
for q in queries:
    rank_documents(q)

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
print("v2 Matrix shape:", X_v2.shape)
print("v2 Vocabulary size:", len(vectorizer_v2.get_feature_names_out()))

# %% [markdown]
# ## Exit ticket
# 1. What does a high TF-IDF score mean for a term in one document?
# 2. Why does IDF reduce the weight of common words like "the" or "data"?
# 3. Name one real task where TF-IDF is useful.

# %%
if __name__ == "__main__":
    print("\nActivity complete.")


# %% ----------------------------------------------------------------
# Student Submission
# Student: Hassaan0987
# Date: 2026-10-14
#
# Findings:
# - 10 documents loaded, each >30 words
# - TF-IDF matrix built successfully with stop_words="english"
# - Top terms per document extracted
# - Query "renewable energy and climate policy" -> Climate Change ranked #1
# - Query "password security and phishing" -> Cybersecurity ranked #1
# - Query "healthy sleep exercise and diet" -> Public Health ranked #1
# - v2 vectorizer (bigrams, max_df=0.8, min_df=2) reduced vocabulary and
#   produced more meaningful multi-word terms.
#
# Exit Ticket Answers:
# 1. High TF-IDF means the term is frequent in this document but rare
#    across all documents -> good keyword for that document.
# 2. IDF reduces weight of common words because they appear in almost
#    every document, so they carry little discriminative information.
# 3. Real-world uses: search engines, document similarity, spam filtering,
#    recommendation systems, keyword extraction.
# ----------------------------------------------------------------
print("Task completed by Hassaan0987")
