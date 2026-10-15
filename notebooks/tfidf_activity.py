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

# %%
# Compare vocabulary, surviving bigrams and top terms (v1 vs v2)
features_v2 = vectorizer_v2.get_feature_names_out()
bigrams_v2 = [f for f in features_v2 if " " in f]
print(f"\nVocabulary: v1 = {X.shape[1]}, v2 = {X_v2.shape[1]}")
print(f"Bigrams kept by v2 ({len(bigrams_v2)}): {bigrams_v2}")

df_v2 = pd.DataFrame(X_v2.toarray(), columns=features_v2, index=titles)
print("\nTop 5 terms per document — v1 (unigrams) vs v2 (bigrams, max_df/min_df):")
for t in titles:
    v1_top = ", ".join(df.loc[t].sort_values(ascending=False).head(5).index)
    v2_top = ", ".join(df_v2.loc[t].sort_values(ascending=False).head(5).index)
    print(f"\n{t}:\n  v1: {v1_top}\n  v2: {v2_top}")

print("\nQuery rankings with v2 (top 3):")
for q in queries:
    scores = cosine_similarity(vectorizer_v2.transform([q]), X_v2).flatten()
    ranking = sorted(zip(titles, scores), key=lambda x: x[1], reverse=True)[:3]
    print(f"  {q!r}: " + ", ".join(f"{t} ({s:.3f})" for t, s in ranking))

# %% [markdown]
# ### Findings — do bigrams make the terms more meaningful?
#
# **Not on this corpus — they make things worse.**
#
# - Vocabulary shrinks from **537 → 69** terms. Only **3 bigrams** survive
#   (`economic growth`, `policy involves`, `trade offs`).
# - `min_df=2` keeps only terms that appear in at least 2 of the 10 documents.
#   Each document is about a distinct topic, so the most *distinctive* words
#   (`pasta`, `mars`, `cybersecurity`, `soil`, `inflation`) appear in just one
#   document and are removed. Bigrams are even rarer than unigrams, so almost
#   all of them are dropped too.
# - Top terms become generic: Italian Pasta → `different, long, improve`;
#   Space Exploration → `international, medical, health`.
# - Retrieval degrades: "password security and phishing" scores **0 for every
#   document** (none of those words survive), and "healthy sleep exercise and
#   diet" now ranks *Gardening Vegetables* above Public Health because `sleep`
#   and `exercise` are gone.
# - Bigrams on their own (`ngram_range=(1, 2)` without `min_df=2`) grow the
#   vocabulary to 1182 and *can* add meaningful phrases. `min_df`/`max_df` are
#   meant for large corpora, where pruning rare terms removes noise; with 10
#   documents it removes the signal.
#
# ## Exit ticket
# 1. **What does a high TF-IDF score mean for a term in one document?**
#    The term is frequent in that document but rare across the corpus, so it is
#    characteristic of that document (e.g. `pasta` in Italian Pasta).
# 2. **Why does IDF reduce the weight of common words like "the" or "data"?**
#    IDF = log(N / df). A word found in many documents has a large df, so its
#    IDF — and therefore its weight — is small. It doesn't help tell documents
#    apart (`data` appears in AI, Climate, Cybersecurity, Space, Public Health).
# 3. **Name one real task where TF-IDF is useful.**
#    Search/ranking (matching a query to documents, as in Step 4), keyword
#    extraction, duplicate detection, or features for spam/text classification.
#
# **Limitation:** TF-IDF matches exact tokens only — no stemming or semantics.
# A query like "eating habits" would not match the word "diet", and "health"
# and "healthy" are separate terms. Word order is also ignored.

# %%
if __name__ == "__main__":
    print("\nActivity complete.")
