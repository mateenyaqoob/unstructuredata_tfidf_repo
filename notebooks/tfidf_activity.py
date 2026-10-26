"""
TF-IDF Class Activity
==================================

Run from the repository root:
    python notebooks/tfidf_activity.py

Or run interactively cell-by-cell in a REPL / IDE by copying the
`# %%`-separated blocks below.
"""

# %% [markdown]
# ## Step 1 — Load the Data
# Load the 10 titles and their corresponding long-text documents, confirm 10 documents are loaded, and check the word count for each.

# %% ----------------------------------------------------------------
import sys
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.load_data import load_documents

titles, documents = load_documents()
print(f"{len(documents)} documents loaded.")
assert len(documents) == 10, f"Expected 10 documents, got {len(documents)}"

for t, d in zip(titles, documents):
    word_count = len(d.split())
    print(f"  - {t}: {word_count} words")
    assert word_count > 30, f"Expected >30 words for {t}, got {word_count}"

# %% [markdown]
# ## Step 2 — Build the TF-IDF Matrix
# Import `TfidfVectorizer` and fit it on `documents` using `stop_words="english"`.

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(documents)
vocab_size = len(vectorizer.get_feature_names_out())

print("Matrix shape:", X.shape)              # (10, n_unique_terms)
print("Vocabulary size:", vocab_size)

# %% [markdown]
# ## Step 3 — Inspect the Matrix as a DataFrame
# Convert to a pandas DataFrame, print a sample (3 docs × 10 terms), and print the top-5 TF-IDF terms per document.

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
# ## Step 4 — Query Similarity Challenge
# Rank documents by cosine similarity for a given query in descending order.

# %%
from sklearn.metrics.pairwise import cosine_similarity


def rank_documents(query: str):
    """Print and return documents ranked by cosine similarity to the query."""
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, X).flatten()
    ranking = sorted(zip(titles, scores), key=lambda x: x[1], reverse=True)
    print(f"\n=== Query: {query!r} ===")
    for title, score in ranking:
        print(f"  {title:<25s} {score:.3f}")
    return ranking


# %%
queries = [
    "renewable energy and climate policy",
    "password security and phishing",
    "healthy sleep exercise and diet",
]
for q in queries:
    rank_documents(q)

# %% [markdown]
# ## Step 5 — Optional Challenge
# Re-run the vectorizer with `ngram_range=(1, 2)`, `max_df=0.8`, and `min_df=2`. Compare the vocabulary size and top terms.

# %%
vectorizer_v2 = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_df=0.8,
    min_df=2,
)
X_v2 = vectorizer_v2.fit_transform(documents)
vocab_size_v2 = len(vectorizer_v2.get_feature_names_out())

print(f"\nv1 Vocabulary size: {vocab_size} -> v2 Vocabulary size: {vocab_size_v2}")
print("v2 Matrix shape:", X_v2.shape)

df_v2 = pd.DataFrame(
    X_v2.toarray(),
    columns=vectorizer_v2.get_feature_names_out(),
    index=titles,
)

print("\nTop 5 TF-IDF terms per document (v2 with bigrams & min_df=2):")
for t in titles:
    top5_v2 = df_v2.loc[t].sort_values(ascending=False).head(5)
    non_zero_terms = list(top5_v2[top5_v2 > 0].index)
    print(f"  - {t:<22s}: {non_zero_terms}")

# %% [markdown]
# ## Step 6 — Verify Work & Exit Ticket
#
# **Exit Ticket Answers:**
# 1. **What does a high TF-IDF score mean for a term in one document?**
#    The term appears frequently in that specific document (high Term Frequency) but rarely across the rest of the corpus (high Inverse Document Frequency), making it a distinctive keyword for that document.
# 2. **Why does IDF reduce the weight of common words like "the" or "data"?**
#    Words that appear in many or all documents have a low IDF value (approaching 0) because they do not help distinguish one document from another.
# 3. **Name one real task where TF-IDF is useful.**
#    Search engine document ranking (Information Retrieval), keyword extraction, or text classification (such as spam detection).

# %%
if __name__ == "__main__":
    print("\nActivity complete.")
