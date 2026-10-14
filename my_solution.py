"""
TF-IDF Student Task - complete solution
Is file ko project ke ROOT folder mein rakhein (jahan src/ aur data/ folders hain)
Run:  python my_solution.py
"""
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.load_data import load_documents

# ------------------------------------------------------------------
# STEP 1: Load the data
# ------------------------------------------------------------------
print("=" * 60)
print("STEP 1: Load the data")
print("=" * 60)
titles, documents = load_documents()
print(f"Documents loaded: {len(documents)}")
for t, d in zip(titles, documents):
    print(f"  - {t:<22s} {len(d.split())} words")

# ------------------------------------------------------------------
# STEP 2: Build the TF-IDF matrix
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: Build the TF-IDF matrix")
print("=" * 60)
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(documents)
print("Matrix shape   :", X.shape)
print("Vocabulary size:", len(vectorizer.get_feature_names_out()))

# ------------------------------------------------------------------
# STEP 3: Inspect the matrix as a DataFrame
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: DataFrame + top-5 terms")
print("=" * 60)
df = pd.DataFrame(X.toarray(),
                  columns=vectorizer.get_feature_names_out(),
                  index=titles)

print("\nSample (first 3 documents x 10 terms):")
print(df.iloc[:3, :10].round(3).to_string())

print("\nTop-5 TF-IDF terms per document:")
for title in titles:
    top5 = df.loc[title].sort_values(ascending=False).head(5)
    terms = ", ".join(f"{term} ({score:.3f})" for term, score in top5.items())
    print(f"  {title:<22s} -> {terms}")

# ------------------------------------------------------------------
# STEP 4: Query similarity challenge
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: Query similarity challenge")
print("=" * 60)


def rank_documents(query):
    """Transform query with the SAME vectorizer, compute cosine
    similarity with all docs, return ranking (highest first)."""
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, X).flatten()
    return sorted(zip(titles, scores), key=lambda x: x[1], reverse=True)


queries = {
    "renewable energy and climate policy": "Climate Change",
    "password security and phishing": "Cybersecurity",
    "healthy sleep exercise and diet": "Public Health",
}

for q, expected in queries.items():
    ranking = rank_documents(q)
    print(f"\nQuery: '{q}'")
    for rank, (title, score) in enumerate(ranking, start=1):
        print(f"  {rank:>2}. {title:<22s} {score:.3f}")
    status = "OK" if ranking[0][0] == expected else "MISMATCH"
    print(f"  -> Top result: {ranking[0][0]} (expected {expected}) [{status}]")

# ------------------------------------------------------------------
# STEP 5: Optional challenge (bigrams, max_df, min_df)
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: Optional challenge")
print("=" * 60)
vec2 = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                       max_df=0.8, min_df=2)
X2 = vec2.fit_transform(documents)
df2 = pd.DataFrame(X2.toarray(), columns=vec2.get_feature_names_out(),
                   index=titles)

print(f"Original vocabulary size : {len(vectorizer.get_feature_names_out())}")
print(f"New vocabulary size      : {len(vec2.get_feature_names_out())}")
print(f"Bigrams in new vocabulary: "
      f"{sum(' ' in t for t in vec2.get_feature_names_out())}")

print("\nTop-5 terms with new settings:")
for title in titles:
    top5 = df2.loc[title].sort_values(ascending=False).head(5)
    print(f"  {title:<22s} -> {', '.join(top5.index)}")

print("""
Comparison:
- Vocabulary becomes MUCH smaller (537 -> 69) because min_df=2 removes
  every term that appears in only one document, and max_df=0.8 removes
  terms that appear in more than 80% of documents.
- Our documents are very short (~70 words) and cover different topics,
  so very few terms or bigrams are shared between 2+ documents.
- Result: the top terms become generic (e.g. 'use', 'based') and LESS
  meaningful. Bigrams do NOT help here; they would help on a larger
  corpus where phrases like 'climate change' repeat across documents.
""")

# ------------------------------------------------------------------
# Exit ticket answers
# ------------------------------------------------------------------
print("=" * 60)
print("EXIT TICKET")
print("=" * 60)
print("""
1. High TF-IDF score = the term appears often in THIS document but is
   rare in the other documents, so it is a good identifier of the topic.
2. IDF = log(N / df). Words appearing in many documents (like 'the' or
   'data') have a large df, so IDF is small and their weight goes down.
3. Real tasks: search engines / document retrieval, spam filtering,
   keyword extraction, recommending similar articles.
""")
print("Activity complete.")