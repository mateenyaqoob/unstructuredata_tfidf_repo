"""End-to-end walkthrough of the TF-IDF 30-minute class activity."""
import os
import sys

# Make the repo root importable when this file is run directly (python src/run_activity.py)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.load_data import load_documents
from src.tfidf_utils import (
    build_tfidf,
    to_dataframe,
    top_terms_per_document,
    rank_documents_for_query,
)


def main() -> None:
    titles, documents = load_documents()
    print(f"Loaded {len(documents)} documents.\n")

    # 1) Build TF-IDF matrix
    vectorizer, matrix = build_tfidf(documents)
    print("TF-IDF matrix shape:", matrix.shape)

    df = to_dataframe(matrix, vectorizer, titles)
    print("\nSample of the TF-IDF matrix:")
    print(df.iloc[:3, :10].round(3))

    # 2) Top terms per document
    print("\n=== Top 5 TF-IDF terms per document ===")
    for title, terms in top_terms_per_document(df, n=5).items():
        joined = ", ".join(f"{k}({v:.2f})" for k, v in terms.items())
        print(f"\n{title}:\n  {joined}")

    # 3) Query similarity challenge
    queries = [
        "renewable energy and climate policy",
        "password security and phishing",
        "healthy sleep exercise and diet",
    ]
    for q in queries:
        print(f"\n=== Query: {q!r} ===")
        for title, score in rank_documents_for_query(q, vectorizer, matrix, titles):
            print(f"  {title:25s} {score:.3f}")


    #ans

    #part 2:

    from sklearn.feature_extraction.text import TfidfVectorizer

    print("\n Part 2: TF-IDF matrix ")
    my_vec = TfidfVectorizer(stop_words="english")
    my_matrix = my_vec.fit_transform(documents)          # sparse (10, n_terms)
    print("Matrix shape   :", my_matrix.shape)
    print("Vocabulary size:", len(my_vec.vocabulary_))


    #part 3:
    import pandas as pd

    print("\n Part 3: TF-IDF DataFrame ")
    my_df = pd.DataFrame(
        my_matrix.toarray(),
        index=titles,
        columns=my_vec.get_feature_names_out(),
    )
    print(my_df.iloc[:3, :10].round(3))

    print("\nTop-5 TF-IDF terms per document:")
    for title in my_df.index:
        top5 = my_df.loc[title].sort_values(ascending=False).head(5)
        print(f"  {title:22s} -> " + ", ".join(f"{t}({s:.2f})" for t, s in top5.items()))


    #part 4:

    print("\n Part 4: Query similarity ")

    from sklearn.metrics.pairwise import cosine_similarity

    def rank_documents(query, vec, mat, titles):
        q_vec = vec.transform([query])                    # same fitted vectorizer
        scores = cosine_similarity(q_vec, mat).ravel()    # (10,)
        return sorted(zip(titles, scores), key=lambda x: x[1], reverse=True)

    expected = {
        "renewable energy and climate policy": "Climate Change",
        "password security and phishing": "Cybersecurity",
        "healthy sleep exercise and diet": "Public Health",
    }
    for q, exp in expected.items():
        ranking = rank_documents(q, my_vec, my_matrix, titles)
        print(f"\nQuery: {q!r}")
        for t, s in ranking[:3]:
            print(f"  {t:22s} {s:.3f}")
        print(f"  -> top = {ranking[0][0]} (expected {exp})")


    

if __name__ == "__main__":
    main()
