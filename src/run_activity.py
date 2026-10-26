"""End-to-end walkthrough of the TF-IDF 30-minute class activity."""


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


if __name__ == "__main__":
    main()