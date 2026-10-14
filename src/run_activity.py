"""End-to-end walkthrough of the TF-IDF 30-minute class activity."""

from collections import Counter

from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer

from src.load_data import load_documents
from src.tfidf_utils import (
    build_tfidf,
    rank_documents_for_query,
    to_dataframe,
    top_terms_per_document,
)


def main() -> None:
    titles, documents = load_documents()
    if len(titles) != 10 or len(documents) != 10:
        raise AssertionError(
            f"Expected exactly 10 titled documents; loaded {len(titles)} titles "
            f"and {len(documents)} documents."
        )
    print(f"Loaded {len(documents)} documents.")

    word_counts = [len(document.split()) for document in documents]
    for title, word_count in zip(titles, word_counts):
        print(f"  {title}: {word_count} words")
    if not all(word_count > 30 for word_count in word_counts):
        raise AssertionError("Every document must contain more than 30 words.")
    print("PASS: all 10 documents contain more than 30 words.")

    tokenizer = TreebankWordTokenizer()
    stopword_set = set(stopwords.words("english"))
    print("\n=== NLTK stopword removal ===")
    tokenized_documents = [
        [token.lower() for token in tokenizer.tokenize(document)]
        for document in documents
    ]
    for title, tokens in zip(titles, tokenized_documents):
        removed = sorted(set(tokens) & stopword_set)
        remaining_count = sum(token not in stopword_set for token in tokens)
        print(f"\n{title}:")
        print(f"  Unique stopwords removed: {removed}")
        print(
            f"  Original tokens: {len(tokens)}; "
            f"remaining non-stopword tokens: {remaining_count}"
        )

    vectorizer, matrix = build_tfidf(documents, stop_words="english")
    vocabulary = vectorizer.get_feature_names_out()
    print("\n=== TF-IDF matrix ===")
    print(f"Matrix shape: {matrix.shape}")
    print(f"Total vocabulary size: {len(vocabulary)}")

    dataframe = to_dataframe(matrix, vectorizer, titles)
    print("\n=== Top 5 TF-IDF terms per document ===")
    for title, terms in top_terms_per_document(dataframe, n=5).items():
        joined = ", ".join(f"{term} ({score:.4f})" for term, score in terms.items())
        print(f"{title}: {joined}")

    query = "renewable energy and climate policy"
    ranking = rank_documents_for_query(query, vectorizer, matrix, titles)
    print(f"\n=== Similarity ranking for {query!r} ===")
    for rank, (title, score) in enumerate(ranking, start=1):
        print(f"{rank:2}. {title:25s} {score:.4f}")
    if not ranking or ranking[0][0] != "Climate Change":
        raise AssertionError('"Climate Change" did not rank first for the query.')
    print('PASS: "Climate Change" ranks first.')

    top_document_index = titles.index(ranking[0][0])
    top_document_tokens = Counter(tokenized_documents[top_document_index])
    feature_indices = {term: index for index, term in enumerate(vocabulary)}
    query_terms = dict.fromkeys(
        token.lower()
        for token in tokenizer.tokenize(query)
        if token.isalpha()
    )
    print(f"\n=== Query-term breakdown for {ranking[0][0]} ===")
    for term in query_terms:
        frequency = top_document_tokens[term]
        if term in feature_indices:
            feature_index = feature_indices[term]
            idf = vectorizer.idf_[feature_index]
            tfidf_score = matrix[top_document_index, feature_index]
            print(
                f"{term}: frequency={frequency}, idf={idf:.6f}, "
                f"tf-idf={tfidf_score:.6f}"
            )
        else:
            print(
                f"{term}: frequency={frequency}, idf=not in vocabulary, "
                "tf-idf=not in vocabulary (removed by vectorizer stop words)"
            )


if __name__ == "__main__":
    main()