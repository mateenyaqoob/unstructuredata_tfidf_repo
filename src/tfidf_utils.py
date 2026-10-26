"""Reusable TF-IDF helpers for the class activity."""
from typing import List, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_tfidf(
    documents: List[str],
    stop_words: str = "english",
    ngram_range: Tuple[int, int] = (1, 1),
    max_df: float = 1.0,
    min_df: int = 1,
):
    """Fit a TfidfVectorizer and return (vectorizer, matrix)."""
    vectorizer = TfidfVectorizer(
        stop_words=stop_words,
        ngram_range=ngram_range,
        max_df=max_df,
        min_df=min_df,
    )
    matrix = vectorizer.fit_transform(documents)
    return vectorizer, matrix


def to_dataframe(matrix, vectorizer, titles: List[str]) -> pd.DataFrame:
    """Convert a sparse TF-IDF matrix into a labeled DataFrame."""
    return pd.DataFrame(
        matrix.toarray(),
        columns=vectorizer.get_feature_names_out(),
        index=titles,
    )


def top_terms_per_document(df: pd.DataFrame, n: int = 5) -> dict:
    """Return {title: {term: score}} of the top-n TF-IDF terms per document."""
    result = {}
    for title in df.index:
        top = df.loc[title].sort_values(ascending=False).head(n)
        result[title] = top.to_dict()
    return result


def rank_documents_for_query(
    query: str,
    vectorizer: TfidfVectorizer,
    matrix,
    titles: List[str],
) -> List[Tuple[str, float]]:
    """Return [(title, similarity_score)] sorted by cosine similarity (desc)."""
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix).flatten()
    return sorted(zip(titles, scores), key=lambda x: x[1], reverse=True)