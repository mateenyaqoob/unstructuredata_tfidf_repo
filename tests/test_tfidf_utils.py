"""Basic unit tests for the TF-IDF activity utilities."""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.load_data import load_documents
from src.tfidf_utils import build_tfidf, rank_documents_for_query


def test_load_documents():
    titles, docs = load_documents()
    assert len(titles) == 10
    assert len(docs) == 10
    assert all(len(d.split()) > 30 for d in docs)  # "long text"


def test_tfidf_matrix_shape():
    _, docs = load_documents()
    vectorizer, matrix = build_tfidf(docs)
    assert matrix.shape[0] == 10
    assert matrix.shape[1] == len(vectorizer.get_feature_names_out())


def test_query_ranking_matches_expected_topic():
    titles, docs = load_documents()
    vectorizer, matrix = build_tfidf(docs)
    ranking = rank_documents_for_query(
        "renewable energy and climate policy", vectorizer, matrix, titles
    )
    assert ranking[0][0] == "Climate Change"


def test_query_ranking_cybersecurity():
    titles, docs = load_documents()
    vectorizer, matrix = build_tfidf(docs)
    ranking = rank_documents_for_query(
        "password security and phishing", vectorizer, matrix, titles
    )
    assert ranking[0][0] == "Cybersecurity"


def test_query_ranking_public_health():
    titles, docs = load_documents()
    vectorizer, matrix = build_tfidf(docs)
    ranking = rank_documents_for_query(
        "healthy sleep exercise and diet", vectorizer, matrix, titles
    )
    assert ranking[0][0] == "Public Health"


def test_bigram_variant_shrinks_vocabulary():
    _, docs = load_documents()
    v1, m1 = build_tfidf(docs)
    v2, m2 = build_tfidf(docs, ngram_range=(1, 2), max_df=0.8, min_df=2)
    assert m2.shape[0] == 10
    assert m2.shape[1] < m1.shape[1]  # min_df=2 drops single-document terms
    assert any(" " in term for term in v2.get_feature_names_out())
