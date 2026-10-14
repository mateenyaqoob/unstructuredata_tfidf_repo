# TF-IDF Class Activity

A hands-on Python activity for teaching TF-IDF and cosine similarity using ten
long-text documents.

## Learning objectives

- Explain term frequency, inverse document frequency, and TF-IDF.
- Build and inspect a TF-IDF matrix with scikit-learn.
- Interpret the highest-weighted terms per document.
- Rank documents against a query using cosine similarity.
- Inspect NLTK stopword removal and query-term weights.

## Repository contents

| Path | Purpose |
|---|---|
| `data/documents.py` | Ten documents and their titles |
| `src/load_data.py` | Loads the documents |
| `src/tfidf_utils.py` | TF-IDF and similarity utilities |
| `src/run_activity.py` | End-to-end analysis and detailed output |
| `tests/` | Unit tests |
| `ISSUES.md` | Findings from the supplied workspace and their status |
| `TFIDF_RESULTS.md` | Verified output for the requested query |

## Setup

```powershell
git clone https://github.com/mateenyaqoob/unstructuredata_tfidf_repo.git
cd unstructuredata_tfidf_repo
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m nltk.downloader stopwords
```

## Run the activity

```powershell
python -m src.run_activity
```

The script verifies the document count and lengths, prints NLTK stopword and token
counts, reports the TF-IDF matrix and top terms, ranks the query results, and
shows the top document's query-term frequencies, IDF weights, and TF-IDF scores.
See [TFIDF_RESULTS.md](./TFIDF_RESULTS.md) for the verified result.

## Run tests

```powershell
python -m pytest -q
```
