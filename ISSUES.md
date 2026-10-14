# Identified Issues

This file records findings from the supplied local workspace. The instructor's
upstream repository already had the corrected utility filename and valid import
syntax; the first two findings applied only to the supplied workspace snapshot.

| Issue | Impact | Status |
|---|---|---|
| In the supplied workspace, the TF-IDF helper was named `tfidf_utilis.py`, while activity code and tests imported `src.tfidf_utils`. | Pytest could not collect tests from that snapshot. | Fixed locally by using the correctly named `src/tfidf_utils.py`; upstream already used the correct name. |
| In the supplied workspace, `run_activity.py` had an invalid expression in its import list. | The activity could not be parsed. | Fixed locally with valid imports; upstream already had valid imports. |
| NLTK's `word_tokenize` required `punkt_tab`, which was not installed. | Calling that tokenizer raised an NLTK `LookupError`. | The activity uses NLTK's `TreebankWordTokenizer`, which avoids this additional tokenizer-data requirement. |
| NLTK was missing from the supplied workspace's dependency list. | A fresh install would not include the package needed for stopword analysis. | Added `nltk>=3.8.0` to the project requirements. The README documents downloading NLTK's English stopwords corpus. |
| Pytest was not installed in the active environment at the first test run. | `python -m pytest -q` could not start. | Installed in the active environment; this was an environment setup issue, not an application defect. |

## Verification

- `pytest -q`: 3 tests passed.
- The activity loaded exactly 10 documents, all with more than 30 words.
- The TF-IDF matrix had shape `(10, 537)`.
- "Climate Change" ranked first for the requested query.
- The detailed query output is recorded in [TFIDF_RESULTS.md](./TFIDF_RESULTS.md).
