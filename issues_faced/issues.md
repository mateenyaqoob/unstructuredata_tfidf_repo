# Issues Faced

1. DataFrame conflict: The task asks for a pandas DataFrame in Step 3, but the instruction was to use NLP tools and no DataFrames. I used the sparse TF-IDF matrix with NumPy instead.
2. Sparse matrix is hard to inspect: Printing it gives raw `(row, col) value` entries, so I wrote a small helper to print a labeled 3 x 10 sample and the top-5 terms per document. Many terms had tied scores, so the top-5 order is somewhat arbitrary (e.g. Basketball Strategy).
3. Optional challenge hurts results: With `ngram_range=(1, 2)`, `max_df=0.8`, `min_df=2`, the vocabulary shrinks from 537 to 69 terms, only 3 bigrams survive, and the "password security and phishing" and "healthy sleep exercise and diet" queries fail. With only 10 documents, `min_df=2` removes the topic-specific words.
4. Zero-score queries: When no query word is in the vocabulary, all scores are 0 and the "top" document is just an arbitrary tie.
5. Missing packages: `pytest` was not installed (fixed with `pip install pytest`). `nltk` was not installed and was not needed.
6. Harmless warnings: PowerShell showed an error for pip's upgrade notice, and Jupyter printed zmq/TCP warnings on Windows. Neither affected results.
7. Repo problems: The provided "notebook" was a `.py` script, so I created an `.ipynb`. The helpers in `src/` still use pandas, and `task_to_do.txt` contains stray line numbers (`10|`, `20|`).
