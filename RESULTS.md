# TF-IDF Activity — Results

**Student:** Syed Shahbaz Ali Shah — 24i-8002

Full code and outputs: [`notebooks/tfidf_activity.ipynb`](notebooks/tfidf_activity.ipynb)
(script version: `notebooks/tfidf_activity.py`).

## 1. Load the data
10 documents loaded; every document is long text (83–91 words).

| Document | Words | Document | Words |
|---|---|---|---|
| AI in Healthcare | 85 | Cybersecurity | 88 |
| Climate Change | 83 | Music Theory | 87 |
| Basketball Strategy | 85 | Gardening Vegetables | 89 |
| Italian Pasta | 91 | Economics Inflation | 89 |
| Space Exploration | 86 | Public Health | 91 |

## 2. TF-IDF matrix
`TfidfVectorizer(stop_words="english")` → matrix shape **(10, 537)**, vocabulary size **537**.

## 3. Top-5 terms per document

| Document | Top-5 TF-IDF terms |
|---|---|
| AI in Healthcare | ai, healthcare, use, treatments, faster |
| Climate Change | change, emissions, energy, climate, scale |
| Basketball Strategy | create, spacing, stars, strategy, pace |
| Italian Pasta | pasta, sauces, sauce, different, timer |
| Space Exploration | space, mars, missions, developing, geology |
| Cybersecurity | cybersecurity, systems, data, authentication, firewalls |
| Music Theory | music, musicians, practice, edit, intervals |
| Gardening Vegetables | soil, year, pests, improves, mulch |
| Economics Inflation | spending, inflation, trade, economic, firms |
| Public Health | health, sleep, public, outcomes, chronic |

The top terms work as a summary for most documents. Basketball Strategy is the weak
case: all its top terms tie at 0.13 because no word repeats, so the order is arbitrary.

## 4. Query similarity

| Query | 1st (score) | 2nd (score) | Expected |
|---|---|---|---|
| renewable energy and climate policy | Climate Change (0.323) | Economics Inflation (0.095) | ✅ |
| password security and phishing | Cybersecurity (0.176) | — all others 0 | ✅ |
| healthy sleep exercise and diet | Public Health (0.285) | Gardening Vegetables (0.045) | ✅ |

Economics Inflation ranks second for the climate query because both documents use
"policy". Gardening ranks second for the health query through the shared word "healthy".

## 5. Optional challenge — bigrams with `max_df=0.8, min_df=2`

| | Default | Bigrams + max_df/min_df |
|---|---|---|
| Vocabulary size | 537 | **69** |
| Bigrams kept | 0 | 3 (`economic growth`, `policy involves`, `trade offs`) |

**Do bigrams make the terms more meaningful? Not on this corpus.**
`min_df=2` removes every term that appears in only one document. Since each document
covers a different topic, that removes the most distinctive words (`pasta`, `mars`,
`cybersecurity`, `soil`, `inflation`), and nearly all bigrams with them.

- Top terms become generic, e.g. Italian Pasta → *different, long, improve, flavor, quality*.
- "password security and phishing" scores **0 for every document**.
- "healthy sleep exercise and diet" ranks **Gardening Vegetables first** because
  `sleep`, `exercise` and `diet` were pruned.

Bigrams on their own (`ngram_range=(1, 2)` without `min_df`) raise the vocabulary to
1,182 terms and do add useful phrases. The `min_df`/`max_df` pruning is designed for large
corpora; with 10 documents it removes signal instead of noise.

## 6. Tests
`pytest -q` → **6 passed**. The 3 provided tests pass, plus 3 added: the Cybersecurity
and Public Health queries rank correctly, and the bigram variant shrinks the vocabulary
while keeping at least one bigram.
