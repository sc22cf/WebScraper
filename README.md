# Search Engine Tool

A command-line search engine that crawls [https://quotes.toscrape.com/](https://quotes.toscrape.com/), builds an inverted index of every word on every page, and lets users search for single or multi-word queries. Built for the COMP3011 Web Services coursework at the University of Leeds.

## Project Overview

The tool has four main components:

| Component | File | Purpose |
|-----------|------|---------|
| **Crawler** | `src/crawler.py` | Breadth-first crawl of the target site, respecting a 6-second politeness window between requests. Detects and skips duplicate content via MD5 hashing. |
| **Indexer** | `src/indexer.py` | Builds an inverted index mapping every word to the pages it appears on, storing **frequency** and **token positions** for each word-page pair. |
| **Search** | `src/search.py` | Query layer providing `print` (single-word lookup) and `find` (multi-word AND search with TF-IDF + proximity ranking). |
| **Suggestions** | `src/suggestions.py` | Query suggestion engine providing spelling correction and vocabulary-based recommendations using corpus frequency. |
| **CLI Shell** | `src/main.py` | Interactive command-line interface exposing `build`, `load`, `print`, and `find` commands. |

## Installation & Setup

### Prerequisites

- Python 3.10+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/WebScraper.git
cd WebScraper

# 2. (Recommended) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `requests` | Composing HTTP requests to the target website |
| `beautifulsoup4` | Parsing HTML pages and extracting text / links |

These are listed in `requirements.txt` and installed in step 3 above.

## Usage

Start the interactive shell:

```bash
python -m src.main
```

### Commands

#### `build` — Crawl and index the website

Visits every reachable page on the target site, builds the inverted index, and saves it to `data/index.json`. Observes a **6-second politeness window** between requests.

```
> build
Crawling website and building index...
Crawling: https://quotes.toscrape.com/
Crawling: https://quotes.toscrape.com/login
...
Index built and saved to data/index.json.
```

#### `load` — Load a previously built index

```
> load
Index loaded from data/index.json.
```

#### `print <word>` — View the inverted index entry for a word

Shows every page containing the word along with its frequency and the token positions where it appears.

```
> print good
Index entry for 'good':
  https://quotes.toscrape.com/ — frequency: 2, positions: [54, 128]
  https://quotes.toscrape.com/page/2/ — frequency: 1, positions: [31]
```

#### `find <query>` — Search for pages containing all query words

Returns every page that contains **all** of the given words (AND semantics), ranked by a combined **TF-IDF + proximity** score.  For each result the output shows all three scoring components so the ranking is transparent.

```
> find good friends
Found in 2 page(s) (ranked by relevance):
  page: https://quotes.toscrape.com/
    tfidf_score: 1.8420
    proximity_score: 1.0000
    final_weighted_score: 3.8420
  page: https://quotes.toscrape.com/page/2/
    tfidf_score: 0.6931
    proximity_score: 0.2500
    final_weighted_score: 1.1931

> find nonexistentword
No pages found containing all terms: ['nonexistentword']
```

When no results are found and a query term is misspelled or absent from the index, the engine suggests alternatives:

```
> find fosh
No pages found containing all terms: ['fosh']
Instead of 'fosh', did you mean:
  * fish
  * food
  * foot
```

## Advanced Query Processing

The search engine combines two complementary relevance signals to rank results for multi-word queries:

### TF-IDF (Term Frequency–Inverse Document Frequency)

The base relevance score.  For each query term *t* in a document *d*:

* **TF(t, d)** — how many times *t* appears in *d*.
* **IDF(t)** — `log(N / df)` where *N* is the total number of indexed pages and *df* is the number of pages containing *t*.  Rare terms receive a higher weight.
* **tfidf\_score(d)** = Σ TF(t, d) × IDF(t) for all query terms.

### Proximity scoring

For queries with two or more terms, the engine evaluates how close the query words appear in each document using the **positional postings** already stored in the inverted index:

1. For every consecutive pair of query terms, find the **minimum absolute distance** between any of their recorded positions in the document.
2. Convert each pair distance to a score: `1 / min_distance`.  Adjacency (distance = 1) scores 1.0; distance = 4 scores 0.25, etc.
3. The document's **proximity\_score** is the **average** of all pair scores, keeping the value in a consistent [0, 1] range.

Single-word queries always receive a proximity score of 0.

### Combined ranking

The final score used for sorting is:

```
final_score = tfidf_score + PROXIMITY_WEIGHT × proximity_score
```

`PROXIMITY_WEIGHT` is a configurable constant (default **2.0**) defined at the top of `src/search.py`.  Increasing it makes positional closeness more important relative to term frequency.

### Why this improves multi-word ranking

Plain TF-IDF treats each query term independently — a page where "good" and "friends" appear on opposite ends scores the same as one where they sit side by side.  Adding proximity scoring means **documents with near-adjacent query terms naturally rank higher**, producing more intuitive results for phrase-like queries without requiring the user to use explicit phrase syntax.

## Query Suggestions

When a search returns no results — typically because of a typo or an unknown word — the engine automatically suggests alternative queries.

### How suggestions are generated

The suggestion engine uses two strategies, both derived from the **vocabulary of indexed terms**:

1. **Spelling correction** — finds vocabulary words within a maximum edit distance (default 3) of the query token using [Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance).
2. **Corpus frequency** — more common words in the index are promoted over rare ones.

### How ranking works

Each candidate suggestion is scored using a weighted formula:

```
suggestion_score = (ALPHA × similarity_score) + (BETA × term_frequency_score)
```

| Weight | Default | Purpose |
|--------|---------|---------|
| `ALPHA` | 0.7 | Favours closer spelling matches |
| `BETA` | 0.3 | Favours more frequent (and therefore likely more useful) terms |

* **similarity_score** — `1 - (edit_distance / max_len)`, normalised to [0, 1].
* **term_frequency_score** — total corpus frequency of the candidate divided by the maximum frequency of any term, normalised to [0, 1].

Suggestions are sorted by descending `suggestion_score` and capped at 5 results by default.

### Limitations

* Uses simple Levenshtein distance — no phonetic matching (e.g., Soundex) or context-aware NLP corrections.
* Vocabulary-only: suggestions are limited to words that appear in the crawled corpus.
* Scans the full vocabulary linearly for each unknown token (O(V × L)); sufficient for moderate-sized corpora but not optimised for very large indexes.
* Only suggests corrections for the first unrecognised token in a multi-word query.

## Testing

The test suite uses **pytest** with `unittest.mock` to mock all HTTP calls (no network required).

```bash
# Run the full suite with verbose output
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ --cov=src --cov-report=term-missing
```

### Test files

| File | What it covers |
|------|---------------|
| `tests/test_crawler.py` | RateLimiter, URL normalisation, BFS traversal, error handling, content deduplication |
| `tests/test_indexer.py` | Tokenisation, frequency/position tracking, case insensitivity, edge cases |
| `tests/test_storage.py` | JSON save/load round-trip, missing files, directory creation |
| `tests/test_search.py` | `print_word`, `find` (single/multi-word, empty, unknown), CLI integration |
| `tests/test_suggestions.py` | Levenshtein distance, spelling correction, suggestion ranking, CLI "Did you mean" output |

## Project Structure

```
WebScraper/
├── src/
│   ├── __init__.py
│   ├── crawler.py          # BFS crawler with politeness & deduplication
│   ├── indexer.py           # Inverted index (tokenise, index, persist)
│   ├── search.py            # Query logic (print_word, find)
│   ├── suggestions.py       # Query suggestion engine
│   └── main.py              # CLI shell
├── tests/
│   ├── __init__.py
│   ├── test_crawler.py
│   ├── test_indexer.py
│   ├── test_storage.py
│   ├── test_search.py
│   └── test_suggestions.py
├── data/
│   └── index.json           # Compiled inverted index (generated by build)
├── .github/
│   └── workflows/
│       └── ci.yml           # GitHub Actions CI pipeline
├── conftest.py              # pytest sys.path configuration
├── requirements.txt
└── README.md
```
