# Search Engine Tool

A command-line search engine that crawls [https://quotes.toscrape.com/](https://quotes.toscrape.com/), builds an inverted index of every word on every page, and lets users search for single or multi-word queries. Built for the COMP3011 Web Services coursework at the University of Leeds.

## Project Overview

The tool has four main components:

| Component | File | Purpose |
|-----------|------|---------|
| **Crawler** | `src/crawler.py` | Breadth-first crawl of the target site, respecting a 6-second politeness window between requests. Detects and skips duplicate content via MD5 hashing. |
| **Indexer** | `src/indexer.py` | Builds an inverted index mapping every word to the pages it appears on, storing **frequency** and **token positions** for each word-page pair. |
| **Search** | `src/search.py` | Query layer providing `print` (single-word lookup), `find` (multi-word AND search with TF-IDF + proximity ranking), and spelling suggestions. |
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

## Performance Benchmarking

The file `tests/test_benchmark.py` contains a dedicated benchmarking module that measures the execution time of every core operation.  All benchmarks generate deterministic synthetic corpora (seeded random data) and run each operation **10 times**, reporting the average, minimum, and maximum elapsed time in milliseconds.

### Running the benchmarks

```bash
# -s ensures the structured timing output is printed to the terminal
python -m pytest tests/test_benchmark.py -v -s
```

### What is measured

| Benchmark | Operation | Expected complexity |
|-----------|-----------|---------------------|
| **Index save** | `InvertedIndex.save_to_file()` | O(n) — serialise full index to JSON |
| **Index load** | `InvertedIndex.load_from_file()` | O(n) — parse JSON file |
| **Single-word find** | `SearchEngine.find("word")` | O(D log D) — TF-IDF scoring + sort (no proximity for single words) |
| **Multi-word find** | `SearchEngine.find("alpha bravo")` | O(R × P₁ × P₂) — TF-IDF + proximity + sort |
| **No-results find** | `SearchEngine.find("unknown")` | ≈ O(1) — early exit on missing term |
| **Suggestion (query)** | `SearchEngine.suggest()` | O(V × L²) — full vocabulary Levenshtein scan; L fixed ⇒ O(V) |

Where: K = query terms, D / D_i = docs containing the term(s), R = result count, P = position list length, V = vocabulary size, L = query token length.

### Complexity-oriented graphs

Each benchmark operation produces a **log-log graph** saved to the `benchmarks/` directory.  Most operations are measured across six corpus sizes (100, 250, 500, 1 000, 2 500, 5 000 documents) to reveal how runtime scales.  The **multi-word find benchmark** uses a dedicated proximity corpus where both query terms (`"alpha"` and `"bravo"`) are injected into every document with growing repetition count, so position-list lengths P scale with corpus size and the O(P₁ × P₂) proximity cost becomes the dominant factor.  The **suggestion benchmark** uses a different independent variable — **vocabulary size V** — with six target sizes (500, 1 000, 2 000, 4 000, 8 000, 16 000 unique words) while keeping the query token fixed (`"fosh"`, length 4) so that L is constant and only V drives the scaling.

The graphs plot measured runtime against corpus size with **reference curves** (O(1), O(n), O(n log n), O(n²)) anchored to the first data point.  On a log-log scale power laws become straight lines — slope ≈ 0 means O(1), slope ≈ 1 means O(n), slope ≈ 2 means O(n²).  Matching the measured (blue) curve to a reference line identifies the complexity class.

#### Individual operation graphs

| Graph | What to look for |
|-------|------------------|
| `index_save.png` | Measured curve should track the O(n) reference — saving scales linearly with the number of documents in the index. |
| `index_load.png` | Same as save — JSON parsing is proportional to file size. |
| `find_single_word.png` | Tracks between O(n) and O(n log n).  Single-word queries use **TF-IDF scoring only** (proximity scoring requires at least two terms).  The scoring pass is O(D) but the final sort adds an O(D log D) component. |
| `find_multi-word.png` | Uses a **dedicated proximity corpus** with the fixed query `"alpha bravo"`.  Both terms are injected into every document with repetition count proportional to corpus size, so position-list lengths P₁ and P₂ grow.  The O(R × P₁ × P₂) proximity cost dominates, producing super-linear growth between O(n) and O(n²). |
| `find_no_results.png` | Should hug the O(1) reference — when a query term isn't in the index, `_retrieve_candidates` exits immediately with no scoring work. |
| `suggestion_query.png` | X-axis is **vocabulary size V** (not corpus size).  With the query token held constant (`"fosh"`, L = 4), Levenshtein cost per word is fixed and total runtime scales as O(V).  The measured curve should track the O(V) reference line. |

#### Find (single word) vs Print (word) — comparison graph

The graph `benchmarks/find_vs_print_comparison.png` overlays the two operations that query the **same word** and therefore traverse the **same posting list**.  Both grow with posting-list size D, but `find` (red) is consistently above `print` (green).  The vertical gap represents the extra work `find` performs.

**Why `find` is slower than `print` for the same word:**

Both operations start with the same O(1) dictionary lookup to retrieve the posting list.  After that, they diverge:

| Step | `print_word` | `find` |
|------|-------------|--------|
| Retrieve posting list | O(1) lookup | O(1) lookup |
| Per-document work | Format one string per entry — O(D) | Compute TF × IDF per doc — O(D), then build a `SearchResult` dataclass for each doc |
| Final step | Join strings — O(D) | Sort all results by score — **O(D log D)** |

The `find` operation does everything `print_word` does and more: it computes IDF (requiring `page_count`), multiplies TF × IDF for every matching document, allocates a `SearchResult` object per result, and then sorts the entire result list.  The O(D log D) sort is the key factor that makes `find` measurably slower — and the ratio grows with corpus size because the log D factor increases.  In practice, `find` is roughly **3–4× slower** than `print_word` on the same posting list.

### Scaling evaluation

Each benchmark runs at three corpus sizes to show how performance changes as the index grows:

| Label | Documents | Approximate scale |
|-------|-----------|-------------------|
| small | 50 | ~10 % |
| medium | 250 | ~50 % |
| large | 500 | ~100 % |

A final **scaling summary** test prints a side-by-side comparison table of all operations across the three sizes, making it straightforward to identify bottlenecks.

### Output format

Every benchmark prints structured output in the following format:

```
Benchmark: find_single_word (medium)
  corpus_size: 250
  iterations: 10
  avg_time_ms: 0.1234
  min_time_ms: 0.0987
  max_time_ms: 0.1567
  query: example
  results_count: 42
```

### Interpretation and trade-offs

* **Index load/save** scales with file size.  For the target site (~100 pages) load times are negligible; for larger corpora, switching to a binary format (e.g. `pickle`) would improve I/O speed.
* **Proximity scoring** is the most expensive per-document operation because it compares all position pairs.  Sorting position lists and using a merge-scan would reduce this from O(P₁ × P₂) to O(P₁ + P₂).
* **Suggestion generation** scales linearly with vocabulary size because it computes Levenshtein distance (O(L²) per word) against every word.  With a fixed-length query token, total cost is O(V).  A BK-tree or trie would make this sub-linear for large vocabularies.

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
| `tests/test_search.py` | `print_word`, `find` (single/multi-word, empty, unknown), proximity scoring, suggestions, CLI integration |
| `tests/test_benchmark.py` | Performance benchmarks for indexing, querying, TF-IDF, proximity, suggestions, and scaling |

## Project Structure

```
WebScraper/
├── src/
│   ├── __init__.py
│   ├── crawler.py          
│   ├── indexer.py          
│   ├── search.py           
│   └── main.py
├── tests/
│   ├── __init__.py
│   ├── test_crawler.py
│   ├── test_indexer.py
│   ├── test_storage.py
│   ├── test_search.py
│   └── test_benchmark.py
├── data/
│   └── index.json
├── .github/
│   └── workflows/
│       └── ci.yml
├── conftest.py
├── requirements.txt
└── README.md
```
