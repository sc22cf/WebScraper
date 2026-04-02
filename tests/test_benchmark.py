"""Performance benchmarking tests for the search engine.

Measures execution time of core operations across varying corpus sizes
and query types.  Each benchmark runs 10 iterations and reports
average / minimum / maximum time in milliseconds.

Run benchmarks:
    python -m pytest -m benchmark -v -s

The ``-s`` flag is important — it lets the structured ``Benchmark:``
output reach the terminal.
"""

import math
import random
import string
import time
from pathlib import Path

import pytest

from src.indexer import InvertedIndex
from src.search import SearchEngine, SuggestionEngine

# Mark every test in this module as a benchmark
pytestmark = pytest.mark.benchmark

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ITERATIONS = 10  # Number of repetitions per benchmark for statistical accuracy

# Corpus sizes used for scaling evaluation
CORPUS_SIZES = {
    "small":  50,    # ~10 % of full
    "medium": 250,   # ~50 % of full
    "large":  500,   # ~100 % (full synthetic corpus)
}

# Fixed random seed for reproducibility
_RNG = random.Random(42)

# Fixed misspelled token for suggestion benchmarks (constant length L = 4)
_FIXED_MISSPELLING = "fosh"

# Fixed query terms for multi-word proximity benchmark
_MULTI_WORD_TERMS = ("alpha", "bravo")
_MULTI_WORD_QUERY = "alpha bravo"

# Pre-built word pool so every run generates the same synthetic data
_WORD_POOL: list[str] = []


def _build_word_pool(size: int = 2000) -> list[str]:
    """Generate a deterministic pool of pseudo-random words (length 3–10)."""
    global _WORD_POOL
    if _WORD_POOL:
        return _WORD_POOL
    rng = random.Random(42)
    pool: list[str] = []
    for _ in range(size):
        length = rng.randint(3, 10)
        word = "".join(rng.choices(string.ascii_lowercase, k=length))
        pool.append(word)
    _WORD_POOL = pool
    return pool


# ---------------------------------------------------------------------------
# Synthetic corpus generation
# ---------------------------------------------------------------------------

def _build_corpus(num_docs: int) -> InvertedIndex:
    """Build a synthetic ``InvertedIndex`` with *num_docs* documents.

    Each document contains 30–80 words drawn from a fixed pool, ensuring
    realistic overlap between documents (some words appear in many pages,
    others in few).

    Complexity of building the corpus itself: **O(num_docs × T)** where
    *T* is the average number of tokens per document.
    """
    pool = _build_word_pool()
    rng = random.Random(42)
    idx = InvertedIndex()
    for i in range(num_docs):
        doc_len = rng.randint(30, 80)
        words = [rng.choice(pool) for _ in range(doc_len)]
        text = " ".join(words)
        idx.add_document(f"http://example.com/page/{i}", text)
    return idx


def _pick_existing_word(idx: InvertedIndex) -> str:
    """Return the highest-document-frequency word in *idx*.

    Picking the most common word ensures the posting list (and therefore
    the scoring + sorting work) grows proportionally with corpus size.
    """
    return max(idx.index, key=lambda w: len(idx.index[w]))


def _pick_existing_pair(idx: InvertedIndex) -> str:
    """Return a two-word query where both words are **common** in the index.

    Picks the two highest-document-frequency words that share at least one
    document.  This ensures the AND-intersection grows proportionally with
    corpus size, so the benchmark actually measures scaling behaviour
    rather than a near-constant tiny result set.
    """
    # Sort vocabulary by document frequency (descending).
    ranked = sorted(idx.index.keys(), key=lambda w: len(idx.index[w]), reverse=True)

    for i in range(len(ranked)):
        for j in range(i + 1, min(i + 20, len(ranked))):
            docs_a = set(idx.index[ranked[i]].keys())
            docs_b = set(idx.index[ranked[j]].keys())
            if docs_a & docs_b:
                return f"{ranked[i]} {ranked[j]}"
    # fallback: just return two top words
    return f"{ranked[0]} {ranked[1]}"


def _pick_misspelled_word(idx: InvertedIndex) -> str:
    """Return a word NOT in the index but close to one that is."""
    existing = _pick_existing_word(idx)
    # swap first character to create a plausible misspelling
    c = "z" if existing[0] != "z" else "a"
    return c + existing[1:]


def _build_vocab_corpus(target_vocab: int) -> InvertedIndex:
    """Build a corpus whose vocabulary size ≈ *target_vocab*.

    A dedicated word pool of *target_vocab* unique words (length 3–10) is
    created, and enough documents are generated to ensure most words are
    indexed at least once.  The word ``fosh`` is excluded from the pool
    so it can be used as a fixed misspelled token.
    """
    rng = random.Random(42)
    pool: list[str] = []
    seen: set[str] = set()
    while len(pool) < target_vocab:
        length = rng.randint(3, 10)
        word = "".join(rng.choices(string.ascii_lowercase, k=length))
        if word not in seen and word != _FIXED_MISSPELLING:
            pool.append(word)
            seen.add(word)

    # Enough documents so most words appear at least once.
    # avg ~55 words/doc → need ≈ target_vocab * 3 / 55 docs for coverage.
    num_docs = max(200, target_vocab * 3 // 55)
    idx = InvertedIndex()
    for i in range(num_docs):
        doc_len = rng.randint(30, 80)
        words = [rng.choice(pool) for _ in range(doc_len)]
        text = " ".join(words)
        idx.add_document(f"http://example.com/vocab/{i}", text)
    return idx


def _build_proximity_corpus(num_docs: int) -> InvertedIndex:
    """Build a corpus designed to stress proximity scoring.

    Every document contains both ``alpha`` and ``bravo`` injected
    multiple times, with repetition count proportional to *num_docs*.
    This ensures:

    * **All documents match** the two-word query → intersection size = num_docs.
    * **Position-list lengths P grow** with corpus size, so the O(P₁ × P₂)
      proximity computation becomes the dominant cost.
    * Filler words add realistic noise so position lists are not trivially
      adjacent.

    Repetitions per term per document: ``max(2, num_docs // 100)``.
    Document length: ``base_filler + 2 × reps``.
    """
    rng = random.Random(42)
    pool = _build_word_pool()  # filler words
    # Exclude the query terms from filler to keep positions controlled
    filler = [w for w in pool if w not in _MULTI_WORD_TERMS]

    reps = max(2, num_docs // 100)
    base_filler = 40
    idx = InvertedIndex()

    for i in range(num_docs):
        words: list[str] = [rng.choice(filler) for _ in range(base_filler)]
        # Inject query terms at random positions throughout the document
        for _ in range(reps):
            pos = rng.randint(0, len(words))
            words.insert(pos, "alpha")
            pos = rng.randint(0, len(words))
            words.insert(pos, "bravo")
        text = " ".join(words)
        idx.add_document(f"http://example.com/prox/{i}", text)
    return idx


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

class _BenchmarkResult:
    """Collects timing samples and prints structured output."""

    def __init__(self, name: str, corpus_size: int):
        self.name = name
        self.corpus_size = corpus_size
        self.times_ms: list[float] = []
        self.extra: dict[str, object] = {}

    def record(self, elapsed_s: float) -> None:
        self.times_ms.append(elapsed_s * 1000)

    def report(self) -> None:
        avg = sum(self.times_ms) / len(self.times_ms)
        lo = min(self.times_ms)
        hi = max(self.times_ms)
        parts = [
            f"Benchmark: {self.name}",
            f"  corpus_size: {self.corpus_size}",
            f"  iterations: {len(self.times_ms)}",
            f"  avg_time_ms: {avg:.4f}",
            f"  min_time_ms: {lo:.4f}",
            f"  max_time_ms: {hi:.4f}",
        ]
        for k, v in self.extra.items():
            parts.append(f"  {k}: {v}")
        print("\n".join(parts))


def _bench(func, iterations: int = ITERATIONS) -> list[float]:
    """Run *func* for *iterations* rounds and return elapsed times in seconds."""
    times: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append(end - start)
    return times


# ===================================================================
# 1. Index loading / saving benchmark
# ===================================================================

class TestIndexLoadBenchmark:
    """Benchmark ``InvertedIndex.save_to_file`` and ``load_from_file``.

    Complexity
    ----------
    * **save_to_file**: O(V × D_avg) — serialises the full index to JSON,
      iterating over every word (V) and its posting list.
    * **load_from_file**: O(S) — ``json.load`` parses the file in a single
      pass proportional to the file size *S* in bytes.
    """

    @pytest.mark.parametrize("label,num_docs", list(CORPUS_SIZES.items()))
    def test_save_load_time(self, label, num_docs, tmp_path):
        idx = _build_corpus(num_docs)
        path = str(tmp_path / "bench_index.json")

        # ---- save ----
        save_bench = _BenchmarkResult(f"index_save ({label})", num_docs)
        for t in _bench(lambda: idx.save_to_file(path)):
            save_bench.record(t)
        save_bench.extra["file_bytes"] = Path(path).stat().st_size
        save_bench.report()

        # ---- load ----
        load_bench = _BenchmarkResult(f"index_load ({label})", num_docs)
        for t in _bench(lambda: InvertedIndex.load_from_file(path)):
            load_bench.record(t)
        load_bench.extra["file_bytes"] = Path(path).stat().st_size
        load_bench.report()


# ===================================================================
# 2. Query execution benchmark (end-to-end find)
# ===================================================================

class TestQueryExecutionBenchmark:
    """Benchmark end-to-end ``SearchEngine.find()`` for different query types.

    Complexity
    ----------
    * **Single-word query**: O(D) for TF-IDF where D = docs containing
      the word, plus O(D log D) for sorting.
    * **Multi-word query**: O(K × min(D_i)) for intersection, then
      O(R × K × P²) for proximity scoring (P = position-list length),
      then O(R log R) for sorting.
    * **No-results query**: O(K) — ``_retrieve_candidates`` returns early
      when any term is absent.
    """

    @pytest.mark.parametrize("label,num_docs", list(CORPUS_SIZES.items()))
    def test_single_word_query(self, label, num_docs):
        idx = _build_corpus(num_docs)
        engine = SearchEngine(idx)
        word = _pick_existing_word(idx)

        bench = _BenchmarkResult(f"find_single_word ({label})", num_docs)
        result_count = 0
        for t in _bench(lambda: engine.find(word)):
            bench.record(t)
        result_count = len(engine.find(word))
        bench.extra["query"] = word
        bench.extra["results_count"] = result_count
        bench.report()

    @pytest.mark.parametrize("label,num_docs", list(CORPUS_SIZES.items()))
    def test_multi_word_query(self, label, num_docs):
        idx = _build_proximity_corpus(num_docs)
        engine = SearchEngine(idx)

        bench = _BenchmarkResult(f"find_multi_word ({label})", num_docs)
        for t in _bench(lambda: engine.find(_MULTI_WORD_QUERY)):
            bench.record(t)
        result_count = len(engine.find(_MULTI_WORD_QUERY))
        bench.extra["query"] = _MULTI_WORD_QUERY
        bench.extra["results_count"] = result_count
        bench.report()

    @pytest.mark.parametrize("label,num_docs", list(CORPUS_SIZES.items()))
    def test_no_results_query(self, label, num_docs):
        idx = _build_corpus(num_docs)
        engine = SearchEngine(idx)
        query = "xyznonexistent"

        bench = _BenchmarkResult(f"find_no_results ({label})", num_docs)
        for t in _bench(lambda: engine.find(query)):
            bench.record(t)
        bench.extra["query"] = query
        bench.extra["results_count"] = 0
        bench.report()


# ===================================================================
# 3. TF-IDF computation benchmark (isolated)
# ===================================================================

class TestTFIDFBenchmark:
    """Benchmark the isolated ``_compute_tfidf`` static method.

    Complexity
    ----------
    O(K) per document — one multiplication per query term.  The benchmark
    measures the cost across all candidate documents to show how it scales
    with the number of results.
    """

    @pytest.mark.parametrize("label,num_docs", list(CORPUS_SIZES.items()))
    def test_tfidf_scoring(self, label, num_docs):
        idx = _build_corpus(num_docs)
        engine = SearchEngine(idx)
        word = _pick_existing_word(idx)
        tokens = engine.parse_query(word)
        candidates = engine._retrieve_candidates(tokens)
        assert candidates is not None
        posting_lists, idfs, result_set = candidates

        def score_all():
            for url in result_set:
                engine._compute_tfidf(url, posting_lists, idfs)

        bench = _BenchmarkResult(f"tfidf_scoring ({label})", num_docs)
        for t in _bench(score_all):
            bench.record(t)
        bench.extra["candidate_docs"] = len(result_set)
        bench.report()


# ===================================================================
# 4. Proximity scoring benchmark (isolated)
# ===================================================================

class TestProximityBenchmark:
    """Benchmark the isolated ``_compute_proximity`` static method.

    Complexity
    ----------
    O(K × P₁ × P₂) per document, where P₁ and P₂ are the lengths of
    position lists for consecutive query term pairs.  For very large
    position lists this dominates overall query time.
    """

    @pytest.mark.parametrize("label,num_docs", list(CORPUS_SIZES.items()))
    def test_proximity_scoring(self, label, num_docs):
        idx = _build_corpus(num_docs)
        engine = SearchEngine(idx)
        query = _pick_existing_pair(idx)
        tokens = engine.parse_query(query)
        candidates = engine._retrieve_candidates(tokens)
        if candidates is None:
            pytest.skip("no overlapping docs for chosen pair")
        posting_lists, _, result_set = candidates

        def score_all():
            for url in result_set:
                engine._compute_proximity(url, posting_lists)

        bench = _BenchmarkResult(f"proximity_scoring ({label})", num_docs)
        for t in _bench(score_all):
            bench.record(t)
        bench.extra["candidate_docs"] = len(result_set)
        bench.extra["query"] = query
        bench.report()


# ===================================================================
# 5. Print (posting list lookup) benchmark
# ===================================================================

class TestPrintBenchmark:
    """Benchmark ``SearchEngine.print_word``.

    Complexity
    ----------
    O(D) where D = number of documents containing the word.  The method
    performs a single O(1) hash-table lookup followed by one pass over
    the posting list to build the formatted output string.
    """

    @pytest.mark.parametrize("label,num_docs", list(CORPUS_SIZES.items()))
    def test_print_word(self, label, num_docs):
        idx = _build_corpus(num_docs)
        engine = SearchEngine(idx)
        word = _pick_existing_word(idx)

        bench = _BenchmarkResult(f"print_word ({label})", num_docs)
        for t in _bench(lambda: engine.print_word(word)):
            bench.record(t)
        entry = idx.get_entry(word)
        bench.extra["query"] = word
        bench.extra["posting_list_size"] = len(entry) if entry else 0
        bench.report()


# ===================================================================
# 6. Suggestion engine benchmark
# ===================================================================

# Vocabulary sizes for the parametrised suggestion benchmark (3 sizes)
_SUGGESTION_SIZES = {
    "small":  500,
    "medium": 2000,
    "large":  8000,
}


class TestSuggestionBenchmark:
    """Benchmark ``SuggestionEngine.suggest`` and ``SearchEngine.suggest``.

    Complexity
    ----------
    * **suggest (single token)**: O(V × L²) where V = vocabulary size and
      L = token length.  With a fixed token (``fosh``, L = 4) the
      Levenshtein cost per word is constant, so runtime scales as O(V).
    * **suggest_for_query**: Same worst-case — stops after the first
      unrecognised token.

    The independent variable is **vocabulary size** (controlled via
    word-pool size), not corpus size.
    """

    @pytest.mark.parametrize("label,target_vocab", list(_SUGGESTION_SIZES.items()))
    def test_single_token_suggestion(self, label, target_vocab):
        idx = _build_vocab_corpus(target_vocab)
        se = SuggestionEngine(idx)

        bench = _BenchmarkResult(f"suggest_single ({label})", target_vocab)
        for t in _bench(lambda: se.suggest(_FIXED_MISSPELLING)):
            bench.record(t)
        bench.extra["token"] = _FIXED_MISSPELLING
        bench.extra["vocab_size"] = len(idx.index)
        bench.report()

    @pytest.mark.parametrize("label,target_vocab", list(_SUGGESTION_SIZES.items()))
    def test_query_suggestion(self, label, target_vocab):
        idx = _build_vocab_corpus(target_vocab)
        engine = SearchEngine(idx)

        bench = _BenchmarkResult(f"suggest_query ({label})", target_vocab)
        for t in _bench(lambda: engine.suggest(_FIXED_MISSPELLING)):
            bench.record(t)
        bench.extra["query"] = _FIXED_MISSPELLING
        bench.extra["vocab_size"] = len(idx.index)
        bench.report()


# ===================================================================
# 7. Scaling comparison (all operations at three corpus sizes)
# ===================================================================

class TestScalingSummary:
    """Run a combined scaling comparison across 10 %, 50 %, 100 % of the
    full synthetic corpus and print a summary table.

    This produces a side-by-side view of how each operation's time grows
    with corpus size, making it easy to identify bottlenecks.
    """

    def test_scaling_summary(self, tmp_path):
        rows: list[str] = []

        for label, num_docs in CORPUS_SIZES.items():
            idx = _build_corpus(num_docs)
            engine = SearchEngine(idx)
            path = str(tmp_path / f"scale_{label}.json")

            # ----- save -----
            idx.save_to_file(path)
            save_times = _bench(lambda: idx.save_to_file(path))
            save_avg = sum(save_times) / len(save_times) * 1000

            # ----- load -----
            load_times = _bench(lambda: InvertedIndex.load_from_file(path))
            load_avg = sum(load_times) / len(load_times) * 1000

            # ----- find single -----
            word = _pick_existing_word(idx)
            find1_times = _bench(lambda: engine.find(word))
            find1_avg = sum(find1_times) / len(find1_times) * 1000

            # ----- find multi -----
            pair = _pick_existing_pair(idx)
            find2_times = _bench(lambda: engine.find(pair))
            find2_avg = sum(find2_times) / len(find2_times) * 1000

            # ----- suggestion -----
            misspelled = _pick_misspelled_word(idx)
            sug_times = _bench(lambda: engine.suggest(misspelled))
            sug_avg = sum(sug_times) / len(sug_times) * 1000

            rows.append(
                f"  {label:>6s} ({num_docs:>4d} docs) | "
                f"save: {save_avg:8.3f} ms | "
                f"load: {load_avg:8.3f} ms | "
                f"find1: {find1_avg:8.4f} ms | "
                f"find2: {find2_avg:8.4f} ms | "
                f"suggest: {sug_avg:8.3f} ms"
            )

        header = (
            "\nBenchmark: scaling_summary\n"
            "  Corpus Size         |   save       |   load       |  find(1w)    |  find(2w)    |  suggest\n"
            "  " + "-" * 110
        )
        print(header)
        for row in rows:
            print(row)


# ===================================================================
# 8. Complexity-oriented performance graphs
# ===================================================================
#
# Why these chart types?
# ---------------------
# Bar charts of raw milliseconds make it hard to distinguish O(n) from
# O(n log n) because the visual difference is subtle at small scales.
#
# **Log-log plots** turn power-law relationships into straight lines:
#   slope ≈ 0 → O(1),  slope ≈ 1 → O(n),  slope ≈ 2 → O(n²).
# Overlaying normalised reference curves lets the reader compare
# *shapes*, not absolute magnitudes.
#
# **Diagnostic ratio plots** (runtime / f(n)) compress the question to
# "is this line flat?"  If runtime/n is flat, the operation is O(n).
# Together the two panels give both an intuitive visual and a
# quantitative confirmation of the complexity class.

# Directory where graphs are saved (project root / benchmarks/)
_GRAPH_DIR = Path(__file__).resolve().parent.parent / "benchmarks"

# More corpus sizes than the parametrised tests — extra data points
# make scaling trends visible on log-log axes.
GRAPH_CORPUS_SIZES: list[int] = [100, 250, 500, 1000, 2500, 5000]

# Vocabulary sizes for the suggestion benchmark (V is the independent
# variable — word-pool size grows per run while L stays constant).
SUGGESTION_VOCAB_SIZES: list[int] = [500, 1000, 2000, 4000, 8000, 16000]

# Corpus sizes for the multi-word proximity benchmark.  Kept smaller than
# the main GRAPH_CORPUS_SIZES because every document is scored with O(P²)
# proximity comparisons, which would make the largest sizes impractically
# slow.
MULTI_WORD_CORPUS_SIZES: list[int] = [50, 100, 200, 400, 800, 1600]

# ---------------------------------------------------------------------------
# Expected complexities and reference-curve configuration
# ---------------------------------------------------------------------------
# "expected" — theoretical complexity (shown in title)
# "refs"     — reference curves overlaid on the log-log panel
# "ratios"   — divisors for the diagnostic ratio panel
#
# Rationale per operation:
#   index save/load   → O(S) where S ∝ index size ∝ n
#   single-word find  → O(d log d), d = posting-list length, scales with n
#   multi-word find   → O(min(di)) intersection + O(R·K·P²) proximity
#   no-results find   → O(K) early exit when a term is absent ≈ O(1)
#   print_word        → O(D) single posting-list iteration
#   TF-IDF scoring    → O(d) one pass over candidate docs
#   proximity scoring → O(d·P²) position-list pairs, may be super-linear
#   suggestion        → O(V·L²) full vocabulary scan; L fixed ⇒ O(V)
#                       (graphed separately with V on x-axis)
# ---------------------------------------------------------------------------

_OP_META: dict[str, dict] = {
    "Index Save": {
        "expected": "O(n)",
        "refs": ["O(1)", "O(n)", "O(n log n)"],
    },
    "Index Load": {
        "expected": "O(n)",
        "refs": ["O(1)", "O(n)", "O(n log n)"],
    },
    "Find (single word)": {
        "expected": "O(D log D) — TF-IDF + sort",
        "refs": ["O(1)", "O(log n)", "O(n)", "O(n log n)"],
    },
    "Find (no results)": {
        "expected": "≈ O(1)",
        "refs": ["O(1)", "O(log n)", "O(n)"],
    },
}

# Suggestion graph metadata (separate — uses V on x-axis, not corpus size)
_SUGGESTION_META = {
    "expected": "O(V × L²) — L fixed ⇒ O(V)",
    "refs": ["O(n)", "O(n log n)", "O(n²)"],
}

# Multi-word graph metadata (separate — uses dedicated proximity corpus)
_MULTI_WORD_META = {
    "expected": "O(R × P₁ × P₂) — TF-IDF + proximity + sort",
    "refs": ["O(n)", "O(n log n)", "O(n²)"],
}

# Colours and dash styles for reference curves (log-log panel)
_REF_STYLES: dict[str, dict] = {
    "O(1)":       {"color": "#999999", "ls": "--", "lw": 1.2},
    "O(log n)":   {"color": "#2ca02c", "ls": "-.", "lw": 1.2},
    "O(n)":       {"color": "#ff7f0e", "ls": "--", "lw": 1.2},
    "O(n log n)": {"color": "#d62728", "ls": "-.", "lw": 1.2},
    "O(n²)":      {"color": "#9467bd", "ls": "--", "lw": 1.2},
}

# ---------------------------------------------------------------------------
# Mathematical helpers
# ---------------------------------------------------------------------------

def _f_of_n(n: float, complexity: str) -> float:
    """Evaluate a named complexity function at *n*."""
    if complexity == "O(1)":
        return 1.0
    if complexity == "O(log n)":
        return math.log2(max(n, 2))
    if complexity == "O(n)":
        return float(n)
    if complexity == "O(n log n)":
        return n * math.log2(max(n, 2))
    if complexity == "O(n²)":
        return float(n * n)
    raise ValueError(complexity)


def _reference_curve(
    n_values: list[int],
    complexity: str,
    anchor_n: int,
    anchor_y: float,
) -> list[float]:
    """Return reference values normalised so ``f(anchor_n) ≈ anchor_y``.

    All curves start at the same point as the first measurement, making
    the shape comparison — not absolute magnitude — the focus.
    """
    anchor_f = _f_of_n(anchor_n, complexity)
    scale = anchor_y / anchor_f if anchor_f else 1.0
    return [_f_of_n(n, complexity) * scale for n in n_values]


# ---------------------------------------------------------------------------
# Data collection (extended for graphs)
# ---------------------------------------------------------------------------

def _collect_timings(
    tmp_path: Path,
    corpus_sizes: list[int] | None = None,
) -> dict[str, list[float]]:
    """Run every operation at each corpus size and return avg times (ms).

    *corpus_sizes* defaults to ``GRAPH_CORPUS_SIZES`` when called from the
    graph test, giving more data points than the three-point parametrised
    benchmarks.
    """
    sizes = corpus_sizes if corpus_sizes is not None else GRAPH_CORPUS_SIZES
    results: dict[str, list[float]] = {k: [] for k in _OP_META}
    # Print (word) is not graphed individually but needed for comparison
    results["Print (word)"] = []

    for num_docs in sizes:
        idx = _build_corpus(num_docs)
        engine = SearchEngine(idx)
        path = str(tmp_path / f"graph_{num_docs}.json")
        idx.save_to_file(path)

        def _avg(times: list[float]) -> float:
            return sum(times) / len(times) * 1000

        # save / load
        results["Index Save"].append(_avg(_bench(lambda: idx.save_to_file(path))))
        results["Index Load"].append(
            _avg(_bench(lambda: InvertedIndex.load_from_file(path)))
        )

        # find — single word (TF-IDF scoring only, no proximity)
        word = _pick_existing_word(idx)
        results["Find (single word)"].append(_avg(_bench(lambda: engine.find(word))))

        # find — no results (term absent → early exit)
        results["Find (no results)"].append(
            _avg(_bench(lambda: engine.find("xyznonexistent")))
        )

        # print — single posting-list lookup + format (for comparison graph)
        results["Print (word)"].append(
            _avg(_bench(lambda: engine.print_word(word)))
        )

    return results


def _collect_multi_word_timings(
    corpus_sizes: list[int] | None = None,
) -> tuple[list[int], list[float]]:
    """Collect multi-word find timings using the dedicated proximity corpus.

    Each corpus size gets its own ``_build_proximity_corpus``, which
    injects ``alpha`` and ``bravo`` with growing repetition count so
    position-list lengths P scale with corpus size.  The fixed query
    ``"alpha bravo"`` is used every run — no randomness in term selection.

    Returns ``(sizes, avg_times_ms)``.
    """
    sizes = corpus_sizes if corpus_sizes is not None else MULTI_WORD_CORPUS_SIZES
    times_ms: list[float] = []

    for num_docs in sizes:
        idx = _build_proximity_corpus(num_docs)
        engine = SearchEngine(idx)
        avg_ms = sum(_bench(lambda: engine.find(_MULTI_WORD_QUERY))) / ITERATIONS * 1000
        times_ms.append(avg_ms)

    return list(sizes), times_ms


def _collect_suggestion_timings(
    vocab_sizes: list[int] | None = None,
) -> tuple[list[int], list[float]]:
    """Collect suggestion timings with vocabulary size V as the independent variable.

    For each target vocabulary size a dedicated corpus is built whose
    word-pool size equals the target.  The fixed token ``fosh`` (length 4)
    is used as the misspelled query so that L is constant across all runs,
    isolating V as the only scaling factor.

    Returns ``(actual_vocab_sizes, avg_times_ms)``.
    """
    targets = vocab_sizes if vocab_sizes is not None else SUGGESTION_VOCAB_SIZES
    actual_vs: list[int] = []
    times_ms: list[float] = []

    for target_v in targets:
        idx = _build_vocab_corpus(target_v)
        engine = SearchEngine(idx)
        actual_v = len(idx.index)
        actual_vs.append(actual_v)

        avg_ms = sum(_bench(lambda: engine.suggest(_FIXED_MISSPELLING))) / ITERATIONS * 1000
        times_ms.append(avg_ms)

    return actual_vs, times_ms


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------

class TestPerformanceGraphs:
    """Generate complexity-oriented performance graphs.

    For each operation a log-log graph is produced showing measured
    runtime vs corpus size with reference curves overlaid.  On a
    log-log scale power laws become straight lines whose slope reveals
    the exponent: slope ≈ 0 → O(1), slope ≈ 1 → O(n), slope ≈ 2 → O(n²).

    A find-vs-print comparison graph and a vocabulary-scaled suggestion
    graph are also generated.

    Graphs are saved to ``benchmarks/``.
    Run with ``-s`` to see paths printed to the terminal.
    """

    def test_generate_graphs(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt

        _GRAPH_DIR.mkdir(exist_ok=True)
        sizes = GRAPH_CORPUS_SIZES
        data = _collect_timings(tmp_path, corpus_sizes=sizes)

        # ============================================================
        # Individual log-log figures (one per operation)
        # ============================================================
        for op_name, timings in data.items():
            if op_name not in _OP_META:
                continue
            meta = _OP_META[op_name]
            fig, ax = plt.subplots(figsize=(8, 5))

            # Log-log is the single best chart type for complexity:
            # straight-line slope directly encodes the polynomial exponent.
            ax.loglog(
                sizes, timings, "o-",
                color="#1f77b4", linewidth=2.2, markersize=7,
                label="Measured", zorder=5,
            )

            # Overlay reference curves, all anchored to the first point
            # so that *shape divergence* is immediately visible.
            anchor_n, anchor_y = sizes[0], timings[0]
            for ref_name in meta["refs"]:
                sty = _REF_STYLES[ref_name]
                curve = _reference_curve(sizes, ref_name, anchor_n, anchor_y)
                ax.loglog(
                    sizes, curve,
                    color=sty["color"], ls=sty["ls"], lw=sty["lw"],
                    label=ref_name,
                )

            ax.set_xlabel("Corpus size (documents)", fontsize=10)
            ax.set_ylabel("Avg time (ms)", fontsize=10)
            ax.legend(fontsize=8, loc="upper left")
            ax.grid(True, which="both", ls=":", alpha=0.4)

            # Reading guide placed below the axes
            ax.text(
                0.02, -0.12,
                "How to read: all curves share the first point.\n"
                "slope ≈ 0 → O(1)  ·  slope ≈ 1 → O(n)  ·  "
                "slope ≈ 2 → O(n²)",
                transform=ax.transAxes, fontsize=7.5,
                color="#555555", va="top",
            )

            fig.suptitle(
                f"{op_name}  —  expected {meta['expected']}",
                fontsize=13, fontweight="bold",
            )
            fig.tight_layout(rect=[0, 0.06, 1, 0.93])

            safe = (
                op_name.lower()
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
            )
            path = _GRAPH_DIR / f"{safe}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"Graph saved: {path}")

        # ============================================================
        # Comparison: Find (single word) vs Print (word)
        # ============================================================
        find_times = data["Find (single word)"]
        print_times = data["Print (word)"]

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.loglog(
            sizes, find_times, "o-",
            color="#d62728", linewidth=2.2, markersize=7,
            label="find (single word)", zorder=5,
        )
        ax.loglog(
            sizes, print_times, "s-",
            color="#2ca02c", linewidth=2.2, markersize=7,
            label="print (word)", zorder=5,
        )

        # Reference curves anchored to find (consistent with individual graph)
        anchor_n, anchor_y = sizes[0], find_times[0]
        for ref_name in ["O(1)", "O(n)", "O(n log n)"]:
            sty = _REF_STYLES[ref_name]
            curve = _reference_curve(sizes, ref_name, anchor_n, anchor_y)
            ax.loglog(
                sizes, curve,
                color=sty["color"], ls=sty["ls"], lw=sty["lw"],
                label=ref_name,
            )

        ax.set_xlabel("Corpus size (documents)", fontsize=10)
        ax.set_ylabel("Avg time (ms)", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.text(
            0.02, -0.12,
            "Both operations scale with posting-list size D.\n"
            "find includes TF-IDF scoring + O(D log D) sort → "
            "consistently slower.",
            transform=ax.transAxes, fontsize=7.5,
            color="#555555", va="top",
        )

        fig.suptitle(
            "Find (single word)  vs  Print (word)  —  same data, different work",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0.06, 1, 0.93])
        cmp_path = _GRAPH_DIR / "find_vs_print_comparison.png"
        fig.savefig(cmp_path, dpi=150)
        plt.close(fig)
        print(f"Comparison graph saved: {cmp_path}")

        # ============================================================
        # Multi-word find graph (dedicated proximity corpus)
        # ============================================================
        mw_sizes, mw_times = _collect_multi_word_timings()
        meta = _MULTI_WORD_META
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.loglog(
            mw_sizes, mw_times, "o-",
            color="#1f77b4", linewidth=2.2, markersize=7,
            label="Measured", zorder=5,
        )

        anchor_n, anchor_y = mw_sizes[0], mw_times[0]
        for ref_name in meta["refs"]:
            sty = _REF_STYLES[ref_name]
            curve = _reference_curve(mw_sizes, ref_name, anchor_n, anchor_y)
            ax.loglog(
                mw_sizes, curve,
                color=sty["color"], ls=sty["ls"], lw=sty["lw"],
                label=ref_name,
            )

        ax.set_xlabel("Corpus size (documents)", fontsize=10)
        ax.set_ylabel("Avg time (ms)", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.text(
            0.02, -0.12,
            f"Fixed query: \"{_MULTI_WORD_QUERY}\".  "
            "Both terms repeat per doc (reps ∝ corpus size).\n"
            "Position-list growth drives O(P₁ × P₂) proximity cost.",
            transform=ax.transAxes, fontsize=7.5,
            color="#555555", va="top",
        )

        fig.suptitle(
            f"Find (multi-word)  —  expected {meta['expected']}",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0.06, 1, 0.93])
        mw_path = _GRAPH_DIR / "find_multi-word.png"
        fig.savefig(mw_path, dpi=150)
        plt.close(fig)
        print(f"Graph saved: {mw_path}")

        # ============================================================
        # Suggestion graph (vocabulary size V on x-axis)
        # ============================================================
        vocab_sizes, sug_times = _collect_suggestion_timings()
        meta = _SUGGESTION_META
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.loglog(
            vocab_sizes, sug_times, "o-",
            color="#1f77b4", linewidth=2.2, markersize=7,
            label="Measured", zorder=5,
        )

        anchor_n, anchor_y = vocab_sizes[0], sug_times[0]
        for ref_name in meta["refs"]:
            sty = _REF_STYLES[ref_name]
            curve = _reference_curve(vocab_sizes, ref_name, anchor_n, anchor_y)
            ax.loglog(
                vocab_sizes, curve,
                color=sty["color"], ls=sty["ls"], lw=sty["lw"],
                label=ref_name,
            )

        ax.set_xlabel("Vocabulary size (V = unique indexed words)", fontsize=10)
        ax.set_ylabel("Avg time (ms)", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.text(
            0.02, -0.12,
            f"Fixed query token: \"{_FIXED_MISSPELLING}\" (L = {len(_FIXED_MISSPELLING)}).  "
            "Only V varies.\n"
            "slope ≈ 1 → O(V)  ·  slope ≈ 2 → O(V²)",
            transform=ax.transAxes, fontsize=7.5,
            color="#555555", va="top",
        )

        fig.suptitle(
            f"Suggestion (query)  —  expected {meta['expected']}",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0.06, 1, 0.93])
        sug_path = _GRAPH_DIR / "suggestion_query.png"
        fig.savefig(sug_path, dpi=150)
        plt.close(fig)
        print(f"Graph saved: {sug_path}")
