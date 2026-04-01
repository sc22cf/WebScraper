import math
from dataclasses import dataclass
from src.indexer import InvertedIndex

# Weight applied to the proximity component when combining with TF-IDF.
# final_score = tfidf_score + (PROXIMITY_WEIGHT * proximity_score)
PROXIMITY_WEIGHT: float = 2.0


@dataclass
class SearchResult:
    """Container for a single ranked search result.

    Attributes:
        url:          The document/page identifier.
        tfidf_score:  Sum of TF(t,d) × IDF(t) for all query terms.
        proximity_score: Positional closeness score (0 for single-term queries).
        final_score:  tfidf_score + PROXIMITY_WEIGHT × proximity_score.
    """

    url: str
    tfidf_score: float
    proximity_score: float
    final_score: float


class SearchEngine:
    """Query layer over an InvertedIndex.

    Supports advanced query processing that combines **TF-IDF** relevance
    with **proximity-based scoring** for multi-word queries.

    Scoring formula
    ---------------
    ``final_score = tfidf_score + (PROXIMITY_WEIGHT × proximity_score)``

    * **TF-IDF** measures how important each query term is to a document.
    * **Proximity** measures how close the query terms appear to each other
      inside the document.  Exact adjacency yields the maximum proximity
      score; large gaps yield a score approaching zero.
    * **PROXIMITY_WEIGHT** (module-level constant, default ``2.0``) controls
      how much proximity influences the final ranking relative to TF-IDF.
    """

    def __init__(self, index: InvertedIndex):
        self.index = index

    # ------------------------------------------------------------------
    # print <word>
    # ------------------------------------------------------------------

    def print_word(self, word: str) -> str:
        """Return a formatted string of the posting data for a single word.

        Returns a human-readable message if the word is empty or not found.

        Complexity: **O(D)** where *D* is the number of documents containing
        the word — one iteration over the posting list to build the output.
        """
        word = word.strip()
        if not word:
            return "No word provided."

        entry = self.index.get_entry(word)
        if entry is None:
            return f"'{word}' not found in the index."

        lines: list[str] = [f"Index entry for '{word.lower()}':"]
        for url, stats in entry.items():
            freq = stats["frequency"]
            positions = stats["positions"]
            lines.append(f"  {url} — frequency: {freq}, positions: {positions}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Query parsing
    # ------------------------------------------------------------------

    @staticmethod
    def parse_query(query: str) -> list[str]:
        """Tokenize a raw query string into lowercase alphabetic tokens.

        Delegates to ``InvertedIndex.tokenize`` to stay consistent with
        the indexing pipeline.

        Complexity: **O(N)** where *N* is the length of *query*.
        """
        return InvertedIndex.tokenize(query)

    # ------------------------------------------------------------------
    # Candidate retrieval
    # ------------------------------------------------------------------

    def _retrieve_candidates(
        self, tokens: list[str]
    ) -> tuple[list[dict[str, dict]], list[float], set[str]] | None:
        """Retrieve posting lists and IDF values for *tokens*.

        Returns ``(posting_lists, idfs, candidate_urls)`` where
        *candidate_urls* is the AND-intersection of all tokens' document
        sets.  Returns ``None`` if any token is absent from the index or
        the intersection is empty.

        Complexity: **O(K · min(D_i))** for *K* tokens.
        """
        n = self.index.page_count
        if n == 0:
            return None

        posting_lists: list[dict[str, dict]] = []
        idfs: list[float] = []
        result_set: set[str] | None = None

        for token in tokens:
            entry = self.index.get_entry(token)
            if entry is None:
                return None
            posting_lists.append(entry)
            df = len(entry)
            idfs.append(math.log(n / df))
            docs = set(entry.keys())
            result_set = docs if result_set is None else result_set & docs

        if not result_set:
            return None

        return posting_lists, idfs, result_set

    # ------------------------------------------------------------------
    # TF-IDF scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_tfidf(
        url: str,
        posting_lists: list[dict[str, dict]],
        idfs: list[float],
    ) -> float:
        """Compute the TF-IDF score for *url* across all query terms.

        ``score = Σ TF(t, d) × IDF(t)``

        Complexity: **O(K)** for *K* query terms.
        """
        score = 0.0
        for postings, idf in zip(posting_lists, idfs):
            tf = postings[url]["frequency"]
            score += tf * idf
        return score

    # ------------------------------------------------------------------
    # Proximity scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_proximity(
        url: str,
        posting_lists: list[dict[str, dict]],
    ) -> float:
        """Compute a proximity score for *url* given multiple query terms.

        For every consecutive pair of query terms (i, i+1), the method
        finds the minimum absolute distance between any position of term
        *i* and any position of term *i+1* in the document.  The pair
        score is ``1 / min_distance`` — adjacency (distance = 1) scores
        1.0; distance = 2 scores 0.5; and so on.

        The final proximity score is the **average** of all pair scores,
        keeping the value in a consistent [0, 1] range regardless of the
        number of query terms.

        Returns 0.0 when fewer than two posting lists are supplied
        (single-term queries).

        Complexity: **O(K · P₁ · P₂)** in the worst case where *P₁* and
        *P₂* are the lengths of the position lists for consecutive terms.
        """
        if len(posting_lists) < 2:
            return 0.0

        pair_scores: list[float] = []

        for i in range(len(posting_lists) - 1):
            positions_a = posting_lists[i][url]["positions"]
            positions_b = posting_lists[i + 1][url]["positions"]

            # Find minimum absolute distance between any two positions
            min_dist = float("inf")
            for pa in positions_a:
                for pb in positions_b:
                    dist = abs(pa - pb)
                    if dist < min_dist:
                        min_dist = dist

            # Avoid division by zero (same position for different terms
            # cannot happen in practice, but guard defensively).
            if min_dist == 0:
                min_dist = 1

            pair_scores.append(1.0 / min_dist)

        return sum(pair_scores) / len(pair_scores)

    # ------------------------------------------------------------------
    # Final ranking
    # ------------------------------------------------------------------

    @staticmethod
    def _rank_results(results: list[SearchResult]) -> list[SearchResult]:
        """Sort results by descending *final_score*, alphabetically on ties.

        Complexity: **O(R log R)**.
        """
        return sorted(results, key=lambda r: (-r.final_score, r.url))

    # ------------------------------------------------------------------
    # find <query>  —  public entry point
    # ------------------------------------------------------------------

    def find(self, query: str) -> list[SearchResult]:
        """Return ranked ``SearchResult`` objects for *query*.

        Processing pipeline:
        1. **Parse** — tokenize the raw query.
        2. **Retrieve** — collect posting lists and AND-intersect documents.
        3. **Score (TF-IDF)** — ``Σ TF(t,d) × IDF(t)`` per document.
        4. **Score (proximity)** — average ``1 / min_distance`` over
           consecutive term pairs (0 for single-term queries).
        5. **Combine** — ``final = tfidf + PROXIMITY_WEIGHT × proximity``.
        6. **Rank** — sort by descending *final_score*, then URL.

        Complexity
        ----------
        Let *K* = query tokens, *D_i* = docs for token *i*, *R* = results.

        * Posting lookups: **O(K)**.
        * Set intersections: **O(K · min(D_i))**.
        * TF-IDF scoring: **O(R · K)**.
        * Proximity scoring: **O(R · K · P²)** worst case.
        * Final sort: **O(R log R)**.
        """
        tokens = self.parse_query(query)
        if not tokens:
            return []

        candidates = self._retrieve_candidates(tokens)
        if candidates is None:
            return []

        posting_lists, idfs, result_set = candidates

        results: list[SearchResult] = []
        for url in result_set:
            tfidf = self._compute_tfidf(url, posting_lists, idfs)
            proximity = self._compute_proximity(url, posting_lists)
            final = tfidf + PROXIMITY_WEIGHT * proximity
            results.append(SearchResult(
                url=url,
                tfidf_score=tfidf,
                proximity_score=proximity,
                final_score=final,
            ))

        return self._rank_results(results)
