import math
from src.indexer import InvertedIndex


class SearchEngine:
    """Query layer over an InvertedIndex.

    Supports TF-IDF ranked retrieval.  Term Frequency–Inverse Document
    Frequency (TF-IDF) is a standard information-retrieval weighting scheme
    that scores each document by how relevant it is to the query:

    * **TF(t, d)** — how often term *t* appears in document *d*.
    * **IDF(t)** — ``log(N / df)`` where *N* is the total number of indexed
      pages and *df* is the number of pages containing *t*.  Rare terms
      receive a higher weight.
    * **Score(d, Q)** — sum of ``TF(t, d) × IDF(t)`` for every query
      term *t* in *Q*.
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
    # find <query>
    # ------------------------------------------------------------------

    def find(self, query: str) -> list[tuple[str, float]]:
        """Return URLs containing *every* query token, ranked by TF-IDF score.

        Uses AND semantics: a document must contain all query tokens to be
        included.  Results are sorted by descending TF-IDF score so that the
        most relevant pages appear first.

        Scoring
        -------
        For each candidate document *d* and query token *t*:

        * ``TF(t, d)``  = frequency of *t* in *d* (stored in the index).
        * ``IDF(t)``     = ``log(N / df)`` where *N* = total indexed pages
          and *df* = number of pages containing *t*.
        * ``score(d)``   = Σ TF(t, d) × IDF(t)  for all *t* in the query.

        Complexity
        ----------
        Let *K* = query tokens, *D_i* = docs for token *i*, *R* = results.

        * Posting lookups: **O(K)**.
        * Set intersections: **O(K · min(D_i))**.
        * Scoring: **O(R · K)** — one TF lookup per token per result.
        * Final sort: **O(R log R)**.
        * Overall: **O(K · min(D_i) + R · K + R log R)**.
        """
        tokens = InvertedIndex.tokenize(query)
        if not tokens:
            return []

        n = self.index.page_count
        if n == 0:
            return []

        # Collect posting lists and compute IDF for each token
        posting_lists: list[dict[str, dict]] = []
        idfs: list[float] = []
        result_set: set[str] | None = None

        for token in tokens:
            entry = self.index.get_entry(token)
            if entry is None:
                return []
            posting_lists.append(entry)
            df = len(entry)
            idfs.append(math.log(n / df))
            docs = set(entry.keys())
            result_set = docs if result_set is None else result_set & docs

        if not result_set:
            return []

        # Score each candidate document
        scores: dict[str, float] = {}
        for url in result_set:
            score = 0.0
            for postings, idf in zip(posting_lists, idfs):
                tf = postings[url]["frequency"]
                score += tf * idf
            scores[url] = score

        # Sort by descending score, then alphabetically for ties
        ranked = sorted(result_set, key=lambda u: (-scores[u], u))
        return [(url, scores[url]) for url in ranked]
