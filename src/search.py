from src.indexer import InvertedIndex


class SearchEngine:
    """Query layer over an InvertedIndex."""

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

    def find(self, query: str) -> list[str]:
        """Return URLs that contain *every* token in *query* (AND semantics).

        Tokenisation re-uses ``InvertedIndex.tokenize`` so behaviour is
        consistent with how documents were indexed (lowercase, alphabetic).
        Returns an empty list for empty / whitespace-only queries or if
        any token is missing from the index.

        The algorithm fetches each token's posting set from the index in
        **O(1)** and then intersects the sets. Python's ``set.intersection``
        is implemented in C and runs in **O(min(|A|, |B|))** per pair.

        Complexity
        ----------
        Let *K* = number of query tokens, *D_i* = docs for token *i*.

        * Posting lookups: **O(K)** (each is O(1)).
        * Set intersections: **O(K · min(D_i))** in the worst case.
        * Final sort: **O(R log R)** where *R* = number of result URLs.
        * Overall: **O(K · min(D_i) + R log R)**, which is very fast
          because *K* is small (number of search terms) and each
          intersection shrinks the candidate set.
        """
        tokens = InvertedIndex.tokenize(query)
        if not tokens:
            return []

        # Start with documents for the first token, then intersect
        result_set: set[str] | None = None
        for token in tokens:
            docs = set(self.index.get_documents(token))
            if not docs:
                return []
            result_set = docs if result_set is None else result_set & docs

        return sorted(result_set) if result_set else []
