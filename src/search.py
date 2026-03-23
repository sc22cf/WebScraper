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
        """Return URLs that contain *every* token in *query*.

        Tokenisation re-uses InvertedIndex.tokenize so behaviour is
        consistent with how documents were indexed (lowercase, alphabetic).
        Returns an empty list for empty / whitespace-only queries or if
        any token is missing from the index.
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
