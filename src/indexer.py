import re


class InvertedIndex:
    """
    Inverted index mapping each word to its occurrences across documents.

    Structure:
        index = {
            "word": {
                "url1": {"frequency": int, "positions": [int, ...]},
                "url2": {"frequency": int, "positions": [int, ...]},
            },
            ...
        }
    """

    def __init__(self):
        self.index: dict[str, dict[str, dict]] = {}

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Split text into lowercase, alphabetic tokens."""
        return re.findall(r"[a-zA-Z]+", text.lower())

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_document(self, url: str, text: str) -> None:
        """Tokenize *text* and record each word's frequency and positions for *url*."""
        tokens = self.tokenize(text)

        for position, word in enumerate(tokens):
            if word not in self.index:
                self.index[word] = {}

            if url not in self.index[word]:
                self.index[word][url] = {"frequency": 0, "positions": []}

            self.index[word][url]["frequency"] += 1
            self.index[word][url]["positions"].append(position)

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def get_entry(self, word: str) -> dict | None:
        """Return the posting list for *word*, or None if absent."""
        return self.index.get(word.lower())

    def get_documents(self, word: str) -> list[str]:
        """Return list of URLs containing *word*."""
        entry = self.get_entry(word)
        return list(entry.keys()) if entry else []
