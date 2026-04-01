import json
import re
from pathlib import Path


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
        """Split text into lowercase, alphabetic-only tokens.

        Uses a single-pass regex (``re.findall``) which scans the input
        string once from left to right.

        Complexity: **O(N)** where *N* is the length of *text* in characters.
        """
        return re.findall(r"[a-zA-Z]+", text.lower())

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_document(self, url: str, text: str) -> None:
        """Tokenize *text* and record each word's frequency and positions for *url*.

        For every token the method performs a dictionary lookup and insert,
        both of which are **O(1)** amortised for Python dicts.

        Complexity: **O(T)** where *T* is the number of tokens in *text*.
        """
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

    @property
    def page_count(self) -> int:
        """Return the number of unique URLs indexed."""
        urls: set[str] = set()
        for postings in self.index.values():
            urls.update(postings.keys())
        return len(urls)

    def get_entry(self, word: str) -> dict | None:
        """Return the posting list for *word*, or ``None`` if absent.

        Complexity: **O(1)** amortised — single hash-table lookup.
        """
        return self.index.get(word.lower())

    def get_documents(self, word: str) -> list[str]:
        """Return list of URLs containing *word*.

        Complexity: **O(D)** where *D* is the number of documents that
        contain the word (i.e. the length of the posting list).
        """
        entry = self.get_entry(word)
        return list(entry.keys()) if entry else []

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_to_file(self, path: str) -> None:
        """Serialise the index to a JSON file at *path*."""
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.index, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_from_file(cls, path: str) -> "InvertedIndex":
        """Deserialise an index from a JSON file at *path*.

        Raises FileNotFoundError if the file does not exist.
        """
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Index file not found: {path}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        instance = cls()
        instance.index = data
        return instance
