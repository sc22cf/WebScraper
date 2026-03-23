import pytest
from unittest.mock import patch
from io import StringIO

from src.indexer import InvertedIndex
from src.search import SearchEngine


# ---------------------------------------------------------------------------
# Helper: build a small index for reuse across tests
# ---------------------------------------------------------------------------

def _small_index() -> InvertedIndex:
    """Two-document index for deterministic testing."""
    idx = InvertedIndex()
    idx.add_document("http://a.com", "good friends are good")
    idx.add_document("http://b.com", "old friends stay friends")
    return idx


# ---------------------------------------------------------------------------
# print_word
# ---------------------------------------------------------------------------

class TestPrintWord:
    def test_existing_word(self):
        engine = SearchEngine(_small_index())
        output = engine.print_word("good")
        assert "good" in output
        assert "http://a.com" in output

    def test_unknown_word(self):
        engine = SearchEngine(_small_index())
        output = engine.print_word("banana")
        assert "not found" in output

    def test_case_insensitive_lookup(self):
        engine = SearchEngine(_small_index())
        output = engine.print_word("GOOD")
        assert "http://a.com" in output

    def test_empty_word(self):
        engine = SearchEngine(_small_index())
        output = engine.print_word("")
        assert "No word provided" in output

    def test_whitespace_only_word(self):
        engine = SearchEngine(_small_index())
        output = engine.print_word("   ")
        assert "No word provided" in output

    def test_shows_frequency(self):
        engine = SearchEngine(_small_index())
        output = engine.print_word("good")
        assert "frequency: 2" in output

    def test_shows_positions(self):
        engine = SearchEngine(_small_index())
        output = engine.print_word("good")
        assert "[0, 3]" in output


# ---------------------------------------------------------------------------
# find
# ---------------------------------------------------------------------------

class TestFind:
    def test_single_word_found(self):
        engine = SearchEngine(_small_index())
        results = engine.find("good")
        assert results == ["http://a.com"]

    def test_single_word_in_both_docs(self):
        engine = SearchEngine(_small_index())
        results = engine.find("friends")
        assert sorted(results) == ["http://a.com", "http://b.com"]

    def test_multi_word_intersection(self):
        engine = SearchEngine(_small_index())
        # "good" only in a.com, "friends" in both → intersection is a.com
        results = engine.find("good friends")
        assert results == ["http://a.com"]

    def test_unknown_word_returns_empty(self):
        engine = SearchEngine(_small_index())
        assert engine.find("banana") == []

    def test_one_known_one_unknown(self):
        engine = SearchEngine(_small_index())
        assert engine.find("good banana") == []

    def test_empty_query(self):
        engine = SearchEngine(_small_index())
        assert engine.find("") == []

    def test_whitespace_only_query(self):
        engine = SearchEngine(_small_index())
        assert engine.find("     ") == []

    def test_case_insensitive_query(self):
        engine = SearchEngine(_small_index())
        results = engine.find("GOOD FRIENDS")
        assert results == ["http://a.com"]

    def test_mixed_case_query(self):
        engine = SearchEngine(_small_index())
        results = engine.find("GoOd FrIeNdS")
        assert results == ["http://a.com"]

    def test_repeated_spaces_in_query(self):
        engine = SearchEngine(_small_index())
        results = engine.find("good    friends")
        assert results == ["http://a.com"]

    def test_punctuation_in_query(self):
        engine = SearchEngine(_small_index())
        results = engine.find("good, friends!")
        assert results == ["http://a.com"]

    def test_empty_index(self):
        engine = SearchEngine(InvertedIndex())
        assert engine.find("anything") == []


# ---------------------------------------------------------------------------
# Integration-style: CLI round-trip via main.py shell
# ---------------------------------------------------------------------------

class TestCLIIntegration:
    """Simulate user input through run_shell to verify wiring."""

    def _run_commands(self, commands: list[str]) -> str:
        """Feed commands (including 'quit') to run_shell and capture stdout."""
        from src.main import run_shell

        user_input = "\n".join(commands)
        with patch("sys.stdin", StringIO(user_input)), \
             patch("sys.stdout", new_callable=StringIO) as mock_out:
            run_shell()
        return mock_out.getvalue()

    def test_print_without_load(self):
        output = self._run_commands(["print hello", "quit"])
        assert "No index loaded" in output

    def test_find_without_load(self):
        output = self._run_commands(["find hello", "quit"])
        assert "No index loaded" in output

    def test_load_missing_index(self, tmp_path):
        import src.main as main_mod
        original_path = main_mod.INDEX_PATH
        main_mod.INDEX_PATH = str(tmp_path / "nonexistent.json")
        try:
            output = self._run_commands(["load", "quit"])
        finally:
            main_mod.INDEX_PATH = original_path
        assert "not found" in output.lower() or "error" in output.lower()

    def test_unknown_command(self):
        output = self._run_commands(["foo", "quit"])
        assert "Unknown command" in output

    def test_help_command(self):
        output = self._run_commands(["help", "quit"])
        assert "build" in output
        assert "load" in output
        assert "print" in output
        assert "find" in output

    def test_load_and_search(self, tmp_path):
        """Build a small index to disk, then load + search via the shell."""
        import src.main as main_mod

        idx = _small_index()
        index_file = tmp_path / "index.json"
        idx.save_to_file(str(index_file))

        # Temporarily override the index path used by the shell
        original_path = main_mod.INDEX_PATH
        main_mod.INDEX_PATH = str(index_file)
        try:
            output = self._run_commands([
                "load",
                "print good",
                "find good friends",
                "find banana",
                "quit",
            ])
        finally:
            main_mod.INDEX_PATH = original_path

        assert "Index loaded" in output
        assert "http://a.com" in output
        assert "frequency: 2" in output
        # "find banana" should produce no results
        assert "No pages found" in output
