import pytest
from unittest.mock import MagicMock, patch
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


def _urls(results: list[tuple[str, float]]) -> list[str]:
    """Extract just the URLs from find() results for assertion convenience."""
    return [url for url, _ in results]


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
        assert _urls(results) == ["http://a.com"]

    def test_single_word_in_both_docs(self):
        engine = SearchEngine(_small_index())
        results = engine.find("friends")
        assert sorted(_urls(results)) == ["http://a.com", "http://b.com"]

    def test_multi_word_intersection(self):
        engine = SearchEngine(_small_index())
        # "good" only in a.com, "friends" in both → intersection is a.com
        results = engine.find("good friends")
        assert _urls(results) == ["http://a.com"]

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
        assert _urls(results) == ["http://a.com"]

    def test_mixed_case_query(self):
        engine = SearchEngine(_small_index())
        results = engine.find("GoOd FrIeNdS")
        assert _urls(results) == ["http://a.com"]

    def test_repeated_spaces_in_query(self):
        engine = SearchEngine(_small_index())
        results = engine.find("good    friends")
        assert _urls(results) == ["http://a.com"]

    def test_punctuation_in_query(self):
        engine = SearchEngine(_small_index())
        results = engine.find("good, friends!")
        assert _urls(results) == ["http://a.com"]

    def test_empty_index(self):
        engine = SearchEngine(InvertedIndex())
        assert engine.find("anything") == []


# ---------------------------------------------------------------------------
# TF-IDF ranking
# ---------------------------------------------------------------------------

def _tfidf_index() -> InvertedIndex:
    """Three-document index designed so TF-IDF ordering is deterministic.

    - "good" appears 3× in a.com, 1× in b.com  (not in c.com)
    - "friends" appears in all three docs
    - "rare" appears only in c.com (high IDF)
    """
    idx = InvertedIndex()
    idx.add_document("http://a.com", "good good good friends")
    idx.add_document("http://b.com", "good friends friends")
    idx.add_document("http://c.com", "friends rare")
    return idx


class TestTFIDFRanking:
    def test_higher_tf_ranks_first(self):
        """Doc with higher term frequency should rank above one with lower."""
        engine = SearchEngine(_tfidf_index())
        results = engine.find("good")
        # a.com has tf=3, b.com has tf=1 → a.com should rank first
        assert _urls(results) == ["http://a.com", "http://b.com"]

    def test_rare_term_boosts_score(self):
        """A term in fewer documents (higher IDF) should boost a doc's rank."""
        idx = InvertedIndex()
        idx.add_document("http://a.com", "common common common")
        idx.add_document("http://b.com", "common rare")
        idx.add_document("http://c.com", "common unique")
        engine = SearchEngine(idx)
        # "common" appears in all 3 docs → IDF ≈ 0
        # searching just "common": a.com has tf=3, others tf=1
        results = engine.find("common")
        assert results[0][0] == "http://a.com"

    def test_multi_word_tfidf_scoring(self):
        """Multi-word query sums TF-IDF across all terms."""
        engine = SearchEngine(_tfidf_index())
        # "good friends": only a.com and b.com match (AND)
        # a.com: tf(good)=3, tf(friends)=1; b.com: tf(good)=1, tf(friends)=2
        # IDF(good) = log(3/2), IDF(friends) = log(3/3) = 0
        # a.com score = 3*log(3/2) + 1*0 = 3*log(1.5)
        # b.com score = 1*log(3/2) + 2*0 = 1*log(1.5)
        # a.com should rank higher
        results = engine.find("good friends")
        assert results[0][0] == "http://a.com"

    def test_single_doc_match_returns_one(self):
        """Query matching a single document returns just that document."""
        engine = SearchEngine(_tfidf_index())
        results = engine.find("rare")
        assert _urls(results) == ["http://c.com"]

    def test_alphabetical_tiebreak(self):
        """Documents with identical scores are sorted alphabetically."""
        idx = InvertedIndex()
        idx.add_document("http://z.com", "word")
        idx.add_document("http://a.com", "word")
        engine = SearchEngine(idx)
        results = engine.find("word")
        assert _urls(results) == ["http://a.com", "http://z.com"]

    def test_all_docs_same_tf_idf_zero(self):
        """When every doc contains the term equally, IDF = 0, so scores are 0.
        Results fall back to alphabetical order."""
        idx = InvertedIndex()
        idx.add_document("http://b.com", "hello")
        idx.add_document("http://a.com", "hello")
        engine = SearchEngine(idx)
        results = engine.find("hello")
        assert _urls(results) == ["http://a.com", "http://b.com"]


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

    def test_eofError_exits_gracefully(self):
        """Exhausting stdin (EOFError) should print 'Exiting.' and return."""
        output = self._run_commands([])
        assert "Exiting" in output

    def test_keyboard_interrupt_exits_gracefully(self):
        """KeyboardInterrupt on input should print 'Exiting.' and return."""
        from src.main import run_shell

        with patch("builtins.input", side_effect=KeyboardInterrupt), \
             patch("sys.stdout", new_callable=StringIO) as mock_out:
            run_shell()

        assert "Exiting" in mock_out.getvalue()

    def test_empty_input_is_skipped(self):
        """An empty line should be skipped without error."""
        output = self._run_commands(["", "quit"])
        assert "Exiting" in output

    def test_build_command(self):
        """The build command should crawl, save the index, and confirm."""
        from src.main import run_shell

        with patch("src.main.Crawler") as mock_crawler_cls, \
             patch("src.main.RateLimiter"), \
             patch("src.main.InvertedIndex") as mock_idx_cls, \
             patch("src.main.SearchEngine"), \
             patch("sys.stdin", StringIO("build\nquit\n")), \
             patch("sys.stdout", new_callable=StringIO) as mock_out:
            mock_idx = MagicMock()
            mock_idx_cls.return_value = mock_idx
            mock_crawler = MagicMock()
            mock_crawler_cls.return_value = mock_crawler
            run_shell()

        output = mock_out.getvalue()
        assert "Index built" in output
        mock_crawler.crawl_and_index.assert_called_once_with(mock_idx)
        mock_idx.save_to_file.assert_called_once()

    def test_find_no_terms_provided(self, tmp_path):
        """'find' with an empty argument prints 'No search terms provided.'"""
        import src.main as main_mod

        idx = _small_index()
        index_file = tmp_path / "index.json"
        idx.save_to_file(str(index_file))

        original_path = main_mod.INDEX_PATH
        main_mod.INDEX_PATH = str(index_file)
        try:
            output = self._run_commands(["load", "find", "quit"])
        finally:
            main_mod.INDEX_PATH = original_path

        assert "No search terms provided" in output

    def test_exit_alias(self):
        """The 'exit' command should exit the shell just like 'quit'."""
        output = self._run_commands(["exit"])
        assert "Exiting" in output

    def test_main_entry_point(self):
        """Running the module as __main__ invokes run_shell()."""
        import sys
        import runpy

        # Remove cached entry so runpy doesn't emit a RuntimeWarning about
        # 'src.main' already being in sys.modules before execution.
        sys.modules.pop("src.main", None)

        # Patch input to immediately raise EOFError so run_shell() exits cleanly
        with patch("builtins.input", side_effect=EOFError), \
             patch("sys.stdout", new_callable=StringIO):
            runpy.run_module("src.main", run_name="__main__", alter_sys=True)
