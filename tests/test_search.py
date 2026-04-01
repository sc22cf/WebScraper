import math

import pytest
from unittest.mock import MagicMock, patch
from io import StringIO

from src.indexer import InvertedIndex
from src.search import (
    ALPHA,
    BETA,
    MAX_EDIT_DISTANCE,
    PROXIMITY_WEIGHT,
    SearchEngine,
    SearchResult,
    Suggestion,
    SuggestionEngine,
    levenshtein,
)


# ---------------------------------------------------------------------------
# Helper: build a small index for reuse across tests
# ---------------------------------------------------------------------------

def _small_index() -> InvertedIndex:
    """Two-document index for deterministic testing."""
    idx = InvertedIndex()
    idx.add_document("http://a.com", "good friends are good")
    idx.add_document("http://b.com", "old friends stay friends")
    return idx


def _urls(results: list[SearchResult]) -> list[str]:
    """Extract just the URLs from find() results for assertion convenience."""
    return [r.url for r in results]


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
        assert results[0].url == "http://a.com"

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
        assert results[0].url == "http://a.com"

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


# ---------------------------------------------------------------------------
# Proximity scoring
# ---------------------------------------------------------------------------

def _proximity_index() -> InvertedIndex:
    """Three-document index with varying term proximity.

    - a.com: "good friends" appear adjacent (positions 0, 1)
    - b.com: "good ... friends" are 4 positions apart (0, 4)
    - c.com: "good ... ... ... ... ... friends" are 6 apart (0, 6)
    """
    idx = InvertedIndex()
    idx.add_document("http://a.com", "good friends are here today")
    idx.add_document("http://b.com", "good are here today friends")
    idx.add_document("http://c.com", "good are here today and also friends")
    return idx


class TestProximityScoring:
    def test_adjacent_terms_highest_proximity(self):
        """Adjacent terms should yield the highest proximity score (1.0)."""
        engine = SearchEngine(_proximity_index())
        results = engine.find("good friends")
        scores = {r.url: r.proximity_score for r in results}
        # a.com has adjacent terms → proximity = 1/1 = 1.0
        assert scores["http://a.com"] == pytest.approx(1.0)

    def test_distant_terms_lower_proximity(self):
        """Terms farther apart should yield lower proximity scores."""
        engine = SearchEngine(_proximity_index())
        results = engine.find("good friends")
        scores = {r.url: r.proximity_score for r in results}
        # b.com: distance 4 → 1/4 = 0.25
        # c.com: distance 6 → 1/6 ≈ 0.1667
        assert scores["http://b.com"] > scores["http://c.com"]

    def test_proximity_increases_when_words_closer(self):
        """Proximity score should strictly increase as words get closer."""
        engine = SearchEngine(_proximity_index())
        results = engine.find("good friends")
        scores = {r.url: r.proximity_score for r in results}
        assert scores["http://a.com"] > scores["http://b.com"] > scores["http://c.com"]

    def test_single_word_proximity_is_zero(self):
        """Single-word queries should have proximity_score = 0."""
        engine = SearchEngine(_proximity_index())
        results = engine.find("good")
        for r in results:
            assert r.proximity_score == 0.0

    def test_multi_word_ranking_prefers_closer_matches(self):
        """Documents with closer term matches should rank above distant ones
        when TF-IDF scores are similar."""
        engine = SearchEngine(_proximity_index())
        results = engine.find("good friends")
        urls = [r.url for r in results]
        # a.com (adjacent) should rank first
        assert urls[0] == "http://a.com"

    def test_result_contains_all_score_fields(self):
        """Every SearchResult should expose tfidf_score, proximity_score,
        and final_score."""
        engine = SearchEngine(_proximity_index())
        results = engine.find("good friends")
        for r in results:
            assert isinstance(r.tfidf_score, float)
            assert isinstance(r.proximity_score, float)
            assert isinstance(r.final_score, float)

    def test_final_score_formula(self):
        """final_score must equal tfidf_score + PROXIMITY_WEIGHT * proximity_score."""
        engine = SearchEngine(_proximity_index())
        results = engine.find("good friends")
        for r in results:
            expected = r.tfidf_score + PROXIMITY_WEIGHT * r.proximity_score
            assert r.final_score == pytest.approx(expected)

    def test_three_term_proximity_averages_pairs(self):
        """With three query terms, proximity should be the average of
        the two consecutive-pair scores."""
        idx = InvertedIndex()
        # positions: alpha=0, beta=1, gamma=2  → pairs: (0,1)=1, (1,2)=1
        idx.add_document("http://x.com", "alpha beta gamma")
        engine = SearchEngine(idx)
        results = engine.find("alpha beta gamma")
        # both pairs are adjacent → each pair score = 1.0, average = 1.0
        assert results[0].proximity_score == pytest.approx(1.0)

    def test_three_term_proximity_with_gap(self):
        """Three terms where one pair is distant should lower the average."""
        idx = InvertedIndex()
        # positions: alpha=0, beta=1, filler=2,3,4, gamma=5
        idx.add_document("http://x.com", "alpha beta filler filler filler gamma")
        engine = SearchEngine(idx)
        results = engine.find("alpha beta gamma")
        # pair (alpha, beta): distance=1 → 1.0
        # pair (beta, gamma): distance=4 → 0.25
        # average = (1.0 + 0.25) / 2 = 0.625
        assert results[0].proximity_score == pytest.approx(0.625)

    def test_tfidf_still_correct_with_proximity(self):
        """TF-IDF component should be unchanged by proximity addition."""
        idx = InvertedIndex()
        idx.add_document("http://a.com", "good good good friends")
        idx.add_document("http://b.com", "good friends friends")
        idx.add_document("http://c.com", "friends rare")
        engine = SearchEngine(idx)

        results = engine.find("good")
        scores = {r.url: r.tfidf_score for r in results}
        # IDF(good) = log(3/2)
        idf = math.log(3 / 2)
        assert scores["http://a.com"] == pytest.approx(3 * idf)
        assert scores["http://b.com"] == pytest.approx(1 * idf)

    def test_cli_output_shows_all_scores(self, tmp_path):
        """The find command should display tfidf_score, proximity_score,
        and final_weighted_score for each result."""
        import src.main as main_mod

        idx = _proximity_index()
        index_file = tmp_path / "index.json"
        idx.save_to_file(str(index_file))

        original_path = main_mod.INDEX_PATH
        main_mod.INDEX_PATH = str(index_file)
        try:
            output = self._run_commands(["load", "find good friends", "quit"])
        finally:
            main_mod.INDEX_PATH = original_path

        assert "tfidf_score:" in output
        assert "proximity_score:" in output
        assert "final_weighted_score:" in output

    def _run_commands(self, commands: list[str]) -> str:
        from src.main import run_shell
        user_input = "\n".join(commands)
        with patch("sys.stdin", StringIO(user_input)), \
             patch("sys.stdout", new_callable=StringIO) as mock_out:
            run_shell()
        return mock_out.getvalue()


# ---------------------------------------------------------------------------
# Levenshtein distance
# ---------------------------------------------------------------------------


class TestLevenshtein:
    def test_identical_strings(self):
        assert levenshtein("fish", "fish") == 0

    def test_single_insertion(self):
        assert levenshtein("fis", "fish") == 1

    def test_single_deletion(self):
        assert levenshtein("fish", "fis") == 1

    def test_single_substitution(self):
        assert levenshtein("fish", "fosh") == 1

    def test_completely_different(self):
        assert levenshtein("abc", "xyz") == 3

    def test_empty_strings(self):
        assert levenshtein("", "") == 0

    def test_one_empty(self):
        assert levenshtein("", "hello") == 5
        assert levenshtein("hello", "") == 5

    def test_symmetric(self):
        assert levenshtein("kitten", "sitting") == levenshtein("sitting", "kitten")


# ---------------------------------------------------------------------------
# Helper index for suggestion tests
# ---------------------------------------------------------------------------

def _suggestion_index() -> InvertedIndex:
    """Index with a controlled vocabulary for suggestion testing.

    Vocabulary: fish, fishing, filter, friends, good, food, foot, fine, find
    """
    idx = InvertedIndex()
    idx.add_document("http://a.com", "fish fish fish fishing filter friends")
    idx.add_document("http://b.com", "good food foot fine find find find")
    return idx


# ---------------------------------------------------------------------------
# SuggestionEngine
# ---------------------------------------------------------------------------


class TestSuggestionEngine:
    def test_misspelled_returns_correction(self):
        """'fosh' should suggest 'fish' (edit distance 1)."""
        engine = SuggestionEngine(_suggestion_index())
        results = engine.suggest("fosh")
        terms = [s.term for s in results]
        assert "fish" in terms

    def test_exact_match_returns_no_suggestions(self):
        """If the word exists in the index, no suggestions are needed."""
        engine = SuggestionEngine(_suggestion_index())
        results = engine.suggest("fish")
        assert results == []

    def test_suggestions_ranked_by_score(self):
        """Suggestions should be sorted by descending combined score."""
        engine = SuggestionEngine(_suggestion_index())
        results = engine.suggest("fosh")
        scores = [s.score for s in results]
        assert scores == sorted(scores, reverse=True)

    def test_suggestion_score_formula(self):
        """Each suggestion's score should equal ALPHA * similarity + BETA * frequency."""
        engine = SuggestionEngine(_suggestion_index())
        results = engine.suggest("fosh")
        for s in results:
            expected = ALPHA * s.similarity_score + BETA * s.frequency_score
            assert s.score == pytest.approx(expected)

    def test_max_results_limit(self):
        """Should return at most max_results suggestions."""
        engine = SuggestionEngine(_suggestion_index())
        results = engine.suggest("f", max_results=3)
        assert len(results) <= 3

    def test_empty_token_returns_empty(self):
        engine = SuggestionEngine(_suggestion_index())
        assert engine.suggest("") == []

    def test_wildly_different_word_excluded(self):
        """A token very different from all vocabulary should return limited results."""
        engine = SuggestionEngine(_suggestion_index())
        results = engine.suggest("zzzzzzzzzzz")
        # Should be empty or only contain items within MAX_EDIT_DISTANCE
        for s in results:
            assert s.edit_distance <= MAX_EDIT_DISTANCE

    def test_case_insensitive(self):
        """Suggestions should work regardless of input case."""
        engine = SuggestionEngine(_suggestion_index())
        results = engine.suggest("FOSH")
        terms = [s.term for s in results]
        assert "fish" in terms


# ---------------------------------------------------------------------------
# suggest_for_query (multi-token)
# ---------------------------------------------------------------------------


class TestSuggestForQuery:
    def test_single_unknown_token(self):
        """A query with one unknown word should get suggestions for it."""
        engine = SuggestionEngine(_suggestion_index())
        token, results = engine.suggest_for_query("fosh")
        assert token == "fosh"
        terms = [s.term for s in results]
        assert "fish" in terms

    def test_all_known_tokens_returns_empty(self):
        """A query where every word is in the index needs no suggestions."""
        engine = SuggestionEngine(_suggestion_index())
        token, results = engine.suggest_for_query("fish good")
        assert token == ""
        assert results == []

    def test_first_unknown_token_gets_suggestions(self):
        """When multiple tokens are unknown, suggest for the first one."""
        engine = SuggestionEngine(_suggestion_index())
        token, results = engine.suggest_for_query("fosh goot")
        assert token == "fosh"
        terms = [s.term for s in results]
        assert "fish" in terms

    def test_mixed_known_and_unknown(self):
        """If first token is known but second is not, suggest for second."""
        engine = SuggestionEngine(_suggestion_index())
        token, results = engine.suggest_for_query("fish goot")
        assert token == "goot"
        terms = [s.term for s in results]
        assert "good" in terms


# ---------------------------------------------------------------------------
# Integration: SearchEngine.suggest()
# ---------------------------------------------------------------------------


class TestSearchEngineSuggest:
    def test_suggest_via_search_engine(self):
        """SearchEngine.suggest should delegate to SuggestionEngine."""
        engine = SearchEngine(_suggestion_index())
        token, results = engine.suggest("fosh")
        assert token == "fosh"
        terms = [s.term for s in results]
        assert "fish" in terms

    def test_no_suggestions_for_valid_query(self):
        engine = SearchEngine(_suggestion_index())
        token, results = engine.suggest("fish")
        assert token == ""
        assert results == []

    def test_find_then_suggest_workflow(self):
        """When find returns empty, suggest should provide alternatives."""
        engine = SearchEngine(_suggestion_index())
        search_results = engine.find("fosh")
        assert search_results == []
        token, suggestions = engine.suggest("fosh")
        assert token == "fosh"
        assert len(suggestions) > 0
        assert "fish" in [s.term for s in suggestions]


# ---------------------------------------------------------------------------
# CLI integration: "Did you mean" output
# ---------------------------------------------------------------------------


class TestCLISuggestions:
    def _run_commands(self, commands: list[str]) -> str:
        from src.main import run_shell

        user_input = "\n".join(commands)
        with patch("sys.stdin", StringIO(user_input)), \
             patch("sys.stdout", new_callable=StringIO) as mock_out:
            run_shell()
        return mock_out.getvalue()

    def test_did_you_mean_shown_on_no_results(self, tmp_path):
        """'find fosh' should print 'Did you mean:' with suggestions."""
        import src.main as main_mod

        idx = _suggestion_index()
        index_file = tmp_path / "index.json"
        idx.save_to_file(str(index_file))

        original_path = main_mod.INDEX_PATH
        main_mod.INDEX_PATH = str(index_file)
        try:
            output = self._run_commands(["load", "find fosh", "quit"])
        finally:
            main_mod.INDEX_PATH = original_path

        assert "Instead of 'fosh', did you mean:" in output
        assert "fish" in output

    def test_no_did_you_mean_on_valid_results(self, tmp_path):
        """'find fish' should NOT show 'Did you mean:'."""
        import src.main as main_mod

        idx = _suggestion_index()
        index_file = tmp_path / "index.json"
        idx.save_to_file(str(index_file))

        original_path = main_mod.INDEX_PATH
        main_mod.INDEX_PATH = str(index_file)
        try:
            output = self._run_commands(["load", "find fish", "quit"])
        finally:
            main_mod.INDEX_PATH = original_path

        assert "Did you mean:" not in output
