import pytest
from src.indexer import InvertedIndex


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_basic_sentence(self):
        tokens = InvertedIndex.tokenize("Hello World")
        assert tokens == ["hello", "world"]

    def test_lowercases_all_words(self):
        tokens = InvertedIndex.tokenize("Good MORNING Vietnam")
        assert tokens == ["good", "morning", "vietnam"]

    def test_strips_punctuation(self):
        tokens = InvertedIndex.tokenize("it's a test, right?")
        assert tokens == ["it", "s", "a", "test", "right"]

    def test_strips_numbers(self):
        tokens = InvertedIndex.tokenize("page 1 has 200 quotes")
        assert tokens == ["page", "has", "quotes"]

    def test_empty_string(self):
        assert InvertedIndex.tokenize("") == []

    def test_only_punctuation(self):
        assert InvertedIndex.tokenize("!@#$%^&*()") == []


# ---------------------------------------------------------------------------
# add_document – single document
# ---------------------------------------------------------------------------

class TestAddSingleDocument:
    def test_word_appears_in_index(self):
        idx = InvertedIndex()
        idx.add_document("http://example.com", "hello world")
        assert "hello" in idx.index
        assert "world" in idx.index

    def test_frequency_count(self):
        idx = InvertedIndex()
        idx.add_document("http://example.com", "go go go")
        entry = idx.index["go"]["http://example.com"]
        assert entry["frequency"] == 3

    def test_positions_recorded(self):
        idx = InvertedIndex()
        idx.add_document("http://example.com", "the cat sat on the mat")
        entry = idx.index["the"]["http://example.com"]
        assert entry["positions"] == [0, 4]

    def test_single_word_frequency_and_position(self):
        idx = InvertedIndex()
        idx.add_document("http://example.com", "alone")
        entry = idx.index["alone"]["http://example.com"]
        assert entry["frequency"] == 1
        assert entry["positions"] == [0]


# ---------------------------------------------------------------------------
# Case-insensitive indexing
# ---------------------------------------------------------------------------

class TestCaseInsensitivity:
    def test_mixed_case_same_entry(self):
        idx = InvertedIndex()
        idx.add_document("http://example.com", "Good good GOOD")
        assert "good" in idx.index
        assert "Good" not in idx.index
        assert "GOOD" not in idx.index
        assert idx.index["good"]["http://example.com"]["frequency"] == 3

    def test_lookup_is_case_insensitive(self):
        idx = InvertedIndex()
        idx.add_document("http://example.com", "Hello")
        assert idx.get_entry("HELLO") is not None
        assert idx.get_entry("hello") is not None


# ---------------------------------------------------------------------------
# Empty / edge-case documents
# ---------------------------------------------------------------------------

class TestEmptyDocument:
    def test_empty_text_does_not_add_entries(self):
        idx = InvertedIndex()
        idx.add_document("http://example.com", "")
        assert idx.index == {}

    def test_whitespace_only_text(self):
        idx = InvertedIndex()
        idx.add_document("http://example.com", "   \t\n  ")
        assert idx.index == {}

    def test_punctuation_only_text(self):
        idx = InvertedIndex()
        idx.add_document("http://example.com", "!!! --- ???")
        assert idx.index == {}


# ---------------------------------------------------------------------------
# Multiple documents
# ---------------------------------------------------------------------------

class TestMultipleDocuments:
    def test_same_word_across_two_documents(self):
        idx = InvertedIndex()
        idx.add_document("http://a.com", "hello world")
        idx.add_document("http://b.com", "hello there")

        assert "http://a.com" in idx.index["hello"]
        assert "http://b.com" in idx.index["hello"]

    def test_positions_are_per_document(self):
        idx = InvertedIndex()
        idx.add_document("http://a.com", "cat dog cat")
        idx.add_document("http://b.com", "dog cat")

        assert idx.index["cat"]["http://a.com"]["positions"] == [0, 2]
        assert idx.index["cat"]["http://b.com"]["positions"] == [1]


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

class TestLookupHelpers:
    def test_get_entry_existing_word(self):
        idx = InvertedIndex()
        idx.add_document("http://a.com", "sunshine")
        entry = idx.get_entry("sunshine")
        assert entry is not None
        assert "http://a.com" in entry

    def test_get_entry_missing_word(self):
        idx = InvertedIndex()
        idx.add_document("http://a.com", "sunshine")
        assert idx.get_entry("rain") is None

    def test_get_documents_returns_urls(self):
        idx = InvertedIndex()
        idx.add_document("http://a.com", "word")
        idx.add_document("http://b.com", "word")
        docs = idx.get_documents("word")
        assert sorted(docs) == ["http://a.com", "http://b.com"]

    def test_get_documents_missing_word(self):
        idx = InvertedIndex()
        assert idx.get_documents("nonexistent") == []

    def test_page_count_empty_index(self):
        idx = InvertedIndex()
        assert idx.page_count == 0

    def test_page_count_single_document(self):
        idx = InvertedIndex()
        idx.add_document("http://a.com", "hello world")
        assert idx.page_count == 1

    def test_page_count_multiple_documents(self):
        idx = InvertedIndex()
        idx.add_document("http://a.com", "shared word")
        idx.add_document("http://b.com", "shared word")
        idx.add_document("http://c.com", "unique term here")
        assert idx.page_count == 3
