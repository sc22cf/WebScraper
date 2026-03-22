import os
import pytest
from src.indexer import InvertedIndex


# ---------------------------------------------------------------------------
# save_to_file
# ---------------------------------------------------------------------------

class TestSaveToFile:
    def test_creates_file(self, tmp_path):
        idx = InvertedIndex()
        idx.add_document("http://example.com", "hello world")
        path = tmp_path / "index.json"

        idx.save_to_file(str(path))

        assert path.exists()

    def test_creates_parent_directories(self, tmp_path):
        idx = InvertedIndex()
        idx.add_document("http://example.com", "test")
        path = tmp_path / "data" / "nested" / "index.json"

        idx.save_to_file(str(path))

        assert path.exists()

    def test_empty_index_creates_file(self, tmp_path):
        idx = InvertedIndex()
        path = tmp_path / "index.json"

        idx.save_to_file(str(path))

        assert path.exists()


# ---------------------------------------------------------------------------
# load_from_file
# ---------------------------------------------------------------------------

class TestLoadFromFile:
    def test_restores_identical_index(self, tmp_path):
        idx = InvertedIndex()
        idx.add_document("http://a.com", "the cat sat on the mat")
        idx.add_document("http://b.com", "the dog sat")
        path = tmp_path / "index.json"

        idx.save_to_file(str(path))
        loaded = InvertedIndex.load_from_file(str(path))

        assert loaded.index == idx.index

    def test_empty_index_round_trip(self, tmp_path):
        idx = InvertedIndex()
        path = tmp_path / "index.json"

        idx.save_to_file(str(path))
        loaded = InvertedIndex.load_from_file(str(path))

        assert loaded.index == {}

    def test_missing_file_raises_error(self, tmp_path):
        path = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError):
            InvertedIndex.load_from_file(str(path))


# ---------------------------------------------------------------------------
# Loaded index retains full functionality
# ---------------------------------------------------------------------------

class TestLoadedIndexFunctionality:
    def _save_and_load(self, idx, tmp_path):
        path = tmp_path / "index.json"
        idx.save_to_file(str(path))
        return InvertedIndex.load_from_file(str(path))

    def test_get_entry_after_load(self, tmp_path):
        idx = InvertedIndex()
        idx.add_document("http://a.com", "sunshine rain")
        loaded = self._save_and_load(idx, tmp_path)

        entry = loaded.get_entry("sunshine")
        assert entry is not None
        assert "http://a.com" in entry

    def test_get_documents_after_load(self, tmp_path):
        idx = InvertedIndex()
        idx.add_document("http://a.com", "word")
        idx.add_document("http://b.com", "word")
        loaded = self._save_and_load(idx, tmp_path)

        docs = loaded.get_documents("word")
        assert sorted(docs) == ["http://a.com", "http://b.com"]

    def test_frequency_preserved(self, tmp_path):
        idx = InvertedIndex()
        idx.add_document("http://a.com", "go go go")
        loaded = self._save_and_load(idx, tmp_path)

        entry = loaded.get_entry("go")
        assert entry["http://a.com"]["frequency"] == 3

    def test_positions_preserved(self, tmp_path):
        idx = InvertedIndex()
        idx.add_document("http://a.com", "the cat sat on the mat")
        loaded = self._save_and_load(idx, tmp_path)

        entry = loaded.get_entry("the")
        assert entry["http://a.com"]["positions"] == [0, 4]
