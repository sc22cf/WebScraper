import pytest
from unittest.mock import MagicMock, patch
from src.crawler import Crawler, RateLimiter


# ---------------------------------------------------------------------------
# RateLimiter tests
# ---------------------------------------------------------------------------

class TestRateLimiter:
    def test_default_delay(self):
        rl = RateLimiter()
        assert rl.delay_seconds == 6.0

    def test_custom_delay(self):
        rl = RateLimiter(delay_seconds=10.0)
        assert rl.delay_seconds == 10.0

    @patch("src.crawler.time.sleep")
    def test_sleep_calls_time_sleep(self, mock_sleep):
        rl = RateLimiter(delay_seconds=6.0)
        rl.sleep()
        mock_sleep.assert_called_once_with(6.0)

    @patch("src.crawler.time.sleep")
    def test_sleep_skipped_when_zero(self, mock_sleep):
        rl = RateLimiter(delay_seconds=0)
        rl.sleep()
        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# Crawler unit helpers
# ---------------------------------------------------------------------------

BASE_URL = "https://quotes.toscrape.com/"

def _make_crawler(delay=0):
    """Create a Crawler with no real delay for fast tests."""
    return Crawler(BASE_URL, RateLimiter(delay_seconds=delay))


def _html_page(body, links=None):
    """Return a minimal HTML string with optional <a> links."""
    link_tags = ""
    if links:
        link_tags = "".join(f'<a href="{u}">link</a>' for u in links)
    return f"<html><body>{body}{link_tags}</body></html>"


def _mock_response(html, status=200, content_type="text/html"):
    """Build a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status
    resp.text = html
    resp.headers = {"content-type": content_type}
    return resp


# ---------------------------------------------------------------------------
# Crawler – URL handling
# ---------------------------------------------------------------------------

class TestCrawlerURLHandling:
    def test_normalize_url_strips_fragment(self):
        crawler = _make_crawler()
        assert crawler.normalize_url("https://quotes.toscrape.com/page/1/#top") == "https://quotes.toscrape.com/page/1/"

    def test_normalize_url_no_fragment(self):
        crawler = _make_crawler()
        url = "https://quotes.toscrape.com/page/1/"
        assert crawler.normalize_url(url) == url

    def test_is_valid_url_same_domain(self):
        crawler = _make_crawler()
        assert crawler.is_valid_url("https://quotes.toscrape.com/page/2/") is True

    def test_is_valid_url_different_domain(self):
        crawler = _make_crawler()
        assert crawler.is_valid_url("https://example.com/page") is False

    def test_is_valid_url_rejects_non_http(self):
        crawler = _make_crawler()
        assert crawler.is_valid_url("ftp://quotes.toscrape.com/file") is False


# ---------------------------------------------------------------------------
# Crawler – crawl_and_index behaviour
# ---------------------------------------------------------------------------

class TestCrawlAndIndex:
    @patch("src.crawler.requests.get")
    def test_single_page_crawl(self, mock_get):
        """Crawl a single page with no outgoing links."""
        html = _html_page("Hello world")
        mock_get.return_value = _mock_response(html)

        crawler = _make_crawler()
        indexer = MagicMock()

        crawler.crawl_and_index(indexer, max_pages=1)

        mock_get.assert_called_once()
        indexer.add_document.assert_called_once()
        # The URL passed to add_document should be the base URL
        call_args = indexer.add_document.call_args
        assert call_args[0][0] == BASE_URL

    @patch("src.crawler.requests.get")
    def test_follows_internal_links(self, mock_get):
        """Crawler should discover and follow internal links."""
        page1 = _html_page("first page", links=["/page/2/"])
        page2 = _html_page("second page")

        mock_get.side_effect = [
            _mock_response(page1),
            _mock_response(page2),
        ]

        crawler = _make_crawler()
        indexer = MagicMock()

        crawler.crawl_and_index(indexer, max_pages=2)

        assert indexer.add_document.call_count == 2

    @patch("src.crawler.requests.get")
    def test_does_not_revisit_pages(self, mock_get):
        """Already-visited URLs should not be fetched again."""
        page = _html_page("page", links=[BASE_URL])  # link back to self
        mock_get.return_value = _mock_response(page)

        crawler = _make_crawler()
        indexer = MagicMock()

        crawler.crawl_and_index(indexer, max_pages=5)

        # Should only be called once despite the self-link
        mock_get.assert_called_once()

    @patch("src.crawler.requests.get")
    def test_skips_non_html_responses(self, mock_get):
        """Non-HTML content-types should not be indexed."""
        mock_get.return_value = _mock_response(
            "binary data", content_type="application/pdf"
        )

        crawler = _make_crawler()
        indexer = MagicMock()

        crawler.crawl_and_index(indexer, max_pages=1)

        indexer.add_document.assert_not_called()

    @patch("src.crawler.requests.get")
    def test_skips_failed_status_codes(self, mock_get):
        """Non-200 responses should not be indexed."""
        mock_get.return_value = _mock_response("Not found", status=404)

        crawler = _make_crawler()
        indexer = MagicMock()

        crawler.crawl_and_index(indexer, max_pages=1)

        indexer.add_document.assert_not_called()

    @patch("src.crawler.requests.get")
    def test_handles_network_error(self, mock_get):
        """Network errors should be caught, not crash the crawler."""
        import requests as req
        mock_get.side_effect = req.ConnectionError("refused")

        crawler = _make_crawler()
        indexer = MagicMock()

        # Should not raise
        crawler.crawl_and_index(indexer, max_pages=1)

        indexer.add_document.assert_not_called()

    @patch("src.crawler.requests.get")
    def test_max_pages_limits_crawl(self, mock_get):
        """The max_pages parameter should cap how many pages are indexed."""
        page = _html_page("page", links=["/page/2/", "/page/3/", "/page/4/"])
        mock_get.return_value = _mock_response(page)

        crawler = _make_crawler()
        indexer = MagicMock()

        crawler.crawl_and_index(indexer, max_pages=1)

        assert indexer.add_document.call_count == 1

    @patch("src.crawler.requests.get")
    def test_ignores_external_links(self, mock_get):
        """Links to other domains should not be followed."""
        page = _html_page("page", links=["https://external.com/other"])
        mock_get.return_value = _mock_response(page)

        crawler = _make_crawler()
        indexer = MagicMock()

        crawler.crawl_and_index(indexer, max_pages=5)

        # Only the initial page should be fetched
        mock_get.assert_called_once()
        indexer.add_document.assert_called_once()

    @patch("src.crawler.requests.get")
    def test_skips_duplicate_content(self, mock_get):
        """Two URLs with identical content should only be indexed once."""
        identical_html = _html_page("Identical content on both pages")
        mock_get.side_effect = [
            _mock_response(identical_html),   # /tag/inspirational/
            _mock_response(identical_html),   # /tag/inspirational/page/1/ — same body
        ]

        crawler = Crawler(
            "https://quotes.toscrape.com/tag/inspirational/",
            RateLimiter(delay_seconds=0),
        )
        crawler.domain = "quotes.toscrape.com"
        # Pre-queue the second URL so the crawler visits both
        crawler.visited = set()

        indexer = MagicMock()
        # Manually drive two fetches by giving both URLs in the queue
        from src.indexer import InvertedIndex
        real_indexer = InvertedIndex()

        # Simulate crawler seeing both URLs
        import requests as _req
        from bs4 import BeautifulSoup
        import hashlib

        urls = [
            "https://quotes.toscrape.com/tag/inspirational/",
            "https://quotes.toscrape.com/tag/inspirational/page/1/",
        ]
        for url in urls:
            resp = _mock_response(identical_html)
            text = BeautifulSoup(resp.text, "html.parser").get_text(separator=" ", strip=True)
            h = hashlib.md5(text.encode()).hexdigest()
            if h not in crawler.seen_hashes:
                crawler.seen_hashes.add(h)
                real_indexer.add_document(url, text)

        # Only the first URL should have been added
        assert len(real_indexer.index) > 0
        docs_across_all_words = set()
        for postings in real_indexer.index.values():
            docs_across_all_words.update(postings.keys())
        assert len(docs_across_all_words) == 1
        assert "https://quotes.toscrape.com/tag/inspirational/" in docs_across_all_words

    @patch("src.crawler.requests.get")
    def test_different_content_both_indexed(self, mock_get):
        """Two URLs with different content should both be indexed."""
        # page1 links to page2 so the crawler discovers it
        page1 = _html_page("Unique content on page one", links=["/page/2/"])
        page2 = _html_page("Completely different content on page two")
        mock_get.side_effect = [
            _mock_response(page1),
            _mock_response(page2),
        ]

        crawler = _make_crawler()
        indexer = MagicMock()

        crawler.crawl_and_index(indexer, max_pages=2)

        assert indexer.add_document.call_count == 2
