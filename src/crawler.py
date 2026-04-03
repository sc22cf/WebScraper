import hashlib
import time
import requests
from urllib.parse import urljoin, urlparse, urldefrag
from bs4 import BeautifulSoup
from src.indexer import InvertedIndex

class RateLimiter:
    """Handles the politeness delay between requests."""
    def __init__(self, delay_seconds: float = 6.0):
        self.delay_seconds = delay_seconds

    def sleep(self) -> None:
        """Pause execution for the specified delay."""
        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)

class Crawler:
    """
    Crawls a target website, respecting domain boundaries and politeness.
    Extracts text to build an inverted index.
    """
    def __init__(self, base_url: str, rate_limiter: RateLimiter):
        self.base_url = base_url
        self.rate_limiter = rate_limiter
        self.visited: set[str] = set()
        self.seen_hashes: set[str] = set()
        self.domain = urlparse(base_url).netloc

    def normalize_url(self, url: str) -> str:
        """Strip fragment identifiers to avoid treating # as separate pages."""
        url, _ = urldefrag(url)
        return url

    def is_valid_url(self, url: str) -> bool:
        """Check if it's an HTTP/HTTPS URL and within our target domain."""
        parsed = urlparse(url)
        return parsed.scheme in ('http', 'https') and parsed.netloc == self.domain

    def crawl_and_index(self, indexer: InvertedIndex, max_pages: int = -1) -> None:
        """Crawl the website using breadth-first search (BFS) and index each page.

        BFS is used because it explores pages level-by-level outward from the
        start URL, which naturally discovers the most important (closest-linked)
        pages first.  A FIFO queue (``urls_to_visit``) drives the traversal
        while a ``visited`` set prevents re-fetching.

        Content-hashing (MD5) is performed on the extracted text so that
        duplicate pages served at different URLs are indexed only once.

        Complexity
        ----------
        Let *P* = number of pages crawled, *L* = average links per page.

        * Each page is dequeued, fetched, and parsed exactly once → **O(P)**
          page visits.
        * URL membership checks use a ``set``, so ``in self.visited`` is
          **O(1)** amortised per check.
        * Duplicate-content lookup via ``seen_hashes`` is also **O(1)**
          amortised.
        * Link discovery iterates every anchor on a page → **O(L)** per page.
        * Overall traversal complexity: **O(P · L)**.
        * Network I/O dominates wall-clock time; the politeness delay of ≥ 6 s
          per request means total crawl time ≈ 6 · P seconds.
        """
        urls_to_visit = [self.base_url]
        pages_crawled = 0

        while urls_to_visit:
            if 0 < max_pages <= pages_crawled:
                break

            current_url = urls_to_visit.pop(0)
            current_url = self.normalize_url(current_url)

            if current_url in self.visited:
                continue

            # Mark as visited
            self.visited.add(current_url)
            print(f"Crawling: {current_url}")

            try:
                # Add politeness delay before requesting
                self.rate_limiter.sleep()
                response = requests.get(current_url, timeout=10)
                
                # Only process successful HTML responses
                if response.status_code != 200:
                    print(f"Failed to fetch {current_url}: HTTP {response.status_code}")
                    continue
                
                content_type = response.headers.get('content-type', '').lower()
                if 'text/html' not in content_type:
                    continue

                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract text for the index (space separator ensures words don't merge)
                page_text = soup.get_text(separator=' ', strip=True)

                # Skip pages whose content we've already indexed (e.g. /tag/x/ == /tag/x/page/1/)
                content_hash = hashlib.md5(page_text.encode()).hexdigest()
                if content_hash in self.seen_hashes:
                    print(f"Skipping duplicate content: {current_url}")
                    continue
                self.seen_hashes.add(content_hash)

                indexer.add_document(current_url, page_text)
                pages_crawled += 1

                # Extract and queue valid internal links
                for link in soup.find_all('a', href=True):
                    next_url = urljoin(current_url, link['href'])
                    next_url = self.normalize_url(next_url)
                    
                    if self.is_valid_url(next_url) and next_url not in self.visited:
                        if next_url not in urls_to_visit:
                            urls_to_visit.append(next_url)
                            
            except requests.RequestException as e:
                print(f"Network error on {current_url}: {e}")
