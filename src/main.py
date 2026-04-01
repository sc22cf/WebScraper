import os
from src.crawler import Crawler, RateLimiter
from src.indexer import InvertedIndex
from src.search import SearchEngine

BASE_URL = "https://quotes.toscrape.com/"
INDEX_PATH = os.path.join("data", "index.json")


def run_shell() -> None:
    """Interactive command-line shell for the search engine."""
    index: InvertedIndex | None = None
    engine: SearchEngine | None = None

    print("Search Engine Shell — type 'help' for commands, 'quit' to exit.")

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not raw:
            continue

        parts = raw.split(maxsplit=1)
        command = parts[0].lower()
        argument = parts[1] if len(parts) > 1 else ""

        if command == "build":
            print("Crawling website and building index...")
            index = InvertedIndex()
            crawler = Crawler(BASE_URL, RateLimiter())
            crawler.crawl_and_index(index)
            index.save_to_file(INDEX_PATH)
            print(f"Index built and saved to {INDEX_PATH}.")
            print(f"Total pages indexed: {index.page_count}.")
            engine = SearchEngine(index)

        elif command == "load":
            try:
                index = InvertedIndex.load_from_file(INDEX_PATH)
                engine = SearchEngine(index)
                print(f"Index loaded from {INDEX_PATH}.")
            except FileNotFoundError:
                print(f"Error: index file not found at {INDEX_PATH}. Run 'build' first.")

        elif command == "print":
            if engine is None:
                print("No index loaded. Run 'build' or 'load' first.")
            else:
                print(engine.print_word(argument))

        elif command == "find":
            if engine is None:
                print("No index loaded. Run 'build' or 'load' first.")
            else:
                results = engine.find(argument)
                if results:
                    print(f"Found in {len(results)} page(s) (ranked by relevance):")
                    for r in results:
                        print(f"  page: {r.url}")
                        print(f"    tfidf_score: {r.tfidf_score:.4f}")
                        print(f"    proximity_score: {r.proximity_score:.4f}")
                        print(f"    final_weighted_score: {r.final_score:.4f}")
                else:
                    tokens = InvertedIndex.tokenize(argument)
                    if not tokens:
                        print("No search terms provided.")
                    else:
                        print(f"No pages found containing all terms: {tokens}")

        elif command in ("quit", "exit"):
            print("Exiting.")
            break

        elif command == "help":
            print("Commands:")
            print("  build              Crawl the website and save the index")
            print("  load               Load a previously saved index")
            print("  print <word>       Show index entry for a word")
            print("  find <query>       Find pages containing all query words")
            print("  quit / exit        Exit the shell")

        else:
            print(f"Unknown command: '{command}'. Type 'help' for usage.")


if __name__ == "__main__":
    run_shell()
