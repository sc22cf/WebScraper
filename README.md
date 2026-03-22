# Python Web Crawler & Search Engine

A command-line search tool that crawls `https://quotes.toscrape.com/`, builds an inverted index, and lets you search for multi-word queries.

## Requirements
- Python 3.8+

## Setup
1. Create a virtual environment (optional but recommended)
2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the main program:
```bash
python -m src.main
```

Available commands in the interactive shell:
- `build`: Crawl the site and build the inverted index (respects the 6-second delay between requests).
- `load`: Load the previously built index from the `data/` directory.
- `print <word>`: Print the index entry for a specific word, showing URLs, frequency, and token positions.
- `find <terms...>`: Return pages that contain all of the provided query terms (AND semantics).
- `help`: See all commands.
- `exit` or `quit`: Exit the program.

## Architecture
- `src/crawler.py`: Polite BFS crawler targeting internal links on the site.
- `src/indexer.py`: Tokenizes content into lowercase alphanumeric words, stores inverted index.
- `src/search.py`: Intersects postings lists to perform AND queries.
- `src/main.py`: Command-line REPL.
- `tests/`: Module unit tests.