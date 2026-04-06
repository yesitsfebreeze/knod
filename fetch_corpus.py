"""
Fetch Wikipedia articles and cache them as plain text files in corpus/.

Run once (or whenever you want to refresh). The other scripts read from corpus/.

Usage: python fetch_corpus.py [--force]
  --force  re-download even if files already exist
"""

import argparse
import os
import re
import sys
import urllib.request

CORPUS_DIR = "corpus"

# Each entry: (filename_stem, url, category)
# category is "on" (turtle-related) or "off" (noise).
SOURCES = [
    # --- on-topic: turtles, anatomy, ecology, conservation ---
    ("turtle", "https://en.wikipedia.org/wiki/Turtle", "on"),
    ("sea_turtle", "https://en.wikipedia.org/wiki/Sea_turtle", "on"),
    ("tortoise", "https://en.wikipedia.org/wiki/Tortoise", "on"),
    ("turtle_shell", "https://en.wikipedia.org/wiki/Turtle_shell", "on"),
    ("green_sea_turtle", "https://en.wikipedia.org/wiki/Green_sea_turtle", "on"),
    ("leatherback", "https://en.wikipedia.org/wiki/Leatherback_sea_turtle", "on"),
    ("hawksbill", "https://en.wikipedia.org/wiki/Hawksbill_sea_turtle", "on"),
    ("loggerhead", "https://en.wikipedia.org/wiki/Loggerhead_sea_turtle", "on"),
    ("box_turtle", "https://en.wikipedia.org/wiki/Box_turtle", "on"),
    ("snapping_turtle", "https://en.wikipedia.org/wiki/Common_snapping_turtle", "on"),
    ("flatback", "https://en.wikipedia.org/wiki/Flatback_sea_turtle", "on"),
    ("turtle_excluder", "https://en.wikipedia.org/wiki/Turtle_excluder_device", "on"),
    # --- off-topic: unrelated domains ---
    ("surfing", "https://en.wikipedia.org/wiki/Surfing", "off"),
    ("volcano", "https://en.wikipedia.org/wiki/Volcano", "off"),
    ("jazz", "https://en.wikipedia.org/wiki/Jazz", "off"),
    ("chess", "https://en.wikipedia.org/wiki/Chess", "off"),
    ("bicycle", "https://en.wikipedia.org/wiki/Bicycle", "off"),
]

MAX_CHARS = 10000


def strip_html(html: str) -> str:
    """Extract readable text from HTML paragraph content."""
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL)
    paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", html, flags=re.DOTALL)
    text = "\n\n".join(paragraphs)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def fetch_one(name: str, url: str) -> str:
    """Fetch and extract text from a Wikipedia article."""
    req = urllib.request.Request(url, headers={"User-Agent": "knod-test/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    text = strip_html(html)
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]
    return text


def main():
    parser = argparse.ArgumentParser(
        description="Fetch corpus articles for knod testing"
    )
    parser.add_argument(
        "--force", action="store_true", help="re-download existing files"
    )
    args = parser.parse_args()

    os.makedirs(CORPUS_DIR, exist_ok=True)

    total = len(SOURCES)
    fetched = 0
    skipped = 0
    errors = 0

    for name, url, category in SOURCES:
        path = os.path.join(CORPUS_DIR, f"{name}.txt")
        if os.path.exists(path) and not args.force:
            size = os.path.getsize(path)
            print(f"  skip  {name} ({size:,} bytes, already cached)")
            skipped += 1
            continue

        try:
            text = fetch_one(name, url)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            fetched += 1
            print(f"  fetch {name} [{category}] -> {len(text):,} chars")
        except Exception as e:
            errors += 1
            print(f"  ERROR {name}: {e}")

    # Write a manifest so other scripts know the category of each file.
    manifest_path = os.path.join(CORPUS_DIR, "manifest.txt")
    with open(manifest_path, "w", encoding="utf-8") as f:
        for name, url, category in SOURCES:
            f.write(f"{name}\t{category}\t{url}\n")

    print(f"\nDone: {fetched} fetched, {skipped} skipped, {errors} errors (of {total})")
    print(f"Corpus directory: {CORPUS_DIR}/")

    if errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
