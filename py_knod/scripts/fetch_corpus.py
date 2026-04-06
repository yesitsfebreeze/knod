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
# category is "on" (herpetology-related) or "off" (noise).
SOURCES = [
	# --- turtles & tortoises ---
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
	("painted_turtle", "https://en.wikipedia.org/wiki/Painted_turtle", "on"),
	("aldabra_tortoise", "https://en.wikipedia.org/wiki/Aldabra_giant_tortoise", "on"),
	(
		"galapagos_tortoise",
		"https://en.wikipedia.org/wiki/Gal%C3%A1pagos_tortoise",
		"on",
	),
	("olive_ridley", "https://en.wikipedia.org/wiki/Olive_ridley_sea_turtle", "on"),
	("kemp_ridley", "https://en.wikipedia.org/wiki/Kemp%27s_ridley_sea_turtle", "on"),
	("chelonia", "https://en.wikipedia.org/wiki/Testudines", "on"),
	# --- snakes ---
	("snake", "https://en.wikipedia.org/wiki/Snake", "on"),
	("king_cobra", "https://en.wikipedia.org/wiki/King_cobra", "on"),
	("reticulated_python", "https://en.wikipedia.org/wiki/Reticulated_python", "on"),
	("boa_constrictor", "https://en.wikipedia.org/wiki/Boa_constrictor", "on"),
	("rattlesnake", "https://en.wikipedia.org/wiki/Rattlesnake", "on"),
	("venom", "https://en.wikipedia.org/wiki/Venom", "on"),
	# --- lizards ---
	("lizard", "https://en.wikipedia.org/wiki/Lizard", "on"),
	("komodo_dragon", "https://en.wikipedia.org/wiki/Komodo_dragon", "on"),
	("gecko", "https://en.wikipedia.org/wiki/Gecko", "on"),
	("iguana", "https://en.wikipedia.org/wiki/Iguana", "on"),
	("chameleon", "https://en.wikipedia.org/wiki/Chameleon", "on"),
	("monitor_lizard", "https://en.wikipedia.org/wiki/Monitor_lizard", "on"),
	# --- crocodilians ---
	("crocodilian", "https://en.wikipedia.org/wiki/Crocodilia", "on"),
	("nile_crocodile", "https://en.wikipedia.org/wiki/Nile_crocodile", "on"),
	("american_alligator", "https://en.wikipedia.org/wiki/American_alligator", "on"),
	("gharial", "https://en.wikipedia.org/wiki/Gharial", "on"),
	# --- amphibians (outgroup for comparison) ---
	("frog", "https://en.wikipedia.org/wiki/Frog", "on"),
	("salamander", "https://en.wikipedia.org/wiki/Salamander", "on"),
	("axolotl", "https://en.wikipedia.org/wiki/Axolotl", "on"),
	# --- reptile biology & conservation ---
	("reptile", "https://en.wikipedia.org/wiki/Reptile", "on"),
	("ectotherm", "https://en.wikipedia.org/wiki/Ectotherm", "on"),
	("scale_anatomy", "https://en.wikipedia.org/wiki/Scale_(anatomy)", "on"),
	("reptile_egg", "https://en.wikipedia.org/wiki/Egg", "on"),
	("hibernation", "https://en.wikipedia.org/wiki/Hibernation", "on"),
	("cites", "https://en.wikipedia.org/wiki/CITES", "on"),
	("iucn_red_list", "https://en.wikipedia.org/wiki/IUCN_Red_List", "on"),
	("wildlife_trade", "https://en.wikipedia.org/wiki/Wildlife_trade", "on"),
	# --- off-topic noise ---
	("surfing", "https://en.wikipedia.org/wiki/Surfing", "off"),
	("volcano", "https://en.wikipedia.org/wiki/Volcano", "off"),
	("jazz", "https://en.wikipedia.org/wiki/Jazz", "off"),
	("chess", "https://en.wikipedia.org/wiki/Chess", "off"),
	("bicycle", "https://en.wikipedia.org/wiki/Bicycle", "off"),
	("blockchain", "https://en.wikipedia.org/wiki/Blockchain", "off"),
	("opera", "https://en.wikipedia.org/wiki/Opera", "off"),
	("typography", "https://en.wikipedia.org/wiki/Typography", "off"),
]

MAX_CHARS = 25_000


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
	parser = argparse.ArgumentParser(description="Fetch corpus articles for knod testing")
	parser.add_argument("--force", action="store_true", help="re-download existing files")
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
			print(f"  skip  {name} [{category}] ({size:,} bytes, cached)")
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

	on_count = sum(1 for _, _, c in SOURCES if c == "on")
	off_count = sum(1 for _, _, c in SOURCES if c == "off")
	print(f"\nDone: {fetched} fetched, {skipped} skipped, {errors} errors")
	print(f"Corpus: {on_count} on-topic, {off_count} off-topic, {MAX_CHARS:,} chars/article max")
	print(f"Directory: {CORPUS_DIR}/")

	if errors > 0:
		sys.exit(1)


if __name__ == "__main__":
	main()
