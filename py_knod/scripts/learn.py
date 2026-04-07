"""Continuous learning — fetch Wikipedia articles in our topic area and ingest them.

Picks articles from Wikipedia categories related to knowledge graphs, AI,
retrieval systems, and computing infrastructure. Fetches plain text, ingests
into the persistent graph, and keeps going until stopped (Ctrl-C) or --rounds
is exhausted.

Usage:
  python -m py_knod.scripts.learn                # run until Ctrl-C
  python -m py_knod.scripts.learn --rounds 5     # ingest 5 articles then stop
  python -m py_knod.scripts.learn --fresh        # start with empty graph
"""

import json
import logging
import os
import random
import re
import sys
import time
import urllib.request
import urllib.parse

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("learn")

from py_knod.config import Config
from py_knod.handler import Handler

# --- Wikipedia categories to sample from ---
CATEGORIES = [
	"Knowledge_graphs",
	"Graph_databases",
	"Semantic_Web",
	"Knowledge_representation",
	"Ontology_(information_science)",
	"Graph_neural_networks",
	"Artificial_neural_networks",
	"Deep_learning",
	"Machine_learning",
	"Natural_language_processing",
	"Information_retrieval",
	"Search_algorithms",
	"Word_embeddings",
	"Dimensionality_reduction",
	"Graph_theory",
	"Graph_algorithms",
	"Markov_chain_Monte_Carlo",
	"Bayesian_statistics",
	"Data_structures",
	"Computer_memory",
	"Cache_(computing)",
	"Data_centers",
	"Server_hardware",
	"Computer_networking",
	"Application_programming_interfaces",
	"Distributed_computing",
	"Database_management_systems",
	"Recommender_systems",
]

GRAPH_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "bin", "data"))
GRAPH_BASE = os.path.join(GRAPH_DIR, "test_edges")  # .graph / .pt added by save_all
MAX_CHARS = 20_000
UA = "knod-learner/1.0 (educational project; contact: none)"


def wiki_api(params: dict) -> dict:
	"""Call the Wikipedia API and return JSON."""
	params["format"] = "json"
	url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
	req = urllib.request.Request(url, headers={"User-Agent": UA})
	with urllib.request.urlopen(req, timeout=30) as resp:
		return json.loads(resp.read().decode("utf-8"))


def get_category_members(category: str, limit: int = 50) -> list[str]:
	"""Get article titles from a Wikipedia category."""
	data = wiki_api(
		{
			"action": "query",
			"list": "categorymembers",
			"cmtitle": f"Category:{category}",
			"cmtype": "page",
			"cmlimit": str(limit),
		}
	)
	return [m["title"] for m in data.get("query", {}).get("categorymembers", [])]


def get_article_text(title: str) -> str:
	"""Fetch plain text extract of a Wikipedia article."""
	data = wiki_api(
		{
			"action": "query",
			"titles": title,
			"prop": "extracts",
			"explaintext": "1",
			"exlimit": "1",
		}
	)
	pages = data.get("query", {}).get("pages", {})
	for page in pages.values():
		text = page.get("extract", "")
		if text:
			# Clean up
			text = re.sub(r"\n{3,}", "\n\n", text)
			if len(text) > MAX_CHARS:
				text = text[:MAX_CHARS]
			return text
	return ""


def pick_random_article(seen: set[str]) -> tuple[str, str] | None:
	"""Pick a random article from a random category. Returns (title, text) or None."""
	random.shuffle(CATEGORIES)
	for cat in CATEGORIES:
		try:
			members = get_category_members(cat, limit=50)
			random.shuffle(members)
			for title in members:
				if title in seen:
					continue
				if title.startswith("Category:") or title.startswith("List of"):
					continue
				text = get_article_text(title)
				if len(text) > 500:  # skip stubs
					return title, text
		except Exception as e:
			log.warning("Failed to fetch from %s: %s", cat, e)
			continue
	return None


def main():
	import argparse

	parser = argparse.ArgumentParser(description="Continuous Wikipedia learning")
	parser.add_argument("--rounds", type=int, default=0, help="Number of articles to ingest (0=unlimited)")
	parser.add_argument("--fresh", action="store_true", help="Start with empty graph")
	args = parser.parse_args()

	cfg = Config.load()
	# Point graph_path at our persistent store
	cfg.graph_path = GRAPH_BASE + ".graph"
	os.makedirs(GRAPH_DIR, exist_ok=True)

	if args.fresh:
		for ext in (".graph", ".pt"):
			p = GRAPH_BASE + ext
			if os.path.exists(p):
				os.remove(p)

	handler = Handler(cfg)
	handler.init()

	seen = set(t.source for t in handler.graph.thoughts.values())
	round_num = 0
	total_new = 0

	print(f"\n{'=' * 60}")
	print(f"  KNOD LEARNER — ingesting Wikipedia articles")
	print(f"  Graph: {handler.graph.num_thoughts} thoughts, {handler.graph.num_edges} edges")
	print(f"  Known sources: {len(seen)}")
	print(f"  Rounds: {'unlimited' if args.rounds == 0 else args.rounds}")
	print(f"{'=' * 60}\n")

	try:
		while True:
			if args.rounds > 0 and round_num >= args.rounds:
				break

			result = pick_random_article(seen)
			if result is None:
				log.warning("Could not find any new articles, retrying...")
				time.sleep(5)
				continue

			title, text = result
			seen.add(title)
			round_num += 1

			print(f"\n--- Round {round_num} ---")
			print(f"  Article: {title} ({len(text):,} chars)")

			before_t = handler.graph.num_thoughts
			before_e = handler.graph.num_edges
			t0 = time.time()
			handler.ingest_sync(text, source=title)
			dt = time.time() - t0
			new_thoughts = handler.graph.num_thoughts - before_t
			new_edges = handler.graph.num_edges - before_e
			total_new += new_thoughts

			g = handler.graph
			print(f"  Ingested: +{new_thoughts} thoughts, +{new_edges} edges ({dt:.1f}s)")
			print(f"  Graph:    {g.num_thoughts} thoughts, {g.num_edges} edges, maturity={g.maturity:.3f}")

	except KeyboardInterrupt:
		print("\n\nStopping...")

	handler.save()
	print(f"\n{'=' * 60}")
	print(f"  DONE — {round_num} rounds, +{total_new} thoughts ingested")
	print(f"  Final: {handler.graph.num_thoughts} thoughts, {handler.graph.num_edges} edges")
	print(f"{'=' * 60}")


if __name__ == "__main__":
	main()
