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
# category is "on" (knowledge systems / AI / retrieval) or "off" (noise).
SOURCES = [
	# --- knowledge representation ---
	("knowledge_graph", "https://en.wikipedia.org/wiki/Knowledge_graph", "on"),
	("semantic_web", "https://en.wikipedia.org/wiki/Semantic_Web", "on"),
	("ontology_cs", "https://en.wikipedia.org/wiki/Ontology_(information_science)", "on"),
	("rdf", "https://en.wikipedia.org/wiki/Resource_Description_Framework", "on"),
	("triple_store", "https://en.wikipedia.org/wiki/Triplestore", "on"),
	("knowledge_representation", "https://en.wikipedia.org/wiki/Knowledge_representation_and_reasoning", "on"),
	("graph_database", "https://en.wikipedia.org/wiki/Graph_database", "on"),
	# --- neural networks & GNNs ---
	("neural_network", "https://en.wikipedia.org/wiki/Neural_network_(machine_learning)", "on"),
	("graph_neural_network", "https://en.wikipedia.org/wiki/Graph_neural_network", "on"),
	("transformer_model", "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)", "on"),
	("attention_mechanism", "https://en.wikipedia.org/wiki/Attention_(machine_learning)", "on"),
	("deep_learning", "https://en.wikipedia.org/wiki/Deep_learning", "on"),
	("backpropagation", "https://en.wikipedia.org/wiki/Backpropagation", "on"),
	("recurrent_nn", "https://en.wikipedia.org/wiki/Recurrent_neural_network", "on"),
	# --- embeddings & similarity ---
	("word_embedding", "https://en.wikipedia.org/wiki/Word_embedding", "on"),
	("word2vec", "https://en.wikipedia.org/wiki/Word2vec", "on"),
	("cosine_similarity", "https://en.wikipedia.org/wiki/Cosine_similarity", "on"),
	("vector_space_model", "https://en.wikipedia.org/wiki/Vector_space_model", "on"),
	("dimensionality_reduction", "https://en.wikipedia.org/wiki/Dimensionality_reduction", "on"),
	# --- information retrieval ---
	("information_retrieval", "https://en.wikipedia.org/wiki/Information_retrieval", "on"),
	("semantic_search", "https://en.wikipedia.org/wiki/Semantic_search", "on"),
	("vector_database", "https://en.wikipedia.org/wiki/Vector_database", "on"),
	("tf_idf", "https://en.wikipedia.org/wiki/Tf%E2%80%93idf", "on"),
	("inverted_index", "https://en.wikipedia.org/wiki/Inverted_index", "on"),
	("relevance_feedback", "https://en.wikipedia.org/wiki/Relevance_feedback", "on"),
	# --- probabilistic methods ---
	("markov_chain", "https://en.wikipedia.org/wiki/Markov_chain", "on"),
	("mcmc", "https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo", "on"),
	("bayesian_inference", "https://en.wikipedia.org/wiki/Bayesian_inference", "on"),
	("metropolis_hastings", "https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm", "on"),
	# --- graph algorithms ---
	("graph_theory", "https://en.wikipedia.org/wiki/Graph_theory", "on"),
	("dijkstra", "https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm", "on"),
	("breadth_first_search", "https://en.wikipedia.org/wiki/Breadth-first_search", "on"),
	("pagerank", "https://en.wikipedia.org/wiki/PageRank", "on"),
	# --- systems / infrastructure ---
	("rack_server", "https://en.wikipedia.org/wiki/19-inch_rack", "on"),
	("data_center", "https://en.wikipedia.org/wiki/Data_center", "on"),
	("server_hardware", "https://en.wikipedia.org/wiki/Server_(computing)", "on"),
	("api", "https://en.wikipedia.org/wiki/API", "on"),
	("rest_api", "https://en.wikipedia.org/wiki/REST", "on"),
	("tcp_protocol", "https://en.wikipedia.org/wiki/Transmission_Control_Protocol", "on"),
	("memory_management", "https://en.wikipedia.org/wiki/Memory_management", "on"),
	("cache_computing", "https://en.wikipedia.org/wiki/Cache_(computing)", "on"),
	# --- off-topic noise ---
	("surfing", "https://en.wikipedia.org/wiki/Surfing", "off"),
	("origami", "https://en.wikipedia.org/wiki/Origami", "off"),
	("jazz", "https://en.wikipedia.org/wiki/Jazz", "off"),
	("chess", "https://en.wikipedia.org/wiki/Chess", "off"),
	("pottery", "https://en.wikipedia.org/wiki/Pottery", "off"),
	("ballet", "https://en.wikipedia.org/wiki/Ballet", "off"),
	("beekeeping", "https://en.wikipedia.org/wiki/Beekeeping", "off"),
	("calligraphy", "https://en.wikipedia.org/wiki/Calligraphy", "off"),
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
