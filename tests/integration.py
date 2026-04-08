"""Full-loop integration test for shard.

Exercises the complete pipeline via Handler:
  1. Ingest AI-domain articles (embedding API calls)
  2. Verify graph structure (thoughts, edges, maturity)
  3. Verify GNN was trained
  4. Re-ingest same content to test dedup
  5. Ask questions and verify relevant answers
  6. Verify access tracking (feedback loop)
  7. Verify persistence (save, reload, ask again)

Requires a valid OpenAI API key in ~/.config/shard/config or OPENAI_API_KEY env var.

Usage:
  python tests/integration.py [--fresh]

  --fresh  delete persisted graph and re-ingest from scratch
"""

import logging
import os
import sys
import tempfile
import time

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("integration")

from shard.config import Config
from shard.Shard.graph import Graph
from shard.Shard.gnn import ShardMPNN, ShardLayer
from shard.Shard.trainer import GNNTrainer
from shard.Shard.store import save_shard, load_shard
from shard.ingest import Ingester
from shard.handler import Handler

# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

passed = 0
failed = 0


def check(desc, cond):
	global passed, failed
	if cond:
		passed += 1
		print(f"  PASS  {desc}")
	else:
		failed += 1
		print(f"  FAIL  {desc}")


# ---------------------------------------------------------------------------
# Inline corpus — self-contained AI-domain articles
# ---------------------------------------------------------------------------

ARTICLES = {
	"knowledge_graphs": (
		"A knowledge graph is a structured representation of facts where entities "
		"are nodes and relationships are directed edges. Each edge carries a type "
		"label such as 'is-a', 'part-of', or 'related-to'. Knowledge graphs power "
		"question answering, recommendation systems, and semantic search by enabling "
		"multi-hop reasoning across connected facts. Google's Knowledge Graph, "
		"Wikidata, and DBpedia are large-scale examples. Embedding-based methods "
		"like TransE and RotatE learn low-dimensional representations of entities "
		"and relations, allowing link prediction and knowledge completion. Graph "
		"neural networks can operate directly on knowledge graph structure to "
		"propagate information between neighboring nodes."
	),
	"embeddings": (
		"Word embeddings map tokens to dense real-valued vectors in a continuous "
		"space where semantic similarity corresponds to geometric proximity. "
		"Word2Vec introduced skip-gram and CBOW architectures that learn embeddings "
		"from co-occurrence statistics. GloVe combines global matrix factorization "
		"with local context windows. Modern sentence embeddings from models like "
		"BERT and text-embedding-3 produce contextual representations that capture "
		"meaning at the sentence level. Cosine similarity between embedding vectors "
		"is the standard measure for semantic relatedness. Embeddings enable "
		"semantic search, clustering, classification, and retrieval-augmented "
		"generation pipelines."
	),
	"graph_neural_networks": (
		"Graph neural networks operate on graph-structured data by iteratively "
		"aggregating information from neighboring nodes through message passing. "
		"Each GNN layer computes a new representation for every node by combining "
		"its current features with a summary of its neighbors' features. Common "
		"architectures include GCN (graph convolutional network), GAT (graph "
		"attention network), and GraphSAGE. The message-passing framework consists "
		"of three steps: message computation, aggregation, and update. After k "
		"layers, each node's representation captures structural information within "
		"its k-hop neighborhood. GNNs are used for node classification, link "
		"prediction, graph classification, and recommendation systems."
	),
	"retrieval_systems": (
		"Information retrieval systems find relevant documents from a large corpus "
		"given a user query. Classical approaches use TF-IDF and BM25 for lexical "
		"matching based on term frequency and inverse document frequency. Neural "
		"retrieval uses dense embeddings to compute semantic similarity between "
		"queries and documents. Hybrid systems combine both approaches through "
		"reciprocal rank fusion or learned reranking. The retrieval pipeline "
		"typically consists of indexing, candidate generation, scoring, and "
		"reranking stages. Vector databases like Pinecone, Milvus, and FAISS "
		"enable approximate nearest neighbor search over millions of embeddings "
		"with sub-millisecond latency."
	),
	"mcmc_methods": (
		"Markov chain Monte Carlo methods generate samples from probability "
		"distributions that are difficult to sample from directly. The Metropolis-"
		"Hastings algorithm proposes a candidate state and accepts or rejects it "
		"based on an acceptance ratio that ensures detailed balance. Gibbs sampling "
		"is a special case that samples each variable conditionally given all "
		"others. MCMC acceptance gating in knowledge systems uses the Metropolis "
		"criterion to decide whether a new piece of information should be committed "
		"to the graph: at low maturity, almost everything is accepted; at high "
		"maturity, only high-quality additions pass the gate. This controls the "
		"signal-to-noise ratio as the knowledge base grows."
	),
}

QUESTIONS = [
	(
		"How do knowledge graphs represent relationships between entities?",
		["node", "edge", "relation", "entit", "triple", "graph", "link", "direct"],
	),
	(
		"What is the role of embeddings in semantic search?",
		["vector", "embed", "semantic", "similar", "cosine", "represent", "dense"],
	),
	(
		"How do graph neural networks propagate information?",
		["message", "pass", "neighbor", "aggregat", "layer", "propagat", "node"],
	),
	(
		"How does MCMC acceptance gating work in knowledge systems?",
		["accept", "reject", "matur", "metropolis", "probabili", "gate", "commit"],
	),
	(
		"What are the stages of an information retrieval pipeline?",
		["index", "retriev", "rank", "score", "candidate", "query", "search"],
	),
]


# ---------------------------------------------------------------------------
# Test phases
# ---------------------------------------------------------------------------


def setup_handler(cfg: Config, graph_path: str, fresh: bool) -> Handler:
	"""Create a Handler and either load existing graph or start fresh."""
	cfg.graph_path = graph_path
	handler = Handler(cfg)

	if not fresh and os.path.exists(graph_path):
		print("\n=== Loading Persisted Graph ===")
		handler.graph, handler.model, handler.Shard = load_shard(cfg, graph_path)
		handler.trainer = GNNTrainer(handler.model, handler.Shard, cfg)
		handler.ingester = Ingester(handler.graph, handler.provider, cfg)
		g = handler.graph
		print(f"  Loaded: {g.num_thoughts} thoughts, {g.num_edges} edges, maturity={g.maturity:.3f}")
	else:
		print("\n=== Creating New Graph ===")
		handler.graph = Graph(
			max_thoughts=cfg.max_thoughts,
			max_edges=cfg.max_edges,
			maturity_divisor=cfg.maturity_divisor,
		)
		handler.model = ShardMPNN(cfg)
		handler.Shard = ShardLayer(cfg.hidden_dim)
		handler.trainer = GNNTrainer(handler.model, handler.Shard, cfg)
		handler.ingester = Ingester(handler.graph, handler.provider, cfg)

	return handler


def test_ingest(handler: Handler) -> None:
	"""Ingest all articles and verify basic graph growth."""
	print("\n=== Ingest ===")
	for i, (name, text) in enumerate(ARTICLES.items()):
		t0 = time.time()
		r = handler.ingest_sync(text, source=name)
		dt = time.time() - t0
		print(f"  [{i + 1}/{len(ARTICLES)}] {name}: {r['committed']} committed, {r['edges']} edges ({dt:.1f}s)")

	g = handler.graph
	print(f"\n  Total: {g.num_thoughts} thoughts, {g.num_edges} edges, maturity={g.maturity:.3f}")


def test_graph_structure(handler: Handler) -> None:
	"""Verify the graph has meaningful structure after ingestion."""
	print("\n=== Graph Structure ===")
	g = handler.graph
	check(f"thoughts > 10 ({g.num_thoughts})", g.num_thoughts > 10)
	check(f"edges > 3 ({g.num_edges})", g.num_edges > 3)
	check(f"maturity > 0 ({g.maturity:.3f})", g.maturity > 0)
	check(f"maturity <= 1.0", g.maturity <= 1.0)

	sources = set(t.source for t in g.thoughts.values())
	check(f"multiple sources tracked ({len(sources)})", len(sources) >= 3)


def test_gnn_trained(handler: Handler) -> None:
	"""Verify the GNN model and Shard layer exist and were trained."""
	print("\n=== GNN Training ===")
	check("model exists", handler.model is not None)
	check("Shard exists", handler.Shard is not None)

	# Train explicitly and verify loss is finite
	if handler.graph.num_edges > 0:
		loss = handler.trainer.train_on_graph(handler.graph)
		check(f"training loss is finite ({loss:.4f})", loss == loss and loss < float("inf"))


def test_dedup(handler: Handler) -> None:
	"""Re-ingest the same content and verify dedup prevents explosion."""
	print("\n=== Dedup (re-ingest) ===")
	before = handler.graph.num_thoughts
	first_article = list(ARTICLES.values())[0]
	first_name = list(ARTICLES.keys())[0]
	r = handler.ingest_sync(first_article, source=first_name)
	after = handler.graph.num_thoughts
	increase = after - before
	check(
		f"re-ingest growth is small (before={before}, after={after}, +{increase})",
		increase < max(before // 2, 5),
	)
	check(f"dedup count > 0 ({r['deduplicated']})", r["deduplicated"] > 0)


def test_ask(handler: Handler) -> None:
	"""Ask questions and verify answers are relevant to the ingested content."""
	print("\n=== Ask Questions ===")
	for q, hint_words in QUESTIONS:
		answer_text, srcs = handler.ask(q)
		has_answer = len(answer_text) > 10
		answer_lower = answer_text.lower()
		relevant = any(w in answer_lower for w in hint_words)

		print(f"\n  Q: {q}")
		print(f"  A: {answer_text[:200]}{'...' if len(answer_text) > 200 else ''}")
		check(f"got answer ({len(answer_text)} chars)", has_answer)
		check(f"answer relevant", relevant)
		if not relevant:
			print(f"    Expected one of: {hint_words}")
		check(f"sources returned ({len(srcs)})", len(srcs) > 0)


def test_access_tracking(handler: Handler) -> None:
	"""Verify that retrieval updates access counters on thoughts."""
	print("\n=== Access Tracking ===")
	g = handler.graph
	accessed = [t for t in g.thoughts.values() if t.access_count > 0]
	check(f"accessed thoughts tracked ({len(accessed)})", len(accessed) > 0)
	if accessed:
		max_access = max(t.access_count for t in accessed)
		check(f"max access_count = {max_access}", max_access >= 1)
		check("last_accessed is recent", accessed[0].last_accessed > time.time() - 300)


def test_persistence(handler: Handler, cfg: Config, graph_path: str) -> None:
	"""Save graph, reload into a fresh handler, and verify data survived."""
	print("\n=== Persistence ===")

	# Save
	save_shard(handler.graph, handler.model, handler.Shard, graph_path)
	check("save_shard succeeded", os.path.exists(graph_path))

	# Reload
	g2, m2, s2 = load_shard(cfg, graph_path)
	check(f"thoughts preserved ({g2.num_thoughts})", g2.num_thoughts == handler.graph.num_thoughts)
	check(f"edges preserved ({g2.num_edges})", g2.num_edges == handler.graph.num_edges)
	check(f"limbo preserved ({len(g2.limbo)})", len(g2.limbo) == len(handler.graph.limbo))

	# Ask a question on the reloaded graph to verify it's functional
	handler2 = Handler(cfg)
	handler2.graph = g2
	handler2.model = m2
	handler2.Shard = s2
	handler2.trainer = GNNTrainer(m2, s2, cfg)
	handler2.ingester = Ingester(g2, handler2.provider, cfg)

	answer_text, srcs = handler2.ask("How do knowledge graphs work?")
	check(f"post-reload answer ({len(answer_text)} chars)", len(answer_text) > 10)
	check(f"post-reload sources ({len(srcs)})", len(srcs) > 0)


def test_edge_samples(handler: Handler) -> None:
	"""Print a sample of edges for manual inspection."""
	print("\n=== Edge Samples ===")
	import random

	g = handler.graph
	if g.num_edges == 0:
		print("  (no edges)")
		return
	sample = random.sample(g.edges, min(5, len(g.edges)))
	for e in sample:
		s = g.thoughts.get(e.source_id)
		t = g.thoughts.get(e.target_id)
		if s and t:
			print(f"  [{s.text[:45]}] -> [{t.text[:45]}] w={e.weight:.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
	fresh = "--fresh" in sys.argv

	cfg = Config.load()
	if not cfg.api_key:
		print("ERROR: no API key found. Set OPENAI_API_KEY or add api_key to ~/.config/shard/config")
		sys.exit(1)

	with tempfile.TemporaryDirectory(prefix="shard_integ_") as tmpdir:
		graph_path = os.path.join(tmpdir, "integration.shard")

		handler = setup_handler(cfg, graph_path, fresh=True)

		# Phase 1: Ingest
		test_ingest(handler)

		# Phase 2: Verify structure
		test_graph_structure(handler)

		# Phase 3: GNN training
		test_gnn_trained(handler)

		# Phase 4: Dedup
		test_dedup(handler)

		# Phase 5: Ask questions
		test_ask(handler)

		# Phase 6: Access tracking
		test_access_tracking(handler)

		# Phase 7: Edge samples (informational)
		test_edge_samples(handler)

		# Phase 8: Persistence
		test_persistence(handler, cfg, graph_path)

	total = passed + failed
	print(f"\n{'=' * 60}")
	print(f"RESULTS: {passed}/{total} passed, {failed} failed")
	print(f"{'=' * 60}")

	if failed > 0:
		sys.exit(1)


if __name__ == "__main__":
	main()
