"""Unit tests for retrieval/expand.py — graph traversal expansion.

Tests cover:
  - Direct neighbours of seeds are discovered (depth=1)
  - Expansion respects traversal_depth limit
  - Expansion respects traversal_fan_out limit
  - Seeds are not duplicated in output
  - Disconnected thoughts are not included
  - Score assigned to expanded thoughts is > 0
  - Thoughts already in seeds are not re-added
"""

import numpy as np
import pytest

from shard.config import Config
from shard.shard.graph import Graph
from shard.retrieval.expand import expand


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 8  # tiny embedding dimension for tests


def make_emb(*values) -> np.ndarray:
	"""Return a normalised float32 ndarray from a list of values, padded to DIM."""
	v = list(values) + [0.0] * (DIM - len(values))
	arr = np.array(v[:DIM], dtype=np.float32)
	norm = np.linalg.norm(arr)
	return arr / (norm + 1e-10)


def add_thought(graph: Graph, text: str, emb: np.ndarray | None = None) -> int:
	if emb is None:
		emb = np.random.randn(DIM).astype(np.float32)
		emb /= np.linalg.norm(emb) + 1e-10
	t = graph.add_thought(text, emb)
	return t.id


def add_edge(graph: Graph, src: int, tgt: int, weight: float = 0.8, emb: np.ndarray | None = None):
	if emb is None:
		emb = np.random.randn(DIM).astype(np.float32)
		emb /= np.linalg.norm(emb) + 1e-10
	graph.add_edge(src, tgt, weight=weight, reasoning="test", embedding=emb)


def default_cfg(**overrides) -> Config:
	cfg = Config()
	cfg.traversal_depth = overrides.get("traversal_depth", 2)
	cfg.traversal_fan_out = overrides.get("traversal_fan_out", 20)
	cfg.similarity_threshold = 0.0  # no filtering in unit tests
	cfg.top_k = 5
	return cfg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExpandDirectNeighbours:
	"""Depth-1: direct neighbours of seeds are discovered."""

	def test_single_seed_one_neighbour(self):
		g = Graph()
		seed_emb = make_emb(1, 0, 0)
		neighbour_emb = make_emb(0.9, 0.1, 0)
		query_emb = make_emb(1, 0, 0)

		s_id = add_thought(g, "knowledge graphs store entities as nodes", seed_emb)
		n_id = add_thought(g, "edges encode relationships between entities", neighbour_emb)
		add_edge(g, s_id, n_id, weight=0.9)

		seed_thought = g.thoughts[s_id]
		seeds: list[tuple] = [(seed_thought, 0.95)]
		cfg = default_cfg()

		result, _chains = expand(seeds, query_emb, g, cfg)

		texts = {t.text for t, _ in result}
		assert "knowledge graphs store entities as nodes" in texts, "seed should be in result"
		assert "edges encode relationships between entities" in texts, "direct neighbour should be discovered"

	def test_neighbour_score_positive(self):
		g = Graph()
		query_emb = make_emb(1, 0, 0)
		s_id = add_thought(g, "embeddings map words to dense vectors", make_emb(1, 0, 0))
		n_id = add_thought(g, "cosine similarity measures vector alignment", make_emb(0.8, 0.2, 0))
		# Use an edge embedding correlated with the query so edge_cos > 0
		edge_emb = make_emb(1, 0, 0)
		add_edge(g, s_id, n_id, weight=0.7, emb=edge_emb)

		seeds = [(g.thoughts[s_id], 0.9)]
		result, _chains = expand(seeds, query_emb, g, default_cfg())

		neighbour_entry = next((t, s) for t, s in result if t.text == "cosine similarity measures vector alignment")
		assert neighbour_entry[1] > 0.0, "discovered neighbour should have positive score"

	def test_disconnected_thought_not_included(self):
		g = Graph()
		query_emb = make_emb(1, 0, 0)
		s_id = add_thought(g, "GNNs propagate messages along edges", make_emb(1, 0, 0))
		_ = add_thought(g, "unrelated thought about cooking", make_emb(0, 1, 0))  # no edge to seed

		seeds = [(g.thoughts[s_id], 0.9)]
		result, _chains = expand(seeds, query_emb, g, default_cfg())

		texts = {t.text for t, _ in result}
		assert "unrelated thought about cooking" not in texts, "disconnected thought must not appear"


class TestExpandDepthLimit:
	"""Traversal depth is respected."""

	CHAIN_LABELS = [
		"transformer architecture uses self-attention",
		"attention computes query-key-value products",
		"softmax normalizes attention weights",
		"multi-head attention splits into subspaces",
	]

	def _build_chain(self, length: int):
		"""Build a chain: t0 -> t1 -> t2 -> ... -> t_{length}."""
		g = Graph()
		ids = [add_thought(g, self.CHAIN_LABELS[i], make_emb(1, i * 0.01, 0)) for i in range(length + 1)]
		for i in range(length):
			add_edge(g, ids[i], ids[i + 1], weight=0.8)
		return g, ids

	def test_depth_1_stops_at_first_hop(self):
		g, ids = self._build_chain(3)
		query_emb = make_emb(1, 0, 0)
		seeds = [(g.thoughts[ids[0]], 0.9)]
		cfg = default_cfg(traversal_depth=1)

		result, _chains = expand(seeds, query_emb, g, cfg)
		texts = {t.text for t, _ in result}

		assert self.CHAIN_LABELS[0] in texts, "seed always present"
		assert self.CHAIN_LABELS[1] in texts, "1 hop from seed"
		assert self.CHAIN_LABELS[2] not in texts, "2 hops away, beyond depth=1"
		assert self.CHAIN_LABELS[3] not in texts, "3 hops away, beyond depth=1"

	def test_depth_2_reaches_two_hops(self):
		g, ids = self._build_chain(3)
		query_emb = make_emb(1, 0, 0)
		seeds = [(g.thoughts[ids[0]], 0.9)]
		cfg = default_cfg(traversal_depth=2)

		result, _chains = expand(seeds, query_emb, g, cfg)
		texts = {t.text for t, _ in result}

		assert self.CHAIN_LABELS[1] in texts
		assert self.CHAIN_LABELS[2] in texts
		assert self.CHAIN_LABELS[3] not in texts


class TestExpandFanOutLimit:
	"""traversal_fan_out caps the total number of newly added thoughts."""

	HUB = "MCMC acceptance gating filters low-quality thoughts"
	SPOKES = [
		"Metropolis-Hastings samples from posterior",
		"rejection sampling discards unlikely proposals",
		"Gibbs sampling iterates over conditionals",
		"Hamiltonian Monte Carlo uses gradient information",
		"simulated annealing explores energy landscapes",
		"particle filters approximate sequential posteriors",
		"importance sampling reweights proposal draws",
		"slice sampling adapts step size automatically",
		"parallel tempering exchanges between chains",
		"reversible jump MCMC handles variable dimensions",
	]

	def test_fan_out_cap_respected(self):
		g = Graph()
		query_emb = make_emb(1, 0, 0)
		s_id = add_thought(g, self.HUB, make_emb(1, 0, 0))
		for i, label in enumerate(self.SPOKES):
			n_id = add_thought(g, label, make_emb(0.9, i * 0.01, 0))
			add_edge(g, s_id, n_id, weight=0.5)

		seeds = [(g.thoughts[s_id], 0.9)]
		cfg = default_cfg(traversal_fan_out=3)

		result, _chains = expand(seeds, query_emb, g, cfg)
		# seeds = 1, expanded <= fan_out = 3 -> total <= 4
		assert len(result) <= 1 + 3, f"fan_out=3 means at most 4 total, got {len(result)}"
		texts = {t.text for t, _ in result}
		assert self.HUB in texts

	def test_fan_out_zero_returns_seeds_only(self):
		g = Graph()
		query_emb = make_emb(1, 0, 0)
		s_id = add_thought(g, "vector databases index high-dimensional embeddings", make_emb(1, 0, 0))
		n_id = add_thought(g, "approximate nearest neighbor enables fast retrieval", make_emb(0.8, 0, 0))
		add_edge(g, s_id, n_id, weight=0.9)

		seeds = [(g.thoughts[s_id], 0.9)]
		cfg = default_cfg(traversal_fan_out=0)

		result, _chains = expand(seeds, query_emb, g, cfg)
		texts = {t.text for t, _ in result}
		assert "approximate nearest neighbor enables fast retrieval" not in texts
		assert "vector databases index high-dimensional embeddings" in texts


class TestExpandNoduplication:
	"""Seeds are not duplicated even if they appear as neighbours of each other."""

	THOUGHT_A = "graph neural networks aggregate neighbor features"
	THOUGHT_B = "message passing computes node representations"

	def test_mutual_edge_no_duplicate_seed(self):
		g = Graph()
		query_emb = make_emb(1, 0, 0)
		a_id = add_thought(g, self.THOUGHT_A, make_emb(1, 0, 0))
		b_id = add_thought(g, self.THOUGHT_B, make_emb(0.9, 0.1, 0))
		add_edge(g, a_id, b_id, weight=0.8)
		add_edge(g, b_id, a_id, weight=0.8)

		seeds = [(g.thoughts[a_id], 0.9), (g.thoughts[b_id], 0.8)]
		result, _chains = expand(seeds, query_emb, g, default_cfg())

		texts = [t.text for t, _ in result]
		assert texts.count(self.THOUGHT_A) == 1, "A must appear exactly once"
		assert texts.count(self.THOUGHT_B) == 1, "B must appear exactly once"

	def test_seed_score_preserved(self):
		"""Original seed scores are not downgraded by expand()."""
		g = Graph()
		query_emb = make_emb(1, 0, 0)
		s_id = add_thought(g, "semantic search ranks by meaning not keywords", make_emb(1, 0, 0))
		n_id = add_thought(g, "BM25 uses term frequency for lexical matching", make_emb(0.5, 0.5, 0))
		add_edge(g, s_id, n_id, weight=0.5)

		original_score = 0.95
		seeds = [(g.thoughts[s_id], original_score)]
		result, _chains = expand(seeds, query_emb, g, default_cfg())

		seed_entry = next((t, s) for t, s in result if t.text == "semantic search ranks by meaning not keywords")
		assert seed_entry[1] == pytest.approx(original_score), "seed score should be unchanged"


class TestExpandEmptyGraph:
	"""Graceful handling of degenerate inputs."""

	def test_empty_seeds_returns_empty(self):
		g = Graph()
		query_emb = make_emb(1, 0, 0)
		result, chains = expand([], query_emb, g, default_cfg())
		assert result == []
		assert chains == []

	def test_no_edges_returns_seeds_unchanged(self):
		g = Graph()
		query_emb = make_emb(1, 0, 0)
		s_id = add_thought(g, "isolated thought with no connections", make_emb(1, 0, 0))
		seeds = [(g.thoughts[s_id], 0.9)]
		result, _chains = expand(seeds, query_emb, g, default_cfg())
		assert len(result) == 1
		assert result[0][0].text == "isolated thought with no connections"
