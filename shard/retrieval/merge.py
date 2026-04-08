"""Retrieval · Merge — adaptive weighting, access boost, deduplication.

Matches FLOW.md Q_MERGE:
  Q_WGT — Adaptive weighting (boost-only):
             blended = 0.4·cos + 0.4·gnn + 0.2·edge  (or fewer signals)
             score   = max(cos, blended)
             GNN/edge signals can only boost, never penalise.
  Q_BST — Access boost: +log1p·0.02 freq  +0.05·exp recency
  Q_THR — Adaptive threshold: scale between floor (0.2) and configured max
             based on graph maturity and query-level match quality
  Q_DED — Deduplicate across shards: best score per thought text
"""

import math
import time as _time

from ..config import Config
from ..shard.graph import Graph, Thought

SIMILARITY_FLOOR = 0.2

# Sources excluded from retrieval — LLM-synthesized non-answers ingested via
# the learning flywheel that pollute rankings with low-information content.
_EXCLUDED_SOURCES = {"query_response"}

# Source prefix excluded from retrieval — registry nodes added to the global
# graph to represent shards.  They are structural routing metadata, not
# knowledge content, and should never appear in answers.
_EXCLUDED_PREFIX = "Shard:"


def _is_excluded(source: str) -> bool:
	return source in _EXCLUDED_SOURCES or source.startswith(_EXCLUDED_PREFIX)


def _effective_threshold(cos: dict[int, float], graph: Graph, cfg: Config) -> float:
	"""Scale similarity threshold between FLOOR and cfg value.

	Two factors drive the scaling:
	  1. Graph maturity — young graphs get a lower bar.
	  2. Query match quality — when the best cosine score is weak,
	     the threshold drops proportionally so sparse topics still surface.
	"""
	mat = graph.maturity  # 0..1, reaches 1 at 50 thoughts

	# Best raw cosine score for this query
	best_cos = max(cos.values()) if cos else 0.0

	# Query-level factor: if the best match is below the threshold,
	# lerp toward the floor so *something* can get through.
	# When best_cos >= threshold the factor is 1.0 (no relaxation).
	query_factor = min(best_cos / cfg.similarity_threshold, 1.0) if cfg.similarity_threshold > 0 else 1.0

	# Combined: both factors independently pull the threshold down
	combined = mat * query_factor
	return SIMILARITY_FLOOR + (cfg.similarity_threshold - SIMILARITY_FLOOR) * combined


def merge(
	cos: dict[int, float],
	gnn: dict[int, float],
	edg: dict[int, float],
	graph: Graph,
	cfg: Config,
) -> list[tuple[Thought, float]]:
	"""Combine cosine / GNN / edge signals, apply access boost, filter by threshold.

	Returns up to cfg.top_k * 3 candidates per shard so that deduplicate()
	can make a fair global selection across all shards.  Returning only
	top_k per shard would let a large shard with many mediocre-but-
	passing thoughts crowd out a small shard with a few highly relevant ones.
	"""
	now = _time.time()
	has_gnn = bool(gnn)
	has_edges = bool(edg)

	# Q_THR — adaptive threshold
	threshold = _effective_threshold(cos, graph, cfg)

	combined = []

	for tid in graph.thought_ids_ordered():
		c = cos.get(tid, 0.0)
		g = gnn.get(tid, 0.0)
		e = edg.get(tid, 0.0)

		# Q_WGT — adaptive weighting
		# GNN / edge signals can only *boost* a thought's score, never drag
		# it below cosine.  This prevents newly-ingested thoughts with strong
		# semantic relevance from being penalised by uninformative GNN/edge
		# scores, while still rewarding well-connected thoughts.
		if has_gnn and has_edges:
			blended = 0.4 * c + 0.4 * g + 0.2 * e
		elif has_gnn:
			blended = 0.5 * c + 0.5 * g
		elif has_edges:
			blended = 0.7 * c + 0.3 * e
		else:
			blended = c
		score = max(c, blended)

		# Q_BST — access boost
		t = graph.thoughts[tid]
		freq_boost = math.log1p(t.access_count) * 0.02
		recency_boost = 0.0
		if t.last_accessed > 0:
			age_hours = (now - t.last_accessed) / 3600.0
			recency_boost = 0.05 * math.exp(-age_hours / 24.0)
		score += min(freq_boost + recency_boost, 0.1)

		if score >= threshold:
			if not _is_excluded(t.source):
				combined.append((t, score))

	combined.sort(key=lambda x: x[1], reverse=True)
	# Return a generous candidate set — deduplicate() does the final top_k cut
	return combined[: cfg.top_k * 3]


def deduplicate(scored_lists: list[list[tuple[Thought, float]]], top_k: int) -> list[tuple[Thought, float]]:
	"""Q_DED: Deduplicate across shards, keeping best score per thought text."""
	seen: dict[str, tuple[Thought, float]] = {}
	for scored in scored_lists:
		for thought, score in scored:
			if thought.text not in seen or score > seen[thought.text][1]:
				seen[thought.text] = (thought, score)
	return sorted(seen.values(), key=lambda x: x[1], reverse=True)[:top_k]


def best_chains_from(
	all_chains: list,
	scored: list[tuple[Thought, float]],
) -> list:
	"""Filter chains to those whose terminal thought survived scoring.

	For each surviving terminal thought, keeps only the highest-scoring chain.
	Returns chains sorted by score descending.
	"""
	scored_ids = {t.id for t, _ in scored}
	relevant: list = []
	seen_terminals: set[int] = set()
	for chain in sorted(all_chains, key=lambda c: c.score, reverse=True):
		tid = chain.terminal.id if chain.terminal else None
		if tid and tid in scored_ids and tid not in seen_terminals:
			seen_terminals.add(tid)
			relevant.append(chain)
	return relevant
