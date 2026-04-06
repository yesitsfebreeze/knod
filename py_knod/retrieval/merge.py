"""Retrieval · Merge — adaptive weighting, access boost, deduplication.

Matches FLOW.md Q_MERGE:
  Q_WGT — Adaptive weighting:
             GNN+edges:  0.4·cos + 0.4·gnn + 0.2·edge
             GNN only:   0.5·cos + 0.5·gnn
             Cosine only: cos
  Q_BST — Access boost: +log1p·0.02 freq  +0.05·exp recency
  Q_DED — Deduplicate across specialists: best score per thought text
"""

import math
import time as _time

from ..config import Config
from ..specialist.graph import Graph, Thought


def merge(
	cos: dict[int, float],
	gnn: dict[int, float],
	edg: dict[int, float],
	graph: Graph,
	cfg: Config,
) -> list[tuple[Thought, float]]:
	"""Combine cosine / GNN / edge signals, apply access boost, filter by threshold."""
	now = _time.time()
	has_gnn = bool(gnn)
	has_edges = bool(edg)
	combined = []

	for tid in graph.thought_ids_ordered():
		c = cos.get(tid, 0.0)
		g = gnn.get(tid, 0.0)
		e = edg.get(tid, 0.0)

		# Q_WGT — adaptive weighting
		if has_gnn and has_edges:
			score = 0.4 * c + 0.4 * g + 0.2 * e
		elif has_gnn:
			score = 0.5 * c + 0.5 * g
		elif has_edges:
			score = 0.7 * c + 0.3 * e
		else:
			score = c

		# Q_BST — access boost
		t = graph.thoughts[tid]
		freq_boost = math.log1p(t.access_count) * 0.02
		recency_boost = 0.0
		if t.last_accessed > 0:
			age_hours = (now - t.last_accessed) / 3600.0
			recency_boost = 0.05 * math.exp(-age_hours / 24.0)
		score += min(freq_boost + recency_boost, 0.1)

		if score >= cfg.similarity_threshold:
			combined.append((t, score))

	combined.sort(key=lambda x: x[1], reverse=True)
	return combined[: cfg.top_k]


def deduplicate(scored_lists: list[list[tuple[Thought, float]]], top_k: int) -> list[tuple[Thought, float]]:
	"""Q_DED: Deduplicate across specialists, keeping best score per thought text."""
	seen: dict[str, tuple[Thought, float]] = {}
	for scored in scored_lists:
		for thought, score in scored:
			if thought.text not in seen or score > seen[thought.text][1]:
				seen[thought.text] = (thought, score)
	return sorted(seen.values(), key=lambda x: x[1], reverse=True)[:top_k]
