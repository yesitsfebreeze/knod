"""Dedup — merge near-duplicate thoughts into existing graph nodes.

Runs after Phase 2 (Snapshot). For each prepared thought, if its closest
existing neighbor exceeds dedup_threshold, merge the embedding into the
existing thought (running average) and remove it from the article so
Phases 3 and 4 skip it entirely.
"""

import logging

import numpy as np

from ..config import Config
from ..specialist.graph import Graph
from .prepare import PreparedArticle

log = logging.getLogger(__name__)


def dedup(article: PreparedArticle, graph: Graph, cfg: Config) -> int:
	"""Remove near-duplicate thoughts from article, merging embeddings into existing thoughts.

	Returns the number of thoughts merged (removed from the article).
	"""
	if cfg.dedup_threshold <= 0 or not graph.thoughts:
		return 0

	merged = 0
	keep = []

	for pt in article.thoughts:
		# Find best match among existing graph thoughts
		best_sim = 0.0
		best_thought = None

		query = pt.embedding / (np.linalg.norm(pt.embedding) + 1e-10)
		for t in graph.thoughts.values():
			t_norm = t.embedding / (np.linalg.norm(t.embedding) + 1e-10)
			sim = float(np.dot(query, t_norm))
			if sim > best_sim:
				best_sim = sim
				best_thought = t

		if best_sim >= cfg.dedup_threshold and best_thought is not None:
			# Merge: running average of embeddings
			best_thought.embedding = (best_thought.embedding + pt.embedding) / 2.0
			best_thought.embedding /= np.linalg.norm(best_thought.embedding) + 1e-10
			merged += 1
			log.debug("Merged (%.3f): %s", best_sim, pt.text[:60])
		else:
			keep.append(pt)

	article.thoughts = keep

	if merged:
		log.info("Dedup: merged %d/%d thoughts (threshold=%.2f)", merged, merged + len(keep), cfg.dedup_threshold)
	return merged
