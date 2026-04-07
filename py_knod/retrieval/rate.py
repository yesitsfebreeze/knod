"""Retrieval · Rate — final re-ranking pass on thoughts against the query.

After Stage 1 (specialist match) and Stage 2 (thought retrieval + merge),
this module re-scores each thought by combining the merge score with a
direct cosine similarity check against the query. This penalises thoughts
that ranked highly on structural signals (GNN / edges) but are tangential
to the actual question.
"""

import numpy as np

from ..specialist.graph import Thought
from ..util.math import cosine


def rate_thoughts(
	query_emb: np.ndarray,
	scored: list[tuple[Thought, float]],
	merge_weight: float = 0.7,
) -> list[tuple[Thought, float]]:
	"""Re-rank *scored* thoughts by blending merge score with direct cosine similarity.

	Parameters
	----------
	query_emb : ndarray
		Embedded query vector.
	scored : list of (Thought, float)
		Pre-sorted merge results from Stage 2.
	merge_weight : float
		Weight given to the original merge score (1 − merge_weight goes to direct cosine).

	Returns
	-------
	list of (Thought, float)
		Re-ranked list sorted by final rating.
	"""
	if not scored:
		return scored

	direct_weight = 1.0 - merge_weight

	rated: list[tuple[Thought, float]] = []
	for thought, merge_score in scored:
		direct_sim = cosine(thought.embedding, query_emb)
		rating = merge_weight * merge_score + direct_weight * direct_sim
		rated.append((thought, rating))

	rated.sort(key=lambda x: x[1], reverse=True)
	return rated
