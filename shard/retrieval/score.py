"""Retrieval · Score — three parallel scoring signals for a query embedding.

Matches FLOW.md Q_FANOUT:
  Q_S1 — GNN scoring: base MPNN + ShardLayer forward pass
  Q_S2 — Cosine similarity: query vs thought embeddings
  Q_S3 — Edge embedding search: query vs reasoning embeddings ×0.8 dampening
"""

import logging

import numpy as np
import torch

from ..config import Config
from ..shard.graph import Graph
from ..shard.gnn import ShardMPNN, ShardLayer
from ..util.math import cosine
from .merge import SIMILARITY_FLOOR

log = logging.getLogger(__name__)


def cosine_scores(query_emb: np.ndarray, graph: Graph) -> dict[int, float]:
	"""Q_S2: Cosine similarity between query and every thought embedding."""
	scores = {}
	for tid, t in list(graph.thoughts.items()):
		scores[tid] = cosine(query_emb, t.embedding)
	return scores


def edge_scores(query_emb: np.ndarray, graph: Graph, cfg: Config) -> dict[int, float]:
	"""Q_S3: Edge embedding search; each matching edge boosts its endpoint nodes."""
	scores: dict[int, float] = {}
	edge_hits = graph.find_edges(query_emb, k=cfg.top_k * 2, threshold=SIMILARITY_FLOOR)
	for edge, esim in edge_hits:
		for tid in (edge.source_id, edge.target_id):
			if tid in graph.thoughts:
				scores[tid] = max(scores.get(tid, 0.0), esim)
	return scores


def gnn_scores(
	query_emb: np.ndarray,
	graph: Graph,
	model: ShardMPNN,
	shard: ShardLayer,
) -> dict[int, float]:
	"""Q_S1: GNN forward pass relevance scores (base MPNN + ShardLayer)."""
	if graph.num_edges == 0:
		return {}

	model.eval()
	shard.eval()

	node_features, edge_index, edge_features, ordered_ids, valid_edges = graph.to_tensors()

	if not valid_edges:
		return {}

	# Modulate edge features by success_rate: edges with retrieval feedback
	# get a small boost (1.0 to 1.5×), preserving dimensionality.
	success_scales = torch.tensor(
		[1.0 + 0.5 * e.success_rate for e in valid_edges],
		dtype=edge_features.dtype,
	).unsqueeze(-1)
	edge_features = edge_features * success_scales

	try:
		with torch.no_grad():
			hidden, scores_t = model(node_features, edge_index, edge_features)
			hidden, scores_t = shard(hidden, edge_index)
	except Exception:
		log.debug("GNN scoring failed, falling back to cosine only", exc_info=True)
		return {}

	# Normalize GNN scores to [0, 1]
	s = scores_t.squeeze(-1)
	s_min, s_max = s.min(), s.max()
	if s_max - s_min > 1e-6:
		s = (s - s_min) / (s_max - s_min)
	else:
		s = torch.zeros_like(s) + 0.5

	return {tid: float(s[i]) for i, tid in enumerate(ordered_ids)}
