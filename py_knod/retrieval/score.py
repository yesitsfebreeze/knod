"""Retrieval · Score — three parallel scoring signals for a query embedding.

Matches FLOW.md Q_FANOUT:
  Q_S1 — GNN scoring: base MPNN + StrandLayer forward pass
  Q_S2 — Cosine similarity: query vs thought embeddings
  Q_S3 — Edge embedding search: query vs reasoning embeddings ×0.8 dampening
"""

import logging

import numpy as np
import torch

from ..config import Config
from ..specialist.graph import Graph
from ..specialist.gnn import KnodMPNN, StrandLayer

log = logging.getLogger(__name__)


def cosine_scores(query_emb: np.ndarray, graph: Graph) -> dict[int, float]:
	"""Q_S2: Cosine similarity between query and every thought embedding."""
	query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
	scores = {}
	for tid, t in graph.thoughts.items():
		t_norm = t.embedding / (np.linalg.norm(t.embedding) + 1e-10)
		scores[tid] = float(np.dot(query_norm, t_norm))
	return scores


def edge_scores(query_emb: np.ndarray, graph: Graph, cfg: Config) -> dict[int, float]:
	"""Q_S3: Edge embedding search; each matching edge boosts its endpoint nodes."""
	scores: dict[int, float] = {}
	edge_hits = graph.find_edges(query_emb, k=cfg.top_k * 2, threshold=cfg.similarity_threshold)
	for edge, esim in edge_hits:
		for tid in (edge.source_id, edge.target_id):
			if tid in graph.thoughts:
				scores[tid] = max(scores.get(tid, 0.0), esim)
	return scores


def gnn_scores(
	query_emb: np.ndarray,
	graph: Graph,
	model: KnodMPNN,
	strand: StrandLayer,
) -> dict[int, float]:
	"""Q_S1: GNN forward pass relevance scores (base MPNN + StrandLayer)."""
	if graph.num_edges == 0:
		return {}

	ordered_ids = graph.thought_ids_ordered()
	id_map = graph.id_to_index()

	model.eval()
	strand.eval()

	node_features = torch.stack([torch.from_numpy(graph.thoughts[tid].embedding) for tid in ordered_ids])

	valid_edges = [e for e in graph.edges if e.source_id in id_map and e.target_id in id_map]
	if not valid_edges:
		return {}

	sources = [id_map[e.source_id] for e in valid_edges]
	targets = [id_map[e.target_id] for e in valid_edges]
	edge_index = torch.tensor([sources, targets], dtype=torch.long)
	edge_features = torch.stack([torch.from_numpy(e.embedding) for e in valid_edges])

	try:
		with torch.no_grad():
			hidden, scores_t = model(node_features, edge_index, edge_features)
			hidden, scores_t = strand(hidden, edge_index)
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
