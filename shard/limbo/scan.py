"""Limbo scan — pairwise cosine similarity + greedy connected components.

Runs as a background scan every 60 s (configurable via cfg.limbo_scan_interval).
Produces clusters of ≥ cfg.limbo_cluster_min thoughts with cosine sim ≥ cfg.limbo_cluster_threshold.
"""

import logging

import numpy as np

from ..config import Config
from ..strand.graph import Graph

log = logging.getLogger(__name__)


def find_clusters(graph: Graph, cfg: Config) -> list[list[int]]:
	"""Return index lists of limbo thoughts that form clusters large enough to promote.

	Steps (matches FLOW.md LIMBO subgraph):
	  1. Pairwise cosine similarity across all limbo thoughts
	  2. Greedy connected components at threshold = cfg.limbo_cluster_threshold
	  3. Keep only clusters with ≥ cfg.limbo_cluster_min thoughts
	"""
	if len(graph.limbo) < cfg.limbo_cluster_min:
		return []

	embeddings = np.array([lt.embedding for lt in graph.limbo])
	norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
	normed = embeddings / norms
	sim_matrix = normed @ normed.T

	n = len(graph.limbo)
	visited = [False] * n
	clusters = []

	for i in range(n):
		if visited[i]:
			continue
		cluster = [i]
		visited[i] = True
		stack = [i]
		while stack:
			current = stack.pop()
			for j in range(n):
				if not visited[j] and sim_matrix[current, j] >= cfg.limbo_cluster_threshold:
					visited[j] = True
					cluster.append(j)
					stack.append(j)
		if len(cluster) >= cfg.limbo_cluster_min:
			clusters.append(cluster)

	return clusters
