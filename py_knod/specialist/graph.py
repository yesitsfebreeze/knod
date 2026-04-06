"""Specialist graph — Thought, Edge, LimboThought, Graph data structures."""

import time
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Thought:
	id: int
	text: str
	embedding: np.ndarray
	source: str = ""
	created_at: float = field(default_factory=time.time)
	access_count: int = 0
	last_accessed: float = 0.0


@dataclass
class Edge:
	source_id: int
	target_id: int
	weight: float
	reasoning: str
	embedding: np.ndarray
	source: str = ""
	created_at: float = field(default_factory=time.time)


@dataclass
class LimboThought:
	"""Thought rejected by MCMC gate, awaiting cluster analysis."""

	text: str
	embedding: np.ndarray
	source: str = ""
	created_at: float = field(default_factory=time.time)


class Graph:
	def __init__(self, purpose: str = "", max_thoughts: int = 0, max_edges: int = 0):
		self.purpose: str = purpose
		self.thoughts: dict[int, Thought] = {}
		self.edges: list[Edge] = []
		self.descriptors: dict[str, str] = {}
		self._next_id: int = 1
		self._profile: np.ndarray | None = None  # running mean of embeddings
		self._registry_nodes: dict[str, int] = {}  # specialist name → thought id in this graph
		self.limbo: list[LimboThought] = []
		self.max_thoughts: int = max_thoughts  # 0 = unlimited
		self.max_edges: int = max_edges  # 0 = unlimited

	@property
	def num_thoughts(self) -> int:
		return len(self.thoughts)

	@property
	def num_edges(self) -> int:
		return len(self.edges)

	@property
	def maturity(self) -> float:
		return min(self.num_thoughts / 50.0, 1.0)

	@property
	def profile(self) -> np.ndarray | None:
		return self._profile

	def add_thought(self, text: str, embedding: np.ndarray, source: str = "") -> Thought | None:
		if self.max_thoughts > 0 and self.num_thoughts >= self.max_thoughts:
			return None
		tid = self._next_id
		self._next_id += 1
		t = Thought(id=tid, text=text, embedding=embedding, source=source)
		self.thoughts[tid] = t
		self._update_profile(embedding)
		return t

	def add_edge(
		self,
		source_id: int,
		target_id: int,
		weight: float,
		reasoning: str,
		embedding: np.ndarray,
	) -> Edge | None:
		if self.max_edges > 0 and self.num_edges >= self.max_edges:
			return None
		e = Edge(
			source_id=source_id,
			target_id=target_id,
			weight=weight,
			reasoning=reasoning,
			embedding=embedding,
		)
		self.edges.append(e)
		return e

	def find_thoughts(self, embedding: np.ndarray, k: int = 5, threshold: float = 0.0) -> list[tuple[Thought, float]]:
		if not self.thoughts:
			return []
		query = embedding / (np.linalg.norm(embedding) + 1e-10)
		scored = []
		for t in self.thoughts.values():
			t_norm = t.embedding / (np.linalg.norm(t.embedding) + 1e-10)
			sim = float(np.dot(query, t_norm))
			if sim >= threshold:
				scored.append((t, sim))
		scored.sort(key=lambda x: x[1], reverse=True)
		return scored[:k]

	def find_edges(self, embedding: np.ndarray, k: int = 5, threshold: float = 0.0) -> list[tuple[Edge, float]]:
		if not self.edges:
			return []
		query = embedding / (np.linalg.norm(embedding) + 1e-10)
		scored = []
		for e in self.edges:
			e_norm = e.embedding / (np.linalg.norm(e.embedding) + 1e-10)
			sim = float(np.dot(query, e_norm)) * 0.8  # edge embedding dampening ×0.8
			if sim >= threshold:
				scored.append((e, sim))
		scored.sort(key=lambda x: x[1], reverse=True)
		return scored[:k]

	def get_neighbors(self, thought_id: int) -> list[tuple[int, Edge]]:
		result = []
		for e in self.edges:
			if e.source_id == thought_id:
				result.append((e.target_id, e))
			elif e.target_id == thought_id:
				result.append((e.source_id, e))
		return result

	def get_adjacency(self) -> tuple[list[list[int]], list[np.ndarray]]:
		"""Return edge_index [2, E] and edge_attr [E, dim] for PyG."""
		if not self.edges:
			return [[], []], []
		sources, targets, attrs = [], [], []
		for e in self.edges:
			sources.append(e.source_id)
			targets.append(e.target_id)
			attrs.append(e.embedding)
		return [sources, targets], attrs

	def thought_ids_ordered(self) -> list[int]:
		return sorted(self.thoughts.keys())

	def id_to_index(self) -> dict[int, int]:
		return {tid: i for i, tid in enumerate(self.thought_ids_ordered())}

	def _update_profile(self, embedding: np.ndarray):
		if self._profile is None:
			self._profile = embedding.copy()
		else:
			n = self.num_thoughts
			self._profile = self._profile * ((n - 1) / n) + embedding / n

	def apply_edge_decay(self, decay_coefficient: float):
		"""Apply time-based decay to edge weights. Removes edges below min_weight."""
		if decay_coefficient <= 0 or not self.edges:
			return
		now = time.time()
		min_weight = 0.01
		surviving = []
		for e in self.edges:
			age_hours = (now - e.created_at) / 3600.0
			if age_hours > 0:
				e.weight *= (1.0 - decay_coefficient) ** age_hours
			if e.weight >= min_weight:
				surviving.append(e)
		self.edges = surviving
