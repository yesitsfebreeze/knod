"""Shard graph — Thought, Edge, LimboThought, Graph data structures."""

import time
from dataclasses import dataclass, field
import numpy as np
import torch

from ..util.math import cosine


@dataclass
class Thought:
	id: int
	text: str
	embedding: np.ndarray
	source: str = ""
	created_at: float = field(default_factory=time.time)
	access_count: int = 0
	last_accessed: float = 0.0

	def to_dict(self) -> dict:
		return {
			"text": self.text,
			"embedding": self.embedding,
			"source": self.source,
			"created_at": self.created_at,
			"access_count": self.access_count,
			"last_accessed": self.last_accessed,
		}

	@classmethod
	def from_dict(cls, tid: int, d: dict) -> "Thought":
		return cls(
			id=tid,
			text=d["text"],
			embedding=d["embedding"],
			source=d.get("source", ""),
			created_at=d.get("created_at", 0.0),
			access_count=d.get("access_count", 0),
			last_accessed=d.get("last_accessed", 0.0),
		)


@dataclass
class Edge:
	source_id: int
	target_id: int
	weight: float
	reasoning: str
	embedding: np.ndarray
	source: str = ""
	created_at: float = field(default_factory=time.time)
	traversal_count: int = 0
	success_count: int = 0
	last_traversed: float = 0.0

	@property
	def success_rate(self) -> float:
		"""Fraction of traversals that led to a thought used in the final answer."""
		return self.success_count / self.traversal_count if self.traversal_count > 0 else 0.0

	def to_dict(self) -> dict:
		return {
			"source_id": self.source_id,
			"target_id": self.target_id,
			"weight": self.weight,
			"reasoning": self.reasoning,
			"embedding": self.embedding,
			"source": self.source,
			"created_at": self.created_at,
			"traversal_count": self.traversal_count,
			"success_count": self.success_count,
			"last_traversed": self.last_traversed,
		}

	@classmethod
	def from_dict(cls, d: dict) -> "Edge":
		return cls(
			source_id=d["source_id"],
			target_id=d["target_id"],
			weight=d["weight"],
			reasoning=d["reasoning"],
			embedding=d["embedding"],
			source=d.get("source", ""),
			created_at=d.get("created_at", 0.0),
			traversal_count=d.get("traversal_count", 0),
			success_count=d.get("success_count", 0),
			last_traversed=d.get("last_traversed", 0.0),
		)


@dataclass
class LimboThought:
	"""Thought rejected by MCMC gate, awaiting cluster analysis."""

	text: str
	embedding: np.ndarray
	source: str = ""
	created_at: float = field(default_factory=time.time)

	def to_dict(self) -> dict:
		return {
			"text": self.text,
			"embedding": self.embedding,
			"source": self.source,
			"created_at": self.created_at,
		}

	@classmethod
	def from_dict(cls, d: dict) -> "LimboThought":
		return cls(
			text=d["text"],
			embedding=d["embedding"],
			source=d.get("source", ""),
			created_at=d.get("created_at", 0.0),
		)


class Graph:
	def __init__(
		self, name: str = "", purpose: str = "", max_thoughts: int = 0, max_edges: int = 0, maturity_divisor: int = 50
	):
		self.name: str = name
		self.purpose: str = purpose
		self.thoughts: dict[int, Thought] = {}
		self.edges: list[Edge] = []
		self.descriptors: dict[str, str] = {}
		self._next_id: int = 1
		self._profile: np.ndarray | None = None  # running mean of embeddings
		self._registry_nodes: dict[str, int] = {}  # Shard name → thought id in this graph
		self.limbo: list[LimboThought] = []
		self.max_thoughts: int = max_thoughts  # 0 = unlimited
		self.max_edges: int = max_edges  # 0 = unlimited
		self.maturity_divisor: int = maturity_divisor

	@property
	def num_thoughts(self) -> int:
		return len(self.thoughts)

	@property
	def num_edges(self) -> int:
		return len(self.edges)

	@property
	def maturity(self) -> float:
		return min(self.num_thoughts / max(self.maturity_divisor, 1), 1.0)

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
		scored = []
		for t in self.thoughts.values():
			sim = cosine(embedding, t.embedding)
			if sim >= threshold:
				scored.append((t, sim))
		scored.sort(key=lambda x: x[1], reverse=True)
		return scored[:k]

	def find_edges(self, embedding: np.ndarray, k: int = 5, threshold: float = 0.0) -> list[tuple[Edge, float]]:
		if not self.edges:
			return []
		scored = []
		for e in self.edges:
			sim = cosine(embedding, e.embedding) * 0.8  # edge embedding dampening x0.8
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

	def thought_ids_ordered(self) -> list[int]:
		return sorted(self.thoughts.keys())

	def id_to_index(self) -> dict[int, int]:
		return {tid: i for i, tid in enumerate(self.thought_ids_ordered())}

	def to_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], list["Edge"]]:
		"""Build (node_features, edge_index, edge_features, ordered_ids, valid_edges) tensors.

		Returns torch tensors ready for the GNN forward pass.
		valid_edges is the filtered edge list (edges whose endpoints exist in the graph).
		"""
		ordered_ids = self.thought_ids_ordered()
		id_map = self.id_to_index()

		node_features = torch.stack([torch.from_numpy(self.thoughts[tid].embedding) for tid in ordered_ids])

		valid_edges = [e for e in self.edges if e.source_id in id_map and e.target_id in id_map]
		sources = [id_map[e.source_id] for e in valid_edges]
		targets = [id_map[e.target_id] for e in valid_edges]
		edge_index = torch.tensor([sources, targets], dtype=torch.long)
		edge_features = torch.stack([torch.from_numpy(e.embedding) for e in valid_edges])

		return node_features, edge_index, edge_features, ordered_ids, valid_edges

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

	def refine_edges(self, boost: float = 0.02, dampen: float = 0.01, min_traversals: int = 3):
		"""Adjust edge weights based on retrieval feedback.

		Edges with high success_rate get a small weight boost.
		Edges with high traversal but zero success get dampened.
		Only adjusts edges with at least min_traversals to avoid noise.
		"""
		for e in self.edges:
			if e.traversal_count < min_traversals:
				continue
			if e.success_rate > 0.5:
				e.weight = min(e.weight + boost, 1.0)
			elif e.success_rate == 0.0:
				e.weight = max(e.weight - dampen, 0.01)
