"""Strand types — shared dataclasses used across handler and limbo layers.

Moved here from handler.py to break the upward dependency from limbo/promote.py
into the top-level handler module.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .graph import Graph
from .gnn import KnodMPNN, StrandLayer


@dataclass(frozen=True)
class GraphEvent:
	"""Typed event fired after significant state changes."""

	kind: str  # "ingest_complete" | "limbo_promoted" | "status_changed"
	thoughts: int
	edges: int
	committed: int  # number newly committed (0 for non-ingest events)
	detail: dict = field(default_factory=dict)


EventListener = Callable[[GraphEvent], None]


@dataclass
class StrandIndexEntry:
	"""Lightweight metadata for one strand, built on startup."""

	name: str
	purpose: str
	descriptors: dict[str, str]
	profile: np.ndarray | None
	num_thoughts: int
	num_edges: int


class Strand:
	"""One strand = graph + model + strand layer."""

	__slots__ = ("name", "purpose", "graph", "model", "strand")

	def __init__(self, name, purpose, graph, model, strand):
		self.name = name
		self.purpose = purpose
		self.graph = graph
		self.model = model
		self.strand = strand


@dataclass
class IngestResult:
	"""Rich result from a synchronous ingest operation."""

	committed: list = field(default_factory=list)  # list of Thought objects
	rejected: int = 0  # sent to limbo
	deduplicated: int = 0  # merged into existing
