"""Persistence — save/load graph + model checkpoints + append-only binary log."""

import pickle
import struct
import time as _time
from pathlib import Path

import torch

from .graph import Graph, Thought, Edge, LimboThought
from .gnn import KnodMPNN, StrandLayer

# --- Binary log entry types ---
_LOG_THOUGHT = 1
_LOG_EDGE = 2
_LOG_LIMBO = 3
_LOG_MAGIC = b"KNODLOG1"


class GraphLog:
	"""Append-only binary log for streaming graph mutations.

	Each entry: [magic(8)] [type(1)] [length(4)] [pickle payload(length)]
	compact() rewrites the full graph state as a single pickle (the .graph file).
	"""

	def __init__(self, path: str | Path):
		self._path = Path(path).with_suffix(".graphlog")

	def append_thought(self, thought: Thought):
		"""Append a thought entry to the log."""
		payload = pickle.dumps(
			{
				"id": thought.id,
				"text": thought.text,
				"embedding": thought.embedding,
				"source": thought.source,
				"created_at": thought.created_at,
				"access_count": thought.access_count,
				"last_accessed": thought.last_accessed,
			}
		)
		self._append(_LOG_THOUGHT, payload)

	def append_edge(self, edge: Edge):
		"""Append an edge entry to the log."""
		payload = pickle.dumps(
			{
				"source_id": edge.source_id,
				"target_id": edge.target_id,
				"weight": edge.weight,
				"reasoning": edge.reasoning,
				"embedding": edge.embedding,
				"created_at": edge.created_at,
			}
		)
		self._append(_LOG_EDGE, payload)

	def append_limbo(self, lt: LimboThought):
		"""Append a limbo thought entry to the log."""
		payload = pickle.dumps(
			{
				"text": lt.text,
				"embedding": lt.embedding,
				"source": lt.source,
				"created_at": lt.created_at,
			}
		)
		self._append(_LOG_LIMBO, payload)

	def _append(self, entry_type: int, payload: bytes):
		self._path.parent.mkdir(parents=True, exist_ok=True)
		with open(self._path, "ab") as f:
			f.write(_LOG_MAGIC)
			f.write(struct.pack(">BI", entry_type, len(payload)))
			f.write(payload)

	def replay(self, graph: Graph):
		"""Replay log entries into an existing graph (for recovery after crash)."""
		if not self._path.exists():
			return
		with open(self._path, "rb") as f:
			while True:
				magic = f.read(8)
				if len(magic) < 8:
					break
				if magic != _LOG_MAGIC:
					break  # corrupted entry, stop
				header = f.read(5)
				if len(header) < 5:
					break
				entry_type, length = struct.unpack(">BI", header)
				payload = f.read(length)
				if len(payload) < length:
					break
				data = pickle.loads(payload)
				if entry_type == _LOG_THOUGHT:
					tid = data["id"]
					if tid not in graph.thoughts:
						graph.thoughts[tid] = Thought(
							id=tid,
							text=data["text"],
							embedding=data["embedding"],
							source=data.get("source", ""),
							created_at=data.get("created_at", 0),
							access_count=data.get("access_count", 0),
							last_accessed=data.get("last_accessed", 0.0),
						)
						graph._next_id = max(graph._next_id, tid + 1)
				elif entry_type == _LOG_EDGE:
					graph.edges.append(
						Edge(
							source_id=data["source_id"],
							target_id=data["target_id"],
							weight=data["weight"],
							reasoning=data["reasoning"],
							embedding=data["embedding"],
							created_at=data.get("created_at", 0),
						)
					)
				elif entry_type == _LOG_LIMBO:
					graph.limbo.append(
						LimboThought(
							text=data["text"],
							embedding=data["embedding"],
							source=data.get("source", ""),
							created_at=data.get("created_at", 0),
						)
					)

	def clear(self):
		"""Remove the log file (called after compaction)."""
		if self._path.exists():
			self._path.unlink()

	@property
	def exists(self) -> bool:
		return self._path.exists()


def save_graph(graph: Graph, path: str | Path):
	"""Save full graph state as pickle (compaction). Clears the append log."""
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)

	state = {
		"purpose": graph.purpose,
		"descriptors": graph.descriptors,
		"next_id": graph._next_id,
		"profile": graph._profile,
		"max_thoughts": graph.max_thoughts,
		"max_edges": graph.max_edges,
		"thoughts": {
			tid: {
				"text": t.text,
				"embedding": t.embedding,
				"source": t.source,
				"created_at": t.created_at,
				"access_count": t.access_count,
				"last_accessed": t.last_accessed,
			}
			for tid, t in graph.thoughts.items()
		},
		"edges": [
			{
				"source_id": e.source_id,
				"target_id": e.target_id,
				"weight": e.weight,
				"reasoning": e.reasoning,
				"embedding": e.embedding,
				"created_at": e.created_at,
			}
			for e in graph.edges
		],
		"limbo": [
			{
				"text": lt.text,
				"embedding": lt.embedding,
				"source": lt.source,
				"created_at": lt.created_at,
			}
			for lt in graph.limbo
		],
	}
	with open(path, "wb") as f:
		pickle.dump(state, f)
	# Compaction: clear the append log since full state is now on disk
	GraphLog(path).clear()


def load_graph(path: str | Path) -> Graph:
	"""Load graph from pickle, then replay any append log entries."""
	with open(path, "rb") as f:
		state = pickle.load(f)

	graph = Graph(purpose=state["purpose"])
	graph.descriptors = state.get("descriptors", {})
	graph._next_id = state["next_id"]
	graph._profile = state.get("profile")
	graph.max_thoughts = state.get("max_thoughts", 0)
	graph.max_edges = state.get("max_edges", 0)

	for tid_str, tdata in state["thoughts"].items():
		tid = int(tid_str) if isinstance(tid_str, str) else tid_str
		graph.thoughts[tid] = Thought(
			id=tid,
			text=tdata["text"],
			embedding=tdata["embedding"],
			source=tdata.get("source", ""),
			created_at=tdata.get("created_at", 0),
			access_count=tdata.get("access_count", 0),
			last_accessed=tdata.get("last_accessed", 0.0),
		)

	for edata in state["edges"]:
		graph.edges.append(
			Edge(
				source_id=edata["source_id"],
				target_id=edata["target_id"],
				weight=edata["weight"],
				reasoning=edata["reasoning"],
				embedding=edata["embedding"],
				created_at=edata.get("created_at", 0),
			)
		)

	for ldata in state.get("limbo", []):
		graph.limbo.append(
			LimboThought(
				text=ldata["text"],
				embedding=ldata["embedding"],
				source=ldata.get("source", ""),
				created_at=ldata.get("created_at", 0),
			)
		)

	# Replay any append log entries written since last compaction
	glog = GraphLog(path)
	if glog.exists:
		glog.replay(graph)

	return graph


def save_model(model: KnodMPNN, strand: StrandLayer, path: str | Path):
	"""Save model + strand checkpoint (per-specialist .pt file)."""
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(
		{
			"model": model.state_dict(),
			"strand": strand.state_dict(),
		},
		path,
	)


def load_model(model: KnodMPNN, strand: StrandLayer, path: str | Path):
	"""Load model + strand checkpoint."""
	checkpoint = torch.load(path, weights_only=True)
	model.load_state_dict(checkpoint["model"])
	strand.load_state_dict(checkpoint["strand"])


# ---- Shared base GNN checkpoint ----

_BASE_GNN_PATH = Path.home() / ".config" / "knod" / "base.gnn"


def save_base_model(model: KnodMPNN):
	"""Save the shared base MPNN to ~/.config/knod/base.gnn."""
	_BASE_GNN_PATH.parent.mkdir(parents=True, exist_ok=True)
	torch.save({"model": model.state_dict()}, _BASE_GNN_PATH)


def load_base_model(model: KnodMPNN) -> bool:
	"""Load the shared base MPNN from ~/.config/knod/base.gnn. Returns True if loaded."""
	if not _BASE_GNN_PATH.exists():
		return False
	checkpoint = torch.load(_BASE_GNN_PATH, weights_only=True)
	model.load_state_dict(checkpoint["model"])
	return True


def save_strand(strand: StrandLayer, path: str | Path):
	"""Save only the strand layer for a specialist."""
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	torch.save({"strand": strand.state_dict()}, path)


def load_strand(strand: StrandLayer, path: str | Path) -> bool:
	"""Load a strand layer from checkpoint. Returns True if loaded."""
	path = Path(path)
	if not path.exists():
		return False
	checkpoint = torch.load(path, weights_only=True)
	strand.load_state_dict(checkpoint["strand"])
	return True


def save_all(graph: Graph, model: KnodMPNN, strand: StrandLayer, base_path: str | Path):
	"""Save graph + model to base_path.graph and base_path.pt. Also updates shared base."""
	base = Path(base_path)
	save_graph(graph, base.with_suffix(".graph"))
	save_model(model, strand, base.with_suffix(".pt"))
	# Update the shared base model
	save_base_model(model)


def load_all(cfg, base_path: str | Path) -> tuple[Graph, KnodMPNN, StrandLayer]:
	"""Load graph + model from base_path.graph and base_path.pt.

	Falls back to shared base.gnn for the model weights if no per-specialist .pt exists.
	"""
	base = Path(base_path)

	graph = load_graph(base.with_suffix(".graph"))

	model = KnodMPNN(cfg)
	strand = StrandLayer(cfg.hidden_dim)

	pt_path = base.with_suffix(".pt")
	if pt_path.exists():
		load_model(model, strand, pt_path)
	else:
		# Try shared base model
		load_base_model(model)

	return graph, model, strand
