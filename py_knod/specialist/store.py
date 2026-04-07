"""Persistence — save/load graph + model checkpoints + append-only binary log.

.knod unified format:
  [magic: 4 bytes "knod" = 0x6b6e6f64]
  [version: 4 bytes LE i32]  — version 2 = Python format
  [section]*:
    [tag: 1 byte]
    [length: 8 bytes LE i64]
    [payload: <length> bytes]

Section tags:
  0x01 GRAPH    — pickled graph state dict
  0x02 MODEL    — torch.save bytes (model + strand state_dict)
  0x03 LIMBO    — pickled limbo list
  0x04 REGISTRY — pickled registry nodes dict
"""

import io
import logging
import pickle
import struct
import time as _time
from pathlib import Path

import torch

from .graph import Graph, Thought, Edge, LimboThought
from .gnn import KnodMPNN, StrandLayer
from ..util.graph_serde import graph_to_state, graph_from_state

log = logging.getLogger(__name__)

# --- .knod format constants ---
KNOD_MAGIC = 0x6B6E6F64  # "knod"
KNOD_VERSION = 2

SECTION_GRAPH = 0x01
SECTION_MODEL = 0x02
SECTION_LIMBO = 0x03
SECTION_REGISTRY = 0x04

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
		d = thought.to_dict()
		d["id"] = thought.id  # log entries need the id for replay
		payload = pickle.dumps(d)
		self._append(_LOG_THOUGHT, payload)

	def append_edge(self, edge: Edge):
		"""Append an edge entry to the log."""
		payload = pickle.dumps(edge.to_dict())
		self._append(_LOG_EDGE, payload)

	def append_limbo(self, lt: LimboThought):
		"""Append a limbo thought entry to the log."""
		payload = pickle.dumps(lt.to_dict())
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
						graph.thoughts[tid] = Thought.from_dict(tid, data)
						graph._next_id = max(graph._next_id, tid + 1)
				elif entry_type == _LOG_EDGE:
					graph.edges.append(Edge.from_dict(data))
				elif entry_type == _LOG_LIMBO:
					graph.limbo.append(LimboThought.from_dict(data))

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

	state = graph_to_state(graph, include_limbo=True)
	with open(path, "wb") as f:
		pickle.dump(state, f)
	# Compaction: clear the append log since full state is now on disk
	GraphLog(path).clear()


def load_graph(path: str | Path) -> Graph:
	"""Load graph from pickle, then replay any append log entries."""
	with open(path, "rb") as f:
		state = pickle.load(f)

	graph = graph_from_state(state)

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
	"""Save everything to a single base_path.knod file. Also updates shared base."""
	base = Path(base_path)
	save_knod(graph, model, strand, base.with_suffix(".knod"))
	# Update the shared base model
	save_base_model(model)


def load_all(cfg, base_path: str | Path) -> tuple[Graph, KnodMPNN, StrandLayer]:
	"""Load from base_path.knod (preferred) or fall back to legacy .graph+.pt files.

	Falls back to shared base.gnn for the model weights if no per-specialist checkpoint exists.
	"""
	base = Path(base_path)
	knod_path = base.with_suffix(".knod")

	if knod_path.exists():
		return load_knod(cfg, knod_path)

	# Legacy fallback: separate .graph + .pt files
	graph_path = base.with_suffix(".graph")
	if graph_path.exists():
		graph = load_graph(graph_path)
		graph.maturity_divisor = getattr(cfg, "maturity_divisor", 50)
		model = KnodMPNN(cfg)
		strand = StrandLayer(cfg.hidden_dim)
		pt_path = base.with_suffix(".pt")
		if pt_path.exists():
			load_model(model, strand, pt_path)
		else:
			load_base_model(model)
		return graph, model, strand

	raise FileNotFoundError(f"No .knod or .graph file at {base}")


# --- .knod unified format ---


def _write_section(f, tag: int, payload: bytes):
	"""Write a tagged section: [tag:1][length:8][payload]."""
	f.write(struct.pack("<B", tag))
	f.write(struct.pack("<q", len(payload)))
	f.write(payload)


def _parse_knod_sections(data: bytes) -> list[tuple[int, bytes]]:
	"""Validate magic + version, then yield all (tag, payload) sections.

	Raises ValueError if the file header is invalid.
	"""
	if len(data) < 8:
		raise ValueError("File too small for .knod format")

	magic = struct.unpack_from("<i", data, 0)[0]
	if magic != KNOD_MAGIC:
		raise ValueError(f"Bad magic: 0x{magic:08x}")

	version = struct.unpack_from("<i", data, 4)[0]
	if version != KNOD_VERSION:
		raise ValueError(f"Unsupported .knod version {version}")

	sections: list[tuple[int, bytes]] = []
	off = 8
	while off < len(data):
		if off + 9 > len(data):
			break
		tag = data[off]
		off += 1
		length = struct.unpack_from("<q", data, off)[0]
		off += 8
		payload = data[off : off + length]
		off += length
		sections.append((tag, payload))
	return sections


def _model_bytes(model: KnodMPNN, strand: StrandLayer) -> bytes:
	"""Serialize model + strand state_dict to bytes via torch.save."""
	buf = io.BytesIO()
	torch.save({"model": model.state_dict(), "strand": strand.state_dict()}, buf)
	return buf.getvalue()


def save_knod(graph: Graph, model: KnodMPNN, strand: StrandLayer, path: str | Path):
	"""Save graph + model + limbo + registry into a single .knod file."""
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)

	with open(path, "wb") as f:
		# Header
		f.write(struct.pack("<i", KNOD_MAGIC))
		f.write(struct.pack("<i", KNOD_VERSION))

		# GRAPH section
		_write_section(f, SECTION_GRAPH, pickle.dumps(graph_to_state(graph)))

		# MODEL section
		_write_section(f, SECTION_MODEL, _model_bytes(model, strand))

		# LIMBO section (only if non-empty)
		if graph.limbo:
			limbo_data = [lt.to_dict() for lt in graph.limbo]
			_write_section(f, SECTION_LIMBO, pickle.dumps(limbo_data))

		# REGISTRY section (only if non-empty)
		if graph._registry_nodes:
			_write_section(f, SECTION_REGISTRY, pickle.dumps(graph._registry_nodes))

	# Clear legacy append log if present
	GraphLog(path).clear()


def load_knod(cfg, path: str | Path) -> tuple[Graph, KnodMPNN, StrandLayer]:
	"""Load graph + model + limbo from a single .knod file."""
	path = Path(path)

	with open(path, "rb") as f:
		data = f.read()

	# Parse sections — validates magic + version
	sections = _parse_knod_sections(data)

	graph_state = None
	model_bytes = None
	limbo_data = None
	registry_nodes = None

	for tag, payload in sections:
		if tag == SECTION_GRAPH:
			graph_state = pickle.loads(payload)
		elif tag == SECTION_MODEL:
			model_bytes = payload
		elif tag == SECTION_LIMBO:
			limbo_data = pickle.loads(payload)
		elif tag == SECTION_REGISTRY:
			registry_nodes = pickle.loads(payload)
		# else: skip unknown sections

	if graph_state is None:
		raise ValueError(f"No GRAPH section in {path}")

	# Reconstruct graph
	graph = graph_from_state(graph_state, maturity_divisor=getattr(cfg, "maturity_divisor", 50))

	if limbo_data:
		for ldata in limbo_data:
			graph.limbo.append(LimboThought.from_dict(ldata))

	if registry_nodes:
		graph._registry_nodes = registry_nodes

	# Reconstruct model
	model = KnodMPNN(cfg)
	strand = StrandLayer(cfg.hidden_dim)

	if model_bytes:
		buf = io.BytesIO(model_bytes)
		checkpoint = torch.load(buf, weights_only=True)
		model.load_state_dict(checkpoint["model"])
		strand.load_state_dict(checkpoint["strand"])
	else:
		load_base_model(model)

	return graph, model, strand


def read_knod_metadata(path: str | Path) -> dict:
	"""Read only the GRAPH section metadata from a .knod file (no model load).

	Returns a dict with keys: purpose, descriptors, profile, num_thoughts, num_edges.
	Raises ValueError if the file is invalid or has no GRAPH section.
	"""
	path = Path(path)
	with open(path, "rb") as f:
		data = f.read()

	# Parse sections — validates magic + version (fixes missing version check)
	sections = _parse_knod_sections(data)

	for tag, payload in sections:
		if tag == SECTION_GRAPH:
			state = pickle.loads(payload)
			return {
				"name": state.get("name", ""),
				"purpose": state.get("purpose", ""),
				"descriptors": state.get("descriptors", {}),
				"profile": state.get("profile"),
				"num_thoughts": len(state.get("thoughts", {})),
				"num_edges": len(state.get("edges", [])),
			}

	raise ValueError(f"No GRAPH section in {path}")
