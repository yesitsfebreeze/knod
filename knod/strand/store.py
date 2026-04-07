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

log = logging.getLogger(__name__)

# --- .knod format constants ---
KNOD_MAGIC = 0x6B6E6F64  # "knod"
KNOD_VERSION = 2

SECTION_GRAPH = 0x01
SECTION_MODEL = 0x02
SECTION_LIMBO = 0x03
SECTION_REGISTRY = 0x04

# ---- Per-specialist model checkpoints (.pt) ----


# --- Graph serialisation helpers ---


def graph_to_state(graph: Graph, *, include_limbo: bool = False) -> dict:
	state = {
		"name": graph.name,
		"purpose": graph.purpose,
		"descriptors": graph.descriptors,
		"next_id": graph._next_id,
		"profile": graph._profile,
		"registry_nodes": graph._registry_nodes,
		"max_thoughts": graph.max_thoughts,
		"max_edges": graph.max_edges,
		"thoughts": {tid: t.to_dict() for tid, t in graph.thoughts.items()},
		"edges": [e.to_dict() for e in graph.edges],
	}
	if include_limbo:
		state["limbo"] = [lt.to_dict() for lt in graph.limbo]
	return state


def graph_from_state(state: dict, *, maturity_divisor: int = 50) -> Graph:
	graph = Graph(name=state.get("name", ""), purpose=state["purpose"])
	graph.descriptors = state.get("descriptors", {})
	graph._next_id = state["next_id"]
	graph._profile = state.get("profile")
	graph._registry_nodes = state.get("registry_nodes", {})
	graph.max_thoughts = state.get("max_thoughts", 0)
	graph.max_edges = state.get("max_edges", 0)
	graph.maturity_divisor = maturity_divisor

	for tid_str, tdata in state["thoughts"].items():
		tid = int(tid_str) if isinstance(tid_str, str) else tid_str
		graph.thoughts[tid] = Thought.from_dict(tid, tdata)

	for edata in state["edges"]:
		graph.edges.append(Edge.from_dict(edata))

	for ldata in state.get("limbo", []):
		graph.limbo.append(LimboThought.from_dict(ldata))

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


def save_all(graph: Graph, model: KnodMPNN, strand: StrandLayer, base_path: str | Path):
	"""Save everything to a single base_path.knod file. Also updates shared base."""
	base = Path(base_path)
	save_knod(graph, model, strand, base.with_suffix(".knod"))
	# Update the shared base model
	save_base_model(model)


def load_all(cfg, base_path: str | Path) -> tuple[Graph, KnodMPNN, StrandLayer]:
	"""Load graph, model, and strand from base_path.knod."""
	base = Path(base_path)
	knod_path = base.with_suffix(".knod")

	if knod_path.exists():
		return load_knod(cfg, knod_path)

	raise FileNotFoundError(f"No .knod file at {base}")


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
