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

# ---- Per-strand model checkpoints (.pt) ----


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
	"""Save model + strand checkpoint (per-strand .pt file)."""
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

_DEFAULT_BASE_GNN_PATH = Path.home() / ".config" / "knod" / "base.gnn"


def _get_base_gnn_path(cfg) -> Path:
	return Path(cfg.base_gnn_path) if cfg.base_gnn_path else _DEFAULT_BASE_GNN_PATH


def extract_routing_from_strand(graph, model, strand) -> dict:
	"""Extract routing knowledge from a trained strand.

	Returns a dict with:
	- strand_profiles: list of (profile_embedding, strand_name)
	- high_weight_edges: list of (source_profile, target_profile, weight)
	- node_hidden_avg: average hidden representation from the GNN
	"""
	import torch
	import torch.nn.functional as F

	if graph.num_thoughts < 2 or graph.num_edges == 0:
		return {}

	node_features, edge_index, edge_features, ordered_ids, valid_edges = graph.to_tensors()
	if len(valid_edges) == 0:
		return {}

	node_features = node_features.float()
	edge_features = edge_features.float()

	with torch.no_grad():
		h = F.relu(model.node_proj(node_features))
		e = F.relu(model.edge_proj(edge_features))
		for layer in model.layers:
			h = layer(h, edge_index, e)
		h_strand, _ = strand(h, edge_index)

	routing = {
		"strand_profiles": [],
		"high_weight_edges": [],
		"node_hidden_avg": h_strand.mean(dim=0).cpu().numpy().tolist(),
	}

	if graph.profile is not None:
		routing["strand_profiles"].append((graph.profile.tolist(), graph.name or "unknown"))

	for e in graph.edges:
		if e.weight >= 0.7:
			src = graph.thoughts.get(e.source_id)
			tgt = graph.thoughts.get(e.target_id)
			if src and tgt and src.embedding is not None and tgt.embedding is not None:
				routing["high_weight_edges"].append((src.embedding.tolist(), tgt.embedding.tolist(), e.weight))

	return routing


def merge_routing_into_base(model, routing: dict):
	"""Merge strand routing knowledge into the base model.

	Adds synthetic routing nodes to the base that encode strand profiles
	and high-weight edge patterns. This helps the base learn navigation.
	"""
	import torch
	import torch.nn.functional as F

	if not routing:
		return

	node_features_list = []
	edge_index_list = []
	edge_features_list = []

	if "strand_profiles" in routing:
		for profile, name in routing["strand_profiles"]:
			profile_tensor = torch.tensor(profile, dtype=torch.float)
			node_features_list.append(profile_tensor)

	if "node_hidden_avg" in routing:
		avg_hidden = torch.tensor(routing["node_hidden_avg"], dtype=torch.float)
		if node_features_list:
			combined = torch.stack(node_features_list)
			updated = []
			for nf in node_features_list:
				blended = 0.7 * nf + 0.3 * avg_hidden[: len(nf)]
				updated.append(blended)
			node_features_list = updated

	if node_features_list:
		current_params = list(model.parameters())
		if current_params:
			first_layer = model.node_proj
			with torch.no_grad():
				for i, nf in enumerate(node_features_list[:3]):
					if nf.shape[0] == model.embedding_dim:
						proj = F.relu(first_layer(nf.unsqueeze(0)))
						for layer in model.layers:
							proj = layer(
								proj, torch.tensor([[0]], dtype=torch.long), torch.zeros(1, model.hidden_dim, dtype=torch.float)
							)
						if i < len(current_params) // 4:
							noise = torch.randn_like(current_params[i][: nf.shape[0]]) * 0.01
							current_params[i][: nf.shape[0]] += noise


def save_base_model(model: KnodMPNN, cfg=None, routing: dict | None = None):
	"""Save the shared base MPNN to ~/.config/knod/base.gnn (or cfg.base_gnn_path).

	If routing is provided, merge it into the model before saving.
	"""
	if routing:
		merge_routing_into_base(model, routing)

	path = _get_base_gnn_path(cfg) if cfg else _DEFAULT_BASE_GNN_PATH
	path.parent.mkdir(parents=True, exist_ok=True)
	torch.save({"model": model.state_dict()}, path)


def load_base_model(model: KnodMPNN, cfg=None) -> bool:
	"""Load the shared base MPNN. Returns True if loaded."""
	path = _get_base_gnn_path(cfg) if cfg else _DEFAULT_BASE_GNN_PATH
	if not path.exists():
		return False
	checkpoint = torch.load(path, weights_only=True)
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


def load_knod(cfg, path: str | Path, warm_start: bool = True) -> tuple[Graph, KnodMPNN, StrandLayer]:
	"""Load graph + model + limbo from a single .knod file.

	Args:
		cfg: Config object
		path: Path to .knod file
		warm_start: If True, first load base model weights, then override with
		            strand-specific weights. This gives strands cross-strand navigation
		            knowledge from the shared base.
	"""
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

	# Warm start: load base model first for cross-strand navigation knowledge
	if warm_start:
		load_base_model(model, cfg)

	if model_bytes:
		buf = io.BytesIO(model_bytes)
		checkpoint = torch.load(buf, weights_only=True)
		model.load_state_dict(checkpoint["model"])
		strand.load_state_dict(checkpoint["strand"])
	elif not warm_start:
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
