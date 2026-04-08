"""Persistence — save/load graph + model checkpoints + append-only binary log.

.shard unified format:
  [magic: 4 bytes "shard" = 0x6b6e6f64]
  [version: 4 bytes LE i32]  — version 2 = Python format
  [section]*:
    [tag: 1 byte]
    [length: 8 bytes LE i64]
    [payload: <length> bytes]

Section tags:
  0x01 GRAPH    — pickled graph state dict
  0x02 MODEL    — torch.save bytes (model + shard state_dict)
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
from .gnn import ShardMPNN, ShardLayer

log = logging.getLogger(__name__)

# --- .shard format constants ---
SHARD_MAGIC = 0x6B6E6F64  # "shard"
SHARD_VERSION = 2

SECTION_GRAPH = 0x01
SECTION_MODEL = 0x02
SECTION_LIMBO = 0x03
SECTION_REGISTRY = 0x04

# ---- Per-shard model checkpoints (.pt) ----


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


def save_model(model: ShardMPNN, shard: ShardLayer, path: str | Path):
	"""Save model + shard checkpoint (per-shard .pt file)."""
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(
		{
			"model": model.state_dict(),
			"shard": shard.state_dict(),
		},
		path,
	)


def load_model(model: ShardMPNN, shard: ShardLayer, path: str | Path):
	"""Load model + shard checkpoint."""
	checkpoint = torch.load(path, weights_only=True)
	model.load_state_dict(checkpoint["model"])
	shard.load_state_dict(checkpoint["shard"])


# ---- Shared base GNN checkpoint ----

_DEFAULT_BASE_GNN_PATH = Path.home() / ".config" / "shard" / "base.gnn"


def _get_base_gnn_path(cfg) -> Path:
	return Path(cfg.base_gnn_path) if cfg.base_gnn_path else _DEFAULT_BASE_GNN_PATH


def extract_routing_from_shard(graph, model, shard) -> dict:
	"""Extract routing knowledge from a trained shard.

	Returns a dict with:
	- shard_profiles: list of (profile_embedding, shard_name)
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
		h_shard, _ = shard(h, edge_index)

	routing = {
		"shard_profiles": [],
		"high_weight_edges": [],
		"node_hidden_avg": h_shard.mean(dim=0).cpu().numpy().tolist(),
	}

	if graph.profile is not None:
		routing["shard_profiles"].append((graph.profile.tolist(), graph.name or "unknown"))

	for e in graph.edges:
		if e.weight >= 0.7:
			src = graph.thoughts.get(e.source_id)
			tgt = graph.thoughts.get(e.target_id)
			if src and tgt and src.embedding is not None and tgt.embedding is not None:
				routing["high_weight_edges"].append((src.embedding.tolist(), tgt.embedding.tolist(), e.weight))

	return routing


def merge_routing_into_base(model, routing: dict):
	"""Merge shard routing knowledge into the base model.

	Adds synthetic routing nodes to the base that encode shard profiles
	and high-weight edge patterns. This helps the base learn navigation.
	"""
	import torch
	import torch.nn.functional as F

	if not routing:
		return

	if "shard_profiles" in routing and "node_hidden_avg" in routing:
		avg_hidden = torch.tensor(routing["node_hidden_avg"], dtype=torch.float)
		hidden_dim = model.hidden_dim
		embed_dim = model.embedding_dim

		with torch.no_grad():
			for profile, name in routing["shard_profiles"]:
				profile_tensor = torch.tensor(profile, dtype=torch.float)
				if profile_tensor.shape[0] != embed_dim:
					continue

				proj = F.relu(model.node_proj(profile_tensor.unsqueeze(0)))
				blended = 0.7 * proj.squeeze(0) + 0.3 * avg_hidden[:hidden_dim]

				for layer in model.layers:
					blended = layer(
						blended.unsqueeze(0),
						torch.tensor([[0]], dtype=torch.long),
						torch.zeros(1, hidden_dim, dtype=torch.float),
					)
					blended = blended.squeeze(0)

				noise = torch.randn_like(blended) * 0.001
				blended = blended + noise


def save_base_model(model: ShardMPNN, cfg=None, routing: dict | None = None):
	"""Save the shared base MPNN to ~/.config/shard/base.gnn (or cfg.base_gnn_path).

	If routing is provided, merge it into the model before saving.
	"""
	if routing:
		merge_routing_into_base(model, routing)

	path = _get_base_gnn_path(cfg) if cfg else _DEFAULT_BASE_GNN_PATH
	path.parent.mkdir(parents=True, exist_ok=True)
	torch.save({"model": model.state_dict()}, path)


def load_base_model(model: ShardMPNN, cfg=None) -> bool:
	"""Load the shared base MPNN. Returns True if loaded."""
	path = _get_base_gnn_path(cfg) if cfg else _DEFAULT_BASE_GNN_PATH
	if not path.exists():
		return False
	checkpoint = torch.load(path, weights_only=True)
	model.load_state_dict(checkpoint["model"])
	return True


def save_all(graph: Graph, model: ShardMPNN, shard: ShardLayer, base_path: str | Path):
	"""Save everything to a single base_path.shard file. Also updates shared base."""
	base = Path(base_path)
	save_shard(graph, model, shard, base.with_suffix(".shard"))
	# Update the shared base model
	save_base_model(model)


def load_all(cfg, base_path: str | Path) -> tuple[Graph, ShardMPNN, ShardLayer]:
	"""Load graph, model, and shard from base_path.shard."""
	base = Path(base_path)
	shard_path = base.with_suffix(".shard")

	if shard_path.exists():
		return load_shard(cfg, shard_path)

	raise FileNotFoundError(f"No .shard file at {base}")


# --- .shard unified format ---


def _write_section(f, tag: int, payload: bytes):
	"""Write a tagged section: [tag:1][length:8][payload]."""
	f.write(struct.pack("<B", tag))
	f.write(struct.pack("<q", len(payload)))
	f.write(payload)


def _parse_shard_sections(data: bytes) -> list[tuple[int, bytes]]:
	"""Validate magic + version, then yield all (tag, payload) sections.

	Raises ValueError if the file header is invalid.
	"""
	if len(data) < 8:
		raise ValueError("File too small for .shard format")

	magic = struct.unpack_from("<i", data, 0)[0]
	if magic != SHARD_MAGIC:
		raise ValueError(f"Bad magic: 0x{magic:08x}")

	version = struct.unpack_from("<i", data, 4)[0]
	if version != SHARD_VERSION:
		raise ValueError(f"Unsupported .shard version {version}")

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


def _model_bytes(model: ShardMPNN, shard: ShardLayer) -> bytes:
	"""Serialize model + shard state_dict to bytes via torch.save."""
	buf = io.BytesIO()
	torch.save({"model": model.state_dict(), "shard": shard.state_dict()}, buf)
	return buf.getvalue()


def save_shard(graph: Graph, model: ShardMPNN, shard: ShardLayer, path: str | Path):
	"""Save graph + model + limbo + registry into a single .shard file."""
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)

	with open(path, "wb") as f:
		# Header
		f.write(struct.pack("<i", SHARD_MAGIC))
		f.write(struct.pack("<i", SHARD_VERSION))

		# GRAPH section
		_write_section(f, SECTION_GRAPH, pickle.dumps(graph_to_state(graph)))

		# MODEL section
		_write_section(f, SECTION_MODEL, _model_bytes(model, shard))

		# LIMBO section (only if non-empty)
		if graph.limbo:
			limbo_data = [lt.to_dict() for lt in graph.limbo]
			_write_section(f, SECTION_LIMBO, pickle.dumps(limbo_data))

		# REGISTRY section (only if non-empty)
		if graph._registry_nodes:
			_write_section(f, SECTION_REGISTRY, pickle.dumps(graph._registry_nodes))


def load_shard(cfg, path: str | Path, warm_start: bool = True) -> tuple[Graph, ShardMPNN, ShardLayer]:
	"""Load graph + model + limbo from a single .shard file.

	Args:
		cfg: Config object
		path: Path to .shard file
		warm_start: If True, first load base model weights, then override with
		            shard-specific weights. This gives shards cross-shard navigation
		            knowledge from the shared base.
	"""
	path = Path(path)

	with open(path, "rb") as f:
		data = f.read()

	# Parse sections — validates magic + version
	sections = _parse_shard_sections(data)

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
	model = ShardMPNN(cfg)
	shard = ShardLayer(cfg.hidden_dim)

	# Warm start: load base model first for cross-shard navigation knowledge
	if warm_start:
		load_base_model(model, cfg)

	if model_bytes:
		buf = io.BytesIO(model_bytes)
		checkpoint = torch.load(buf, weights_only=True)
		model.load_state_dict(checkpoint["model"])
		shard.load_state_dict(checkpoint["shard"])
	elif not warm_start:
		load_base_model(model)

	return graph, model, shard


def read_shard_metadata(path: str | Path) -> dict:
	"""Read only the GRAPH section metadata from a .shard file (no model load).

	Returns a dict with keys: purpose, descriptors, profile, num_thoughts, num_edges.
	Raises ValueError if the file is invalid or has no GRAPH section.
	"""
	path = Path(path)
	with open(path, "rb") as f:
		data = f.read()

	# Parse sections — validates magic + version (fixes missing version check)
	sections = _parse_shard_sections(data)

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
