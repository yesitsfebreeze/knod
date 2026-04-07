"""Configuration management — reads from ~/.config/knod/config or env vars."""

import dataclasses
import os
import typing
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
	api_key: str = ""
	base_url: str = "https://api.openai.com/v1"
	embedding_model: str = "text-embedding-3-small"
	chat_model: str = "gpt-4o-mini"
	embedding_dim: int = 1536
	hidden_dim: int = 512
	num_layers: int = 3
	fallback_base_url: str = ""
	fallback_api_key: str = "ollama"
	fallback_chat_model: str = ""
	edge_mask_ratio: float = 0.15
	margin: float = 0.1
	lr_max: float = 1e-3
	lr_min: float = 5e-5
	weight_decay: float = 0.01
	similarity_threshold: float = 0.7
	min_link_weight: float = 0.1
	maturity_divisor: int = 50
	top_k: int = 5
	max_thoughts: int = 0  # 0 = unlimited
	max_edges: int = 0  # 0 = unlimited
	decay_coefficient: float = 0.0  # per-hour edge weight decay; 0 = disabled
	dedup_threshold: float = 0.95  # cosine similarity above which a new thought merges into existing
	limbo_scan_interval: float = 60.0  # seconds between scans
	limbo_cluster_min: int = 3  # minimum cluster size to promote
	limbo_cluster_threshold: float = 0.50  # cosine threshold for clustering
	strand_match_threshold: float = 0.8
	confidence_threshold: float = 0.85
	query_routing_threshold: float = 0.3  # min profile similarity to include strand in query
	traversal_depth: int = 2  # max hops in Dijkstra path expansion
	traversal_fan_out: int = 10  # max new nodes discovered per expand() call
	refinement_interval: int = 10  # refine edge weights every N retrievals
	refinement_boost: float = 0.02  # weight boost for high-success edges
	refinement_dampen: float = 0.01  # weight reduction for zero-success edges
	refinement_min_traversals: int = 3  # min traversals before refining an edge

	# Server
	tcp_port: int = 7999
	http_port: int = 8080

	# Paths
	graph_path: str = ".knod/knod.knod"

	@classmethod
	def load(cls) -> "Config":
		cfg = cls()

		config_path = Path.home() / ".config" / "knod" / "config"
		if config_path.exists():
			kv: dict[str, str] = {}
			for line in config_path.read_text(encoding="utf-8").splitlines():
				line = line.strip()
				if not line or line.startswith("#"):
					continue
				if "=" in line:
					k, v = line.split("=", 1)
					kv[k.strip()] = v.strip()

			hints = typing.get_type_hints(cls)
			for f in dataclasses.fields(cls):
				if f.name not in kv:
					continue
				raw = kv[f.name]
				ftype = hints.get(f.name, str)
				try:
					if ftype is int:
						setattr(cfg, f.name, int(raw))
					elif ftype is float:
						setattr(cfg, f.name, float(raw))
					elif ftype is bool:
						setattr(cfg, f.name, raw.lower() in ("1", "true", "yes"))
					else:
						setattr(cfg, f.name, raw)
				except (ValueError, TypeError):
					pass

		if v := os.environ.get("OPENAI_API_KEY"):
			cfg.api_key = v
		if v := os.environ.get("KNOD_BASE_URL"):
			cfg.base_url = v
		if v := os.environ.get("KNOD_GRAPH_PATH"):
			cfg.graph_path = v

		return cfg
