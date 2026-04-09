"""Configuration management — reads from ~/.config/shard/config."""

import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
	# OpenAI tier (OpenAI-compatible API)
	openai_api_key: str = ""
	openai_base_url: str = "https://api.openai.com/v1"
	openai_model: str = "gpt-4o-mini"
	openai_embedding_model: str = "text-embedding-3-small"

	# Local tier (Ollama or other local APIs)
	local_api_key: str = "ollama"
	local_base_url: str = ""
	local_model: str = ""
	local_embedding_model: str = ""

	# Embedding settings
	embedding_dim: int = 1024

	# Model settings
	hidden_dim: int = 512
	num_layers: int = 3

	edge_mask_ratio: float = 0.15
	margin: float = 0.1
	lr_max: float = 1e-3
	lr_min: float = 5e-5
	weight_decay: float = 0.01
	similarity_threshold: float = 0.7
	min_link_weight: float = 0.1
	maturity_divisor: int = 50
	top_k: int = 5
	max_thoughts: int = 0
	max_edges: int = 0
	decay_coefficient: float = 0.0
	dedup_threshold: float = 0.95
	limbo_scan_interval: float = 60.0
	limbo_cluster_min: int = 3
	limbo_cluster_threshold: float = 0.50
	mcmc_linked_base: float = 0.5
	mcmc_unlinked_base: float = 0.3
	strictness: float = 1.0
	shard_match_threshold: float = 0.8
	shard_split_threshold: float = 0.85
	shard_split_min: int = 5
	shard_split_interval: float = 300.0
	confidence_threshold: float = 0.85
	query_routing_threshold: float = 0.3
	traversal_depth: int = 2
	traversal_fan_out: int = 10
	refinement_interval: int = 10
	refinement_boost: float = 0.02
	refinement_dampen: float = 0.01
	refinement_min_traversals: int = 3

	tcp_port: int = 7999
	http_port: int = 8080
	mcp_port: int = 8766

	graph_path: str = ".shard/graph.shard"
	base_gnn_path: str = str(Path.home() / ".config" / "shard" / "base.gnn")

	@classmethod
	def load(cls, config_dir: Path | None = None) -> "Config":
		cfg = cls()

		# Load user home config first (lower priority)
		home_config = Path.home() / ".config" / "shard" / "config"
		config_paths = [home_config] if home_config.exists() else []

		# Local config last (higher priority)
		local_config = (config_dir or Path.cwd()) / "config"
		if local_config.exists():
			config_paths.append(local_config)

		# Section to field prefix mapping
		section_prefixes = {
			"model": "",
			"graph": "",
			"training": "",
			"limbo": "",
			"strand": "",
			"traversal": "",
			"refinement": "",
			"server": "",
			"storage": "",
		}

		for config_path in config_paths:
			if not config_path.exists():
				continue

			kv: dict[str, str] = {}
			current_section = ""
			for line in config_path.read_text(encoding="utf-8").splitlines():
				line = line.strip()
				if not line or line.startswith("#"):
					continue
				if line.startswith("[") and line.endswith("]"):
					current_section = line[1:-1]
					continue
				if "=" in line:
					k, v = line.split("=", 1)
					key = k.strip()
					val = v.strip()
					# Apply section prefix unless in common sections
					prefix = section_prefixes.get(current_section, current_section + "_")
					if prefix:
						key = f"{prefix}{key}"
					kv[key] = val

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

		return cfg
