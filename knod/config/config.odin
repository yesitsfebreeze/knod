package config

import "core:fmt"
import "core:os"
import "core:path/filepath"
import "core:strconv"
import "core:strings"

import "../util"

Config :: struct {
	api_key:              string,
	base_url:             string,
	embedding_model:      string,
	chat_model:           string,
	timeout_ms:           int,
	tcp_port:             int,
	http_port:            int,
	graph_path:           string,
	base_path:            string,
	max_thoughts:         int,
	max_edges:            int,
	edge_decay:           f32,
	similarity_threshold: f32,
	find_k:               int,
	max_similar:          int,
	edge_threshold:       f32,
	maturity_threshold:   int,
	min_link_weight:      f32,
	max_context_edges:    int,
	max_tags:             int,
	limbo_cluster_min:    int,
	dedup_threshold:      f32,
}

DEFAULT :: Config {
	api_key              = "",
	base_url             = "https://api.openai.com/v1",
	embedding_model      = "text-embedding-3-small",
	chat_model           = "gpt-4o-mini",
	timeout_ms           = 30_000,
	tcp_port             = 7999,
	http_port            = 8080,
	graph_path           = "knod.strand",
	base_path            = "",
	max_thoughts         = 0,
	max_edges            = 0,
	edge_decay           = 0.0,
	similarity_threshold = 0.7,
	find_k               = 10,
	max_similar          = 5,
	edge_threshold       = 0.3,
	maturity_threshold   = 50,
	min_link_weight      = 0.1,
	max_context_edges    = 3,
	max_tags             = 128,
	limbo_cluster_min    = 3,
	dedup_threshold      = 0.95,
}

release :: proc(cfg: ^Config) {
	delete(cfg.api_key)
	delete(cfg.base_url)
	delete(cfg.embedding_model)
	delete(cfg.chat_model)
	delete(cfg.graph_path)
	if len(cfg.base_path) > 0 {delete(cfg.base_path)}
}

config_path :: proc() -> string {
	home := util.home_dir()
	if len(home) == 0 {
		return ""
	}
	defer delete(home)
	return filepath.join({home, ".config", "knod", "config"})
}

config_dir :: proc() -> string {
	home := util.home_dir()
	if len(home) == 0 {
		return ""
	}
	defer delete(home)
	return filepath.join({home, ".config", "knod"})
}

load :: proc() -> (cfg: Config, ok: bool) {
	path := config_path()
	if len(path) == 0 {
		return DEFAULT, false
	}

	return load_from(path)
}

load_from :: proc(path: string) -> (cfg: Config, ok: bool) {
	data, read_ok := os.read_entire_file(path)
	if !read_ok {
		return DEFAULT, false
	}
	defer delete(data)

	return parse(string(data))
}

parse :: proc(content: string) -> (cfg: Config, ok: bool) {
	cfg = DEFAULT
	cfg.api_key = strings.clone(DEFAULT.api_key)
	cfg.base_url = strings.clone(DEFAULT.base_url)
	cfg.embedding_model = strings.clone(DEFAULT.embedding_model)
	cfg.chat_model = strings.clone(DEFAULT.chat_model)
	cfg.graph_path = strings.clone(DEFAULT.graph_path)
	cfg.base_path = strings.clone(DEFAULT.base_path)

	lines := strings.split(content, "\n")
	defer delete(lines)

	for line in lines {
		trimmed := strings.trim_space(line)

		if len(trimmed) == 0 || trimmed[0] == '#' {
			continue
		}

		eq_idx := strings.index(trimmed, "=")
		if eq_idx < 0 {
			continue
		}

		key := strings.trim_space(trimmed[:eq_idx])
		val := strings.trim_space(trimmed[eq_idx + 1:])

		switch key {
		case "api_key":
			delete(cfg.api_key)
			cfg.api_key = strings.clone(val)
		case "base_url":
			delete(cfg.base_url)
			cfg.base_url = strings.clone(val)
		case "embedding_model":
			delete(cfg.embedding_model)
			cfg.embedding_model = strings.clone(val)
		case "chat_model":
			delete(cfg.chat_model)
			cfg.chat_model = strings.clone(val)
		case "timeout_ms":
			if v, parse_ok := strconv.parse_int(val); parse_ok {
				cfg.timeout_ms = v
			}
		case "tcp_port":
			if v, parse_ok := strconv.parse_int(val); parse_ok {
				cfg.tcp_port = v
			}
		case "http_port":
			if v, parse_ok := strconv.parse_int(val); parse_ok {
				cfg.http_port = v
			}
		case "graph_path":
			delete(cfg.graph_path)
			cfg.graph_path = strings.clone(val)
		case "base_path":
			if len(cfg.base_path) > 0 {delete(cfg.base_path)}
			cfg.base_path = strings.clone(val)
		case "max_thoughts":
			if v, parse_ok := strconv.parse_int(val); parse_ok {
				cfg.max_thoughts = v
			}
		case "max_edges":
			if v, parse_ok := strconv.parse_int(val); parse_ok {
				cfg.max_edges = v
			}
		case "edge_decay":
			if v, parse_ok := strconv.parse_f32(val); parse_ok {
				cfg.edge_decay = v
			}
		case "similarity_threshold":
			if v, parse_ok := strconv.parse_f32(val); parse_ok {
				cfg.similarity_threshold = v
			}
		case "find_k":
			if v, parse_ok := strconv.parse_int(val); parse_ok {
				cfg.find_k = v
			}
		case "max_similar":
			if v, parse_ok := strconv.parse_int(val); parse_ok {
				cfg.max_similar = v
			}
		case "edge_threshold":
			if v, parse_ok := strconv.parse_f32(val); parse_ok {
				cfg.edge_threshold = v
			}
		case "maturity_threshold":
			if v, parse_ok := strconv.parse_int(val); parse_ok {
				cfg.maturity_threshold = v
			}
		case "min_link_weight":
			if v, parse_ok := strconv.parse_f32(val); parse_ok {
				cfg.min_link_weight = v
			}
		case "max_context_edges":
			if v, parse_ok := strconv.parse_int(val); parse_ok {
				cfg.max_context_edges = v
			}
		case "max_tags":
			if v, parse_ok := strconv.parse_int(val); parse_ok {
				cfg.max_tags = v
			}
		case "limbo_cluster_min":
			if v, parse_ok := strconv.parse_int(val); parse_ok {
				cfg.limbo_cluster_min = v
			}
		case "dedup_threshold":
			if v, parse_ok := strconv.parse_f32(val); parse_ok {
				cfg.dedup_threshold = v
			}
		}
	}

	return cfg, true
}

write_default :: proc() -> string {
	dir := config_dir()
	if len(dir) == 0 {
		return ""
	}

	util.ensure_dir(dir)

	path := config_path()
	if len(path) == 0 {
		return ""
	}

	if os.exists(path) {
		return path
	}

	b: strings.Builder
	strings.builder_init(&b)
	defer strings.builder_destroy(&b)

	fmt.sbprintf(&b, "# knod configuration\n")
	fmt.sbprintf(&b, "# Place this file at ~/.config/knod/config\n")
	fmt.sbprintf(&b, "\n")
	fmt.sbprintf(&b, "# OpenAI provider settings\n")
	fmt.sbprintf(&b, "api_key = %s\n", DEFAULT.api_key)
	fmt.sbprintf(&b, "base_url = %s\n", DEFAULT.base_url)
	fmt.sbprintf(&b, "embedding_model = %s\n", DEFAULT.embedding_model)
	fmt.sbprintf(&b, "chat_model = %s\n", DEFAULT.chat_model)
	fmt.sbprintf(&b, "timeout_ms = %d\n", DEFAULT.timeout_ms)
	fmt.sbprintf(&b, "\n")
	fmt.sbprintf(&b, "# Network ports\n")
	fmt.sbprintf(&b, "tcp_port = %d\n", DEFAULT.tcp_port)
	fmt.sbprintf(&b, "http_port = %d\n", DEFAULT.http_port)
	fmt.sbprintf(&b, "\n")
	fmt.sbprintf(&b, "# Graph persistence (relative to executable, or absolute path)\n")
	fmt.sbprintf(&b, "graph_path = %s\n", DEFAULT.graph_path)
	fmt.sbprintf(&b, "# Base MPNN checkpoint path (empty = ~/.config/knod/knod)\n")
	fmt.sbprintf(&b, "base_path = %s\n", DEFAULT.base_path)
	fmt.sbprintf(&b, "max_thoughts = %d\n", DEFAULT.max_thoughts)
	fmt.sbprintf(&b, "max_edges = %d\n", DEFAULT.max_edges)
	fmt.sbprintf(&b, "edge_decay = %f\n", DEFAULT.edge_decay)
	fmt.sbprintf(&b, "similarity_threshold = %f\n", DEFAULT.similarity_threshold)
	fmt.sbprintf(&b, "find_k = %d\n", DEFAULT.find_k)
	fmt.sbprintf(&b, "\n")
	fmt.sbprintf(&b, "# Ingestion\n")
	fmt.sbprintf(&b, "max_similar = %d\n", DEFAULT.max_similar)
	fmt.sbprintf(&b, "edge_threshold = %f\n", DEFAULT.edge_threshold)
	fmt.sbprintf(&b, "maturity_threshold = %d\n", DEFAULT.maturity_threshold)
	fmt.sbprintf(&b, "min_link_weight = %f\n", DEFAULT.min_link_weight)
	fmt.sbprintf(&b, "\n")
	fmt.sbprintf(&b, "# Query\n")
	fmt.sbprintf(&b, "max_context_edges = %d\n", DEFAULT.max_context_edges)
	fmt.sbprintf(&b, "\n")
	fmt.sbprintf(&b, "# Tags\n")
	fmt.sbprintf(&b, "max_tags = %d\n", DEFAULT.max_tags)
	fmt.sbprintf(&b, "\n")
	fmt.sbprintf(&b, "# Limbo — global holding graph for unconnected thoughts.\n")
	fmt.sbprintf(&b, "# Minimum similar thoughts before a cluster is promoted to a new store.\n")
	fmt.sbprintf(&b, "# Set to 0 to disable limbo.\n")
	fmt.sbprintf(&b, "limbo_cluster_min = %d\n", DEFAULT.limbo_cluster_min)
	fmt.sbprintf(&b, "\n")
	fmt.sbprintf(&b, "# Dedup — merge near-duplicate thoughts instead of creating duplicates.\n")
	fmt.sbprintf(&b, "# Cosine similarity threshold (0 = disabled, 0.95 = default).\n")
	fmt.sbprintf(&b, "dedup_threshold = %f\n", DEFAULT.dedup_threshold)

	contents := strings.to_string(b)
	if os.write_entire_file(path, transmute([]u8)contents) {
		return path
	}

	return ""
}
