package config

import "core:os"
import "core:path/filepath"
import "core:strconv"
import "core:strings"

Config :: struct {
	// Provider
	api_key:              string,
	base_url:             string,
	embedding_model:      string,
	chat_model:           string,
	timeout_ms:           int,

	// Network
	tcp_port:             int,
	http_port:            int,

	// Graph
	graph_path:           string,
	base_path:            string, // base MPNN checkpoint path (empty = ~/.config/knod/knod)
	max_thoughts:         int,
	max_edges:            int,
	edge_decay:           f32,
	similarity_threshold: f32,
	find_k:               int,

	// Ingestion
	max_similar:          int,
	edge_threshold:       f32,
	maturity_threshold:   int,
	min_link_weight:      f32,

	// Query
	max_context_edges:    int,

	// Tags
	max_tags:             int,

	// Limbo — global holding graph for unconnected thoughts.
	// Minimum number of similar thoughts in limbo required to promote a cluster
	// into a new specialist store (0 = disable limbo entirely).
	limbo_cluster_min:    int,
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
	similarity_threshold = 0.0,
	find_k               = 10,
	max_similar          = 5,
	edge_threshold       = 0.3,
	maturity_threshold   = 50,
	min_link_weight      = 0.1,
	max_context_edges    = 3,
	max_tags             = 128,
	limbo_cluster_min    = 3,
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
	home := home_dir()
	if len(home) == 0 {
		return ""
	}
	defer delete(home)
	return filepath.join({home, ".config", "knod", "config"})
}

config_dir :: proc() -> string {
	home := home_dir()
	if len(home) == 0 {
		return ""
	}
	defer delete(home)
	return filepath.join({home, ".config", "knod"})
}

@(private)
home_dir :: proc() -> string {
	home := os.get_env("USERPROFILE")
	if len(home) == 0 {
		home = os.get_env("HOME")
	}
	return home
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
		}
	}

	return cfg, true
}

default_contents :: proc() -> string {
	return(
		`# knod configuration
# Place this file at ~/.config/knod/config

# OpenAI provider settings
api_key =
base_url = https://api.openai.com/v1
embedding_model = text-embedding-3-small
chat_model = gpt-4o-mini
timeout_ms = 30000

# Network ports
tcp_port = 7999
http_port = 8080

# Graph persistence (relative to executable, or absolute path)
graph_path = knod.strand
# Base MPNN checkpoint path (empty = ~/.config/knod/knod)
base_path =
max_thoughts = 0
max_edges = 0
edge_decay = 0.0
similarity_threshold = 0.0
find_k = 10

# Ingestion
max_similar = 5
edge_threshold = 0.3
maturity_threshold = 50
min_link_weight = 0.1

# Query
max_context_edges = 3

# Tags
max_tags = 128

# Limbo — global holding graph for unconnected thoughts.
# Minimum similar thoughts before a cluster is promoted to a new store.
# Set to 0 to disable limbo.
limbo_cluster_min = 3
` \
	)
}

write_default :: proc() -> string {
	dir := config_dir()
	if len(dir) == 0 {
		return ""
	}

	ensure_dir(dir)

	path := config_path()
	if len(path) == 0 {
		return ""
	}

	if os.exists(path) {
		return path
	}

	contents := default_contents()
	if os.write_entire_file(path, transmute([]u8)contents) {
		return path
	}

	return ""
}

@(private)
ensure_dir :: proc(path: string) {
	if os.exists(path) {
		return
	}

	parent := filepath.dir(path)
	if len(parent) > 0 && parent != path {
		ensure_dir(parent)
	}

	os.make_directory(path)
}
