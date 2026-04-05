package graph

EMBEDDING_DIM :: 1536 // matches external LLM (OpenAI ada-002 / text-embedding-3)
LOG_MAGIC :: 0x6B6E6F67 // "knog"
LOG_VERSION :: i32(2)

// .strand container format (graph + strand checkpoint in one file)
KNOD_MAGIC   :: 0x6B6E6F64 // "knod"
KNOD_VERSION :: i32(1)
SECTION_GRAPH  :: u8(1)
SECTION_STRAND :: u8(2)

RecordType :: enum u8 {
	THOUGHT = 1,
	EDGE    = 2,
}

Config :: struct {
	// Path for the append-only graph log file.
	data_path:            string,
	// Maximum number of thoughts before eviction (0 = unlimited).
	max_thoughts:         int,
	// Maximum number of edges before eviction (0 = unlimited).
	max_edges:            int,
	// Decay factor applied to edge weights over time (0.0 = no decay).
	edge_decay:           f32,
	// Minimum cosine similarity for a find result to be returned.
	similarity_threshold: f32,
	// Default top-k for find operations.
	default_find_k:       int,
	// Maximum number of edge reasoning strings to include in query context.
	max_context_edges:    int,
}

DEFAULT_CONFIG :: Config {
	data_path            = "knod.graph",
	max_thoughts         = 0,
	max_edges            = 0,
	edge_decay           = 0.0,
	similarity_threshold = 0.0,
	default_find_k       = 10,
	max_context_edges    = 3,
}

cfg: Config = DEFAULT_CONFIG

// Apply external configuration. Call before create().
configure :: proc(c: Config) {
	cfg = c
}
