package graph


EMBEDDING_DIM :: 1536


LOG_MAGIC :: i32(0x6b6e6f67)
LOG_VERSION :: i32(2)
KNOD_MAGIC :: i32(0x6b6e6f64)
KNOD_VERSION :: i32(1)

SECTION_GRAPH    :: u8(1)
SECTION_STRAND   :: u8(2)
SECTION_LIMBO    :: u8(3)
SECTION_REGISTRY :: u8(4)

RecordType :: enum u8 {
	THOUGHT       = 1,
	EDGE          = 2,
	LIMBO_THOUGHT = 3,
}


Embedding :: [EMBEDDING_DIM]f32


Descriptor :: struct {
	name: string,
	text: string,
}


Thought :: struct {
	id:            u64,
	text:          string,
	embedding:     Embedding,
	source:        string,
	created_at:    i64,
	access_count:  u32,
	last_accessed: i64,
}


Edge :: struct {
	source_id:  u64,
	target_id:  u64,
	weight:     f32,
	reasoning:  string,
	embedding:  Embedding,
	created_at: i64,
}


LimboThought :: struct {
	text:       string,
	embedding:  Embedding,
	source:     string,
	created_at: i64,
}


FindResult :: struct {
	id:    u64,
	score: f32,
}


EdgeFindResult :: struct {
	edge_index: int,
	score:      f32,
}


Graph :: struct {
	thoughts:           map[u64]Thought,
	edges:              [dynamic]Edge,
	limbo:              [dynamic]LimboThought,
	outgoing:           map[u64][dynamic]int,
	incoming:           map[u64][dynamic]int,
	next_id:            u64,
	purpose:            string,
	profile:            Embedding,
	profile_count:      u64,
	descriptors:        map[string]Descriptor,
	max_thoughts:       int,
	max_edges:          int,
	maturity_threshold: int,
	registry_nodes:     map[string]u64,
}
