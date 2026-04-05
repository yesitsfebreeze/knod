package graph

import "core:os"

Embedding :: [EMBEDDING_DIM]f32

Tag :: struct {
	dim_index: u16,
	label:     string,
}

Descriptor :: struct {
	name: string,
	text: string,
}

Graph :: struct {
	thoughts:      map[u64]Thought,
	edges:         [dynamic]Edge,
	outgoing:      map[u64][dynamic]int,
	incoming:      map[u64][dynamic]int,
	next_id:       u64,
	purpose:       string,
	stream_handle: os.Handle,
	profile:       Embedding,
	profile_count: u64,
	tags:          [dynamic]Tag,
	descriptors:   map[string]Descriptor,
}

Thought :: struct {
	id:            u64,
	text:          string,
	embedding:     Embedding,
	source_id:     string,
	created_at:    i64,
	access_count:  u32,
	last_accessed: i64,
}

findResult :: struct {
	id:    u64,
	score: f32,
}

Edge :: struct {
	source_id:  u64,
	target_id:  u64,
	weight:     f32,
	reasoning:  string,
	embedding:  Embedding,
	created_at: i64,
}

EdgefindResult :: struct {
	edge_index: int,
	score:      f32,
}
