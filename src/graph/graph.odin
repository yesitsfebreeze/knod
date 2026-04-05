package graph

import "../util"
import "core:math"
import "core:os"


create :: proc(g: ^Graph) {
	g.thoughts = make(map[u64]Thought)
	g.edges = make([dynamic]Edge)
	g.outgoing = make(map[u64][dynamic]int)
	g.incoming = make(map[u64][dynamic]int)
	g.next_id = 1
	g.purpose = ""
	g.stream_handle = os.INVALID_HANDLE
	g.profile = {}
	g.profile_count = 0
	g.tags = make([dynamic]Tag)
	g.descriptors = make(map[string]Descriptor)
}


release :: proc(g: ^Graph) {
	for _, &t in g.thoughts {
		delete(t.text)
		delete(t.source_id)
	}
	delete(g.thoughts)

	for &e in g.edges {
		delete(e.reasoning)
	}
	delete(g.edges)

	for _, &v in g.outgoing {
		delete(v)
	}
	delete(g.outgoing)

	for _, &v in g.incoming {
		delete(v)
	}
	delete(g.incoming)

	if len(g.purpose) > 0 {
		delete(g.purpose)
	}

	release_tags(g)
	release_descriptors(g)

	if g.stream_handle != os.INVALID_HANDLE {
		os.close(g.stream_handle)
		g.stream_handle = os.INVALID_HANDLE
	}
}

set_purpose :: proc(g: ^Graph, purpose: string) {
	if len(g.purpose) > 0 {
		delete(g.purpose)
	}
	g.purpose = util.clone_string(purpose)
}

add_thought :: proc(
	g: ^Graph,
	text, source_id: string,
	embedding: Embedding,
	created_at: i64,
) -> u64 {
	id := g.next_id
	g.next_id += 1

	t := Thought {
		id            = id,
		text          = util.clone_string(text),
		embedding     = embedding,
		source_id     = util.clone_string(source_id),
		created_at    = created_at,
		access_count  = 0,
		last_accessed = 0,
	}
	g.thoughts[id] = t

	n := f32(g.profile_count)
	for i in 0 ..< EMBEDDING_DIM {
		g.profile[i] = (g.profile[i] * n + embedding[i]) / (n + 1.0)
	}
	g.profile_count += 1

	if g.stream_handle != os.INVALID_HANDLE {
		stream_thought(g.stream_handle, &g.thoughts[id])
	}

	return id
}

add_edge :: proc(
	g: ^Graph,
	source_id, target_id: u64,
	weight: f32,
	reasoning: string,
	embedding: Embedding,
	created_at: i64,
) -> bool {
	if source_id not_in g.thoughts || target_id not_in g.thoughts {
		return false
	}

	edge_idx := len(g.edges)
	e := Edge {
		source_id  = source_id,
		target_id  = target_id,
		weight     = weight,
		reasoning  = util.clone_string(reasoning),
		embedding  = embedding,
		created_at = created_at,
	}
	append(&g.edges, e)

	if g.stream_handle != os.INVALID_HANDLE {
		stream_edge(g.stream_handle, &g.edges[edge_idx])
	}

	{
		if source_id not_in g.outgoing {
			g.outgoing[source_id] = make([dynamic]int)
		}
		list := g.outgoing[source_id]
		append(&list, edge_idx)
		g.outgoing[source_id] = list
	}

	{
		if target_id not_in g.incoming {
			g.incoming[target_id] = make([dynamic]int)
		}
		list := g.incoming[target_id]
		append(&list, edge_idx)
		g.incoming[target_id] = list
	}

	return true
}

get_thought :: proc(g: ^Graph, id: u64) -> ^Thought {
	if id in g.thoughts {
		return &g.thoughts[id]
	}
	return nil
}

touch :: proc(g: ^Graph, id: u64, now: i64) {
	if t := get_thought(g, id); t != nil {
		t.access_count += 1
		t.last_accessed = now
	}
}

outgoing :: proc(g: ^Graph, id: u64) -> []Edge {
	if id not_in g.outgoing {
		return {}
	}
	indices := g.outgoing[id]
	if len(indices) == 0 {
		return {}
	}
	result := make([]Edge, len(indices))
	for edge_idx, i in indices {
		result[i] = g.edges[edge_idx]
	}
	return result
}

incoming :: proc(g: ^Graph, id: u64) -> []Edge {
	if id not_in g.incoming {
		return {}
	}
	indices := g.incoming[id]
	if len(indices) == 0 {
		return {}
	}
	result := make([]Edge, len(indices))
	for edge_idx, i in indices {
		result[i] = g.edges[edge_idx]
	}
	return result
}

thought_count :: proc(g: ^Graph) -> int {
	return len(g.thoughts)
}

edge_count :: proc(g: ^Graph) -> int {
	return len(g.edges)
}

cosine_similarity :: proc(a, b: ^Embedding) -> f32 {
	dot: f32 = 0.0
	norm_a: f32 = 0.0
	norm_b: f32 = 0.0
	for i in 0 ..< EMBEDDING_DIM {
		dot += a[i] * b[i]
		norm_a += a[i] * a[i]
		norm_b += b[i] * b[i]
	}
	denom := math.sqrt(norm_a) * math.sqrt(norm_b)
	if denom == 0.0 {
		return 0.0
	}
	return dot / denom
}


find_thoughts :: proc(g: ^Graph, query: ^Embedding, k: int) -> []findResult {
	if len(g.thoughts) == 0 || k <= 0 {
		return {}
	}

	n := min(k, len(g.thoughts))
	results := make([dynamic]findResult, 0, len(g.thoughts))
	defer delete(results)

	for id, &t in g.thoughts {
		sim := cosine_similarity(query, &t.embedding)
		append(&results, findResult{id = id, score = sim})
	}

	for i in 1 ..< len(results) {
		j := i
		for j > 0 && results[j].score > results[j - 1].score {
			results[j], results[j - 1] = results[j - 1], results[j]
			j -= 1
		}
	}

	out := make([]findResult, n)
	for i in 0 ..< n {
		out[i] = results[i]
	}
	return out
}


find_edges :: proc(g: ^Graph, query: ^Embedding, k: int) -> []EdgefindResult {
	if len(g.edges) == 0 || k <= 0 {
		return {}
	}

	n := min(k, len(g.edges))
	results := make([dynamic]EdgefindResult, 0, len(g.edges))
	defer delete(results)

	for i in 0 ..< len(g.edges) {
		sim := cosine_similarity(query, &g.edges[i].embedding)
		append(&results, EdgefindResult{edge_index = i, score = sim})
	}

	for i in 1 ..< len(results) {
		j := i
		for j > 0 && results[j].score > results[j - 1].score {
			results[j], results[j - 1] = results[j - 1], results[j]
			j -= 1
		}
	}

	out := make([]EdgefindResult, n)
	for i in 0 ..< n {
		out[i] = results[i]
	}
	return out
}
