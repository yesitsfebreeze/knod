package graph

import "core:math"
import "core:strings"


MATURITY_THRESHOLD :: 50


EDGE_MIN_WEIGHT :: f32(0.01)


EDGE_SCORE_DISCOUNT :: f32(0.8)

create :: proc(g: ^Graph) {
	g.thoughts = make(map[u64]Thought)
	g.edges = make([dynamic]Edge)
	g.limbo = make([dynamic]LimboThought)
	g.outgoing = make(map[u64][dynamic]int)
	g.incoming = make(map[u64][dynamic]int)
	g.next_id = 1
	g.purpose = ""
	g.profile = {}
	g.profile_count = 0
	g.descriptors = make(map[string]Descriptor)
	g.max_thoughts = 0
	g.max_edges = 0
	g.maturity_threshold = MATURITY_THRESHOLD
}

release :: proc(g: ^Graph) {
	for _, &t in g.thoughts {
		delete(t.text)
		delete(t.source)
	}
	delete(g.thoughts)

	for &e in g.edges {
		delete(e.reasoning)
	}
	delete(g.edges)

	for &lt in g.limbo {
		delete(lt.text)
		delete(lt.source)
	}
	delete(g.limbo)

	for _, &v in g.outgoing {delete(v)}
	delete(g.outgoing)
	for _, &v in g.incoming {delete(v)}
	delete(g.incoming)

	if len(g.purpose) > 0 {delete(g.purpose)}

	for _, &d in g.descriptors {
		delete(d.name)
		delete(d.text)
	}
	delete(g.descriptors)
}

set_purpose :: proc(g: ^Graph, purpose: string) {
	if len(g.purpose) > 0 {delete(g.purpose)}
	g.purpose = strings.clone(purpose)
}


maturity :: proc(g: ^Graph) -> f32 {
	div := g.maturity_threshold if g.maturity_threshold > 0 else MATURITY_THRESHOLD
	return min(f32(len(g.thoughts)) / f32(div), 1.0)
}

thought_count :: proc(g: ^Graph) -> int {return len(g.thoughts)}
edge_count :: proc(g: ^Graph) -> int {return len(g.edges)}
limbo_count :: proc(g: ^Graph) -> int {return len(g.limbo)}


add_thought :: proc(
	g: ^Graph,
	text, source: string,
	embedding: Embedding,
	created_at: i64,
) -> u64 {
	if g.max_thoughts > 0 && len(g.thoughts) >= g.max_thoughts {
		return 0
	}

	id := g.next_id
	g.next_id += 1

	t := Thought {
		id            = id,
		text          = strings.clone(text),
		embedding     = embedding,
		source        = strings.clone(source),
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
	if g.max_edges > 0 && len(g.edges) >= g.max_edges {
		return false
	}

	edge_idx := len(g.edges)
	e := Edge {
		source_id  = source_id,
		target_id  = target_id,
		weight     = weight,
		reasoning  = strings.clone(reasoning),
		embedding  = embedding,
		created_at = created_at,
	}
	append(&g.edges, e)


	if source_id not_in g.outgoing {g.outgoing[source_id] = make([dynamic]int)}
	out := g.outgoing[source_id]; append(&out, edge_idx); g.outgoing[source_id] = out

	if target_id not_in g.incoming {g.incoming[target_id] = make([dynamic]int)}
	inc := g.incoming[target_id]; append(&inc, edge_idx); g.incoming[target_id] = inc

	return true
}


add_limbo :: proc(g: ^Graph, text, source: string, embedding: Embedding, created_at: i64) {
	lt := LimboThought {
		text       = strings.clone(text),
		embedding  = embedding,
		source     = strings.clone(source),
		created_at = created_at,
	}
	append(&g.limbo, lt)
}


remove_limbo_indices :: proc(g: ^Graph, indices: []int) {
	if len(indices) == 0 {return}

	remove_set := make(map[int]bool, len(indices))
	defer delete(remove_set)
	for idx in indices {remove_set[idx] = true}


	for idx in indices {
		if idx >= 0 && idx < len(g.limbo) {
			delete(g.limbo[idx].text)
			delete(g.limbo[idx].source)
		}
	}


	write := 0
	for read in 0 ..< len(g.limbo) {
		if !remove_set[read] {
			g.limbo[write] = g.limbo[read]
			write += 1
		}
	}
	resize(&g.limbo, write)
}

get_thought :: proc(g: ^Graph, id: u64) -> ^Thought {
	if id in g.thoughts {return &g.thoughts[id]}
	return nil
}

touch :: proc(g: ^Graph, id: u64, now: i64) {
	if t := get_thought(g, id); t != nil {
		t.access_count += 1
		t.last_accessed = now
	}
}


cosine_similarity :: proc(a, b: ^Embedding) -> f32 {
	dot, norm_a, norm_b: f32
	for i in 0 ..< EMBEDDING_DIM {
		dot += a[i] * b[i]
		norm_a += a[i] * a[i]
		norm_b += b[i] * b[i]
	}
	denom := math.sqrt(norm_a) * math.sqrt(norm_b)
	if denom == 0.0 {return 0.0}
	return dot / denom
}


find_thoughts :: proc(g: ^Graph, query: ^Embedding, k: int, threshold: f32 = 0.0) -> []FindResult {
	if len(g.thoughts) == 0 || k <= 0 {return {}}

	results := make([dynamic]FindResult, 0, len(g.thoughts))
	defer delete(results)

	for id, &t in g.thoughts {
		sim := cosine_similarity(query, &t.embedding)
		if sim >= threshold {
			append(&results, FindResult{id = id, score = sim})
		}
	}


	for i in 1 ..< len(results) {
		j := i
		for j > 0 && results[j].score > results[j - 1].score {
			results[j], results[j - 1] = results[j - 1], results[j]
			j -= 1
		}
	}

	n := min(k, len(results))
	out := make([]FindResult, n)
	for i in 0 ..< n {out[i] = results[i]}
	return out
}


find_edges :: proc(
	g: ^Graph,
	query: ^Embedding,
	k: int,
	threshold: f32 = 0.0,
) -> []EdgeFindResult {
	if len(g.edges) == 0 || k <= 0 {return {}}

	results := make([dynamic]EdgeFindResult, 0, len(g.edges))
	defer delete(results)

	for i in 0 ..< len(g.edges) {
		raw_sim := cosine_similarity(query, &g.edges[i].embedding)
		sim := raw_sim * EDGE_SCORE_DISCOUNT
		if sim >= threshold {
			append(&results, EdgeFindResult{edge_index = i, score = sim})
		}
	}

	for i in 1 ..< len(results) {
		j := i
		for j > 0 && results[j].score > results[j - 1].score {
			results[j], results[j - 1] = results[j - 1], results[j]
			j -= 1
		}
	}

	n := min(k, len(results))
	out := make([]EdgeFindResult, n)
	for i in 0 ..< n {out[i] = results[i]}
	return out
}


apply_edge_decay :: proc(g: ^Graph, decay_coefficient: f32, now: i64) {
	if decay_coefficient <= 0.0 || len(g.edges) == 0 {return}


	surviving := make([dynamic]Edge, 0, len(g.edges))
	for &e in g.edges {
		age_hours := f32(now - e.created_at) / 3600.0
		if age_hours > 0 {
			e.weight *= math.pow(1.0 - decay_coefficient, age_hours)
		}
		if e.weight >= EDGE_MIN_WEIGHT {
			append(&surviving, e)
		} else {
			delete(e.reasoning)
		}
	}

	delete(g.edges)
	g.edges = surviving

	for _, &v in g.outgoing {delete(v)}
	delete(g.outgoing)
	for _, &v in g.incoming {delete(v)}
	delete(g.incoming)
	g.outgoing = make(map[u64][dynamic]int)
	g.incoming = make(map[u64][dynamic]int)
	for i in 0 ..< len(g.edges) {
		e := &g.edges[i]
		if e.source_id not_in g.outgoing {g.outgoing[e.source_id] = make([dynamic]int)}
		out := g.outgoing[e.source_id]; append(&out, i); g.outgoing[e.source_id] = out
		if e.target_id not_in g.incoming {g.incoming[e.target_id] = make([dynamic]int)}
		inc := g.incoming[e.target_id]; append(&inc, i); g.incoming[e.target_id] = inc
	}
}


set_descriptor :: proc(g: ^Graph, name, text: string) {
	remove_descriptor(g, name)
	d := Descriptor {
		name = strings.clone(name),
		text = strings.clone(text),
	}
	g.descriptors[d.name] = d
}

get_descriptor :: proc(g: ^Graph, name: string) -> ^Descriptor {
	if name in g.descriptors {return &g.descriptors[name]}
	return nil
}

remove_descriptor :: proc(g: ^Graph, name: string) -> bool {
	if name in g.descriptors {
		d := g.descriptors[name]
		delete(d.name)
		delete(d.text)
		delete_key(&g.descriptors, name)
		return true
	}
	return false
}

outgoing_edges :: proc(g: ^Graph, id: u64) -> []Edge {
	indices, ok := g.outgoing[id]
	if !ok || len(indices) == 0 {return {}}
	result := make([]Edge, len(indices))
	for idx, i in indices {result[i] = g.edges[idx]}
	return result
}

incoming_edges :: proc(g: ^Graph, id: u64) -> []Edge {
	indices, ok := g.incoming[id]
	if !ok || len(indices) == 0 {return {}}
	result := make([]Edge, len(indices))
	for idx, i in indices {result[i] = g.edges[idx]}
	return result
}


thought_ids_ordered :: proc(g: ^Graph) -> []u64 {
	if len(g.thoughts) == 0 {return {}}
	ids := make([dynamic]u64, 0, len(g.thoughts))
	for id in g.thoughts {append(&ids, id)}
	for i in 1 ..< len(ids) {
		j := i
		for j > 0 && ids[j] < ids[j - 1] {
			ids[j], ids[j - 1] = ids[j - 1], ids[j]
			j -= 1
		}
	}
	return ids[:]
}
