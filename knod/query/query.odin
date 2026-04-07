package query

import "core:fmt"
import "core:math"
import "core:strings"
import "core:time"

import graph_pkg    "../graph"
import gnn_pkg      "../gnn"
import log          "../logger"
import provider_pkg "../provider"


Config :: struct {

	top_k:                int,

	similarity_threshold: f32,

	max_context_edges:    int,

	confidence_threshold: f32,
}

DEFAULT_CONFIG :: Config{
	top_k                = 10,
	similarity_threshold = 0.0,
	max_context_edges    = 3,
	confidence_threshold = 0.0,
}

SIMILARITY_FLOOR :: f32(0.2)


Scored_Thought :: struct {
	thought: ^graph_pkg.Thought,
	score:   f32,
}







cosine_scores :: proc(g: ^graph_pkg.Graph, query: ^graph_pkg.Embedding) -> map[u64]f32 {
	scores := make(map[u64]f32, len(g.thoughts))
	for id, &t in g.thoughts {
		scores[id] = graph_pkg.cosine_similarity(query, &t.embedding)
	}
	return scores
}









edge_scores :: proc(g: ^graph_pkg.Graph, query: ^graph_pkg.Embedding, cfg: Config = DEFAULT_CONFIG) -> map[u64]f32 {
	scores := make(map[u64]f32)
	edge_hits := graph_pkg.find_edges(g, query, cfg.top_k * 2, cfg.similarity_threshold)
	defer delete(edge_hits)

	for hit in edge_hits {
		e := &g.edges[hit.edge_index]
		tids := [2]u64{e.source_id, e.target_id}
		for tid in tids {
			if tid in g.thoughts {
				existing, _ := scores[tid]
				if hit.score > existing {
					scores[tid] = hit.score
				}
			}
		}
	}
	return scores
}









gnn_scores :: proc(
	g:     ^graph_pkg.Graph,
	mpnn:  ^gnn_pkg.MPNN,
	strand: ^gnn_pkg.StrandMPNN,
) -> map[u64]f32 {
	if mpnn == nil || graph_pkg.edge_count(g) == 0 || graph_pkg.thought_count(g) == 0 {
		return make(map[u64]f32)
	}

	snap, snap_ok := gnn_pkg.snapshot_from_graph(g)
	if !snap_ok {
		return make(map[u64]f32)
	}
	defer gnn_pkg.snapshot_release(&snap)

	cache := gnn_pkg.forward_alloc(mpnn, &snap)
	defer gnn_pkg.forward_cache_release(&cache)


	if strand != nil {
		gnn_pkg.strand_forward_query(strand, mpnn, &snap, &cache)
	}


	src := cache.relevance_scores
	if len(src) == 0 { return make(map[u64]f32) }

	s_min := src[0]; s_max := src[0]
	for i in 1 ..< len(src) {
		if src[i] < s_min { s_min = src[i] }
		if src[i] > s_max { s_max = src[i] }
	}

	result := make(map[u64]f32, len(snap.node_ids))
	for i in 0 ..< len(snap.node_ids) {
		score: f32 = 0.5
		if s_max - s_min > 1e-6 {
			score = (src[i] - s_min) / (s_max - s_min)
		}
		result[snap.node_ids[i]] = score
	}
	return result
}








merge :: proc(
	g:    ^graph_pkg.Graph,
	cos:  map[u64]f32,
	gnn:  map[u64]f32,
	edge: map[u64]f32,
	cfg:  Config = DEFAULT_CONFIG,
) -> []Scored_Thought {
	has_gnn  := len(gnn)  > 0
	has_edge := len(edge) > 0

	mat := graph_pkg.maturity(g)
	floor := SIMILARITY_FLOOR
	effective_threshold := floor + (cfg.similarity_threshold - floor) * mat

	now_unix := time.time_to_unix(time.now())
	now_f    := f32(now_unix)

	results := make([dynamic]Scored_Thought, 0, len(g.thoughts))
	defer delete(results)

	ids := graph_pkg.thought_ids_ordered(g)
	defer delete(ids)

	for tid in ids {
		t, t_ok := &g.thoughts[tid]
		if !t_ok { continue }

		c, _ := cos[tid]
		gv, _ := gnn[tid]
		ev, _ := edge[tid]


		score: f32
		if has_gnn && has_edge {
			score = 0.4*c + 0.4*gv + 0.2*ev
		} else if has_gnn {
			score = 0.5*c + 0.5*gv
		} else if has_edge {
			score = 0.7*c + 0.3*ev
		} else {
			score = c
		}


		freq_boost := math.log1p(f32(t.access_count)) * 0.02
		recency_boost: f32
		if t.last_accessed > 0 {
			age_hours := (now_f - f32(t.last_accessed)) / 3600.0
			recency_boost = 0.05 * math.exp(-age_hours / 24.0)
		}
		boost := min(freq_boost + recency_boost, 0.1)
		score += boost

		if score >= effective_threshold {
			append(&results, Scored_Thought{thought = t, score = score})
		}
	}


	for i in 1 ..< len(results) {
		j := i
		for j > 0 && results[j].score > results[j-1].score {
			results[j], results[j-1] = results[j-1], results[j]
			j -= 1
		}
	}

	n := min(cfg.top_k, len(results))
	out := make([]Scored_Thought, n)
	for i in 0 ..< n { out[i] = results[i] }
	return out
}






Dedup_Entry :: struct {
	text:    string,
	thought: ^graph_pkg.Thought,
	score:   f32,
}




deduplicate :: proc(scored_lists: [][]Scored_Thought, top_k: int) -> []Scored_Thought {
	seen := make(map[string]Scored_Thought)
	defer delete(seen)

	for lst in scored_lists {
		for &entry in lst {
			if existing, ok := seen[entry.thought.text]; !ok || entry.score > existing.score {
				seen[entry.thought.text] = entry
			}
		}
	}

	results := make([dynamic]Scored_Thought, 0, len(seen))
	defer delete(results)
	for _, v in seen { append(&results, v) }

	for i in 1 ..< len(results) {
		j := i
		for j > 0 && results[j].score > results[j-1].score {
			results[j], results[j-1] = results[j-1], results[j]
			j -= 1
		}
	}

	n := min(top_k, len(results))
	out := make([]Scored_Thought, n)
	for i in 0 ..< n { out[i] = results[i] }
	return out
}






Answer_Source :: struct {
	id:         u64,
	text:       string,
	score:      f32,
	source:     string,
}



Answer_Result :: struct {
	answer_text: string,
	sources:     []Answer_Source,
}




answer :: proc(
	g:      ^graph_pkg.Graph,
	p:      ^provider_pkg.Provider,
	query:  string,
	scored: []Scored_Thought,
) -> (result: Answer_Result, ok: bool) {
	if len(scored) == 0 {
		text, gen_ok := p.generate_answer(p, query, "")
		return Answer_Result{answer_text = text, sources = make([]Answer_Source, 0)}, gen_ok
	}

	ctx_builder := strings.builder_make()
	defer strings.builder_destroy(&ctx_builder)

	sources := make([]Answer_Source, len(scored))
	now := time.time_to_unix(time.now())

	for st, i in scored {

		st.thought.access_count += 1
		st.thought.last_accessed = now

		if i > 0 { strings.write_string(&ctx_builder, "\n\n") }
		strings.write_string(&ctx_builder, st.thought.text)

		sources[i] = Answer_Source{
			id     = st.thought.id,
			text   = st.thought.text,
			score  = st.score,
			source = st.thought.source,
		}
	}

	context_text := strings.to_string(ctx_builder)
	answer_text, gen_ok := p.generate_answer(p, query, context_text)
	if !gen_ok {
		delete(sources)
		return {}, false
	}

	return Answer_Result{answer_text = answer_text, sources = sources}, true
}







retrieve :: proc(
	g:      ^graph_pkg.Graph,
	query:  ^graph_pkg.Embedding,
	mpnn:   ^gnn_pkg.MPNN,
	strand: ^gnn_pkg.StrandMPNN,
	cfg:    Config = DEFAULT_CONFIG,
) -> []Scored_Thought {
	cos  := cosine_scores(g, query)
	defer delete(cos)

	gnn  := gnn_scores(g, mpnn, strand)
	defer delete(gnn)

	edge := edge_scores(g, query, cfg)
	defer delete(edge)

	return merge(g, cos, gnn, edge, cfg)
}
