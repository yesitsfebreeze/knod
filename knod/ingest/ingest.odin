package ingest


import "core:fmt"
import "core:math"
import "core:mem/virtual"
import "core:strings"
import "core:thread"
import "core:time"

import graph_pkg "../graph"
import log "../logger"
import provider_pkg "../provider"


Config :: struct {
	max_similar:      int,
	edge_threshold:   f32,
	min_link_weight:  f32,
	dedup_threshold:  f32,
}

DEFAULT_CONFIG :: Config {
	max_similar      = 5,
	edge_threshold   = 0.3,
	min_link_weight  = 0.1,
	dedup_threshold  = 0.95,
}


Prepared_Thought :: struct {
	text:            string,
	embedding:       graph_pkg.Embedding,
	candidate_ids:   [dynamic]u64,
	candidate_texts: [dynamic]string,
	links:           []provider_pkg.Batch_Link_Result,
	links_ok:        bool,
	edge_embeddings: []graph_pkg.Embedding,
	edge_emb_ok:     bool,
	is_duplicate:    bool,
}


Prepared_Article :: struct {
	thoughts:   [dynamic]Prepared_Thought,
	source:     string,
	descriptor: string,
}


prepared_article_release :: proc(pa: ^Prepared_Article) {
	for &pt in pa.thoughts {
		delete(pt.text)
		delete(pt.candidate_ids)
		for s in pt.candidate_texts {delete(s)}
		delete(pt.candidate_texts)
		if pt.links != nil {
			for &lnk in pt.links {delete(lnk.reasoning)}
			delete(pt.links)
		}
		if pt.edge_embeddings != nil {delete(pt.edge_embeddings)}
	}
	delete(pa.thoughts)
	if len(pa.source) > 0 {delete(pa.source)}
	if len(pa.descriptor) > 0 {delete(pa.descriptor)}
}


prepare :: proc(
	p: ^provider_pkg.Provider,
	text: string,
	purpose: string,
	source: string,
	descriptor: string = "",
) -> (
	result: Prepared_Article,
	ok: bool,
) {
	if len(text) == 0 {return {}, false}

	raw_thoughts, decompose_ok := p.decompose_text(p, text, purpose, descriptor)
	if !decompose_ok {
		log.warn("[ingest:prepare] LLM decomposition failed")
		return {}, false
	}
	defer delete(raw_thoughts)

	log.info("[ingest:prepare] decomposed into %d thoughts", len(raw_thoughts))

	embeddings, embed_ok := p.embed_texts(p, raw_thoughts)
	if !embed_ok {
		log.warn("[ingest:prepare] batch embedding failed")
		for s in raw_thoughts {delete(s)}
		return {}, false
	}
	defer delete(embeddings)

	prepared := make([dynamic]Prepared_Thought, 0, len(raw_thoughts))
	for thought_text, i in raw_thoughts {
		append(&prepared, Prepared_Thought{text = thought_text, embedding = embeddings[i]})
	}

	return Prepared_Article {
			thoughts = prepared,
			source = strings.clone(source),
			descriptor = len(descriptor) > 0 ? strings.clone(descriptor) : "",
		},
		true
}


snapshot :: proc(g: ^graph_pkg.Graph, pa: ^Prepared_Article, cfg: Config = DEFAULT_CONFIG) {
	for &pt in pa.thoughts {
		if graph_pkg.thought_count(g) < 1 {continue}

		results := graph_pkg.find_thoughts(
			g,
			&pt.embedding,
			cfg.max_similar + 1,
			cfg.edge_threshold,
		)
		defer delete(results)

		pt.candidate_ids = make([dynamic]u64, 0, len(results))
		pt.candidate_texts = make([dynamic]string, 0, len(results))

		for res in results {
			existing := graph_pkg.get_thought(g, res.id)
			if existing == nil {continue}
			append(&pt.candidate_ids, res.id)
			append(&pt.candidate_texts, strings.clone(existing.text))
		}
	}
	log.info("[ingest:snapshot] candidates snapshotted for %d thoughts", len(pa.thoughts))
}


// dedup checks each prepared thought against existing graph thoughts.
// If cosine similarity >= cfg.dedup_threshold, the existing thought's embedding
// is updated via running average and the prepared thought is marked as a duplicate
// (is_duplicate=true), skipping the link and commit phases for that thought.
dedup :: proc(g: ^graph_pkg.Graph, pa: ^Prepared_Article, cfg: Config = DEFAULT_CONFIG) {
	if cfg.dedup_threshold <= 0.0 || graph_pkg.thought_count(g) < 1 {return}

	merged := 0
	for &pt in pa.thoughts {
		if pt.is_duplicate {continue}

		results := graph_pkg.find_thoughts(g, &pt.embedding, 1, cfg.dedup_threshold)
		defer delete(results)

		if len(results) == 0 {continue}

		// Merge embedding into the existing thought via running average.
		existing := graph_pkg.get_thought(g, results[0].id)
		if existing == nil {continue}

		for i in 0 ..< graph_pkg.EMBEDDING_DIM {
			existing.embedding[i] = (existing.embedding[i] + pt.embedding[i]) / 2.0
		}
		// Update graph-level profile to account for this near-duplicate.
		n := f32(g.profile_count)
		for i in 0 ..< graph_pkg.EMBEDDING_DIM {
			g.profile[i] = (g.profile[i] * n + pt.embedding[i]) / (n + 1.0)
		}
		g.profile_count += 1

		pt.is_duplicate = true
		merged += 1
		log.info("[ingest:dedup] merged %.60s → id %d (sim=%.3f)", pt.text, results[0].id, results[0].score)
	}

	if merged > 0 {
		log.info("[ingest:dedup] %d/%d thoughts merged as duplicates", merged, len(pa.thoughts))
	}
}


@(private = "file")
_Link_Task :: struct {
	pt:  ^Prepared_Thought,
	p:   ^provider_pkg.Provider,
	cfg: Config,
}

@(private = "file")
_link_thought_goroutine :: proc(t: ^thread.Thread) {
	task := (^_Link_Task)(t.user_args[0])
	defer free(task)


	temp_arena: virtual.Arena
	_ = virtual.arena_init_growing(&temp_arena)
	context.temp_allocator = virtual.arena_allocator(&temp_arena)
	defer virtual.arena_destroy(&temp_arena)

	pt := task.pt
	p := task.p
	cfg := task.cfg

	if len(pt.candidate_texts) == 0 {return}

	log.info("[ingest:link] evaluating %d candidates", len(pt.candidate_texts))

	links, links_ok := p.batch_link_reason(p, pt.text, pt.candidate_texts[:])
	pt.links = links
	pt.links_ok = links_ok
	if !links_ok {
		log.warn("[ingest:link] batch_link_reason failed")
		return
	}


	valid: [dynamic]^provider_pkg.Batch_Link_Result
	defer delete(valid)
	for &lnk in pt.links {
		if lnk.index < 0 || lnk.index >= len(pt.candidate_ids) {continue}
		if lnk.weight < cfg.min_link_weight {continue}
		append(&valid, &lnk)
	}
	if len(valid) == 0 {return}

	reasoning_texts := make([]string, len(valid))
	defer delete(reasoning_texts)
	for i in 0 ..< len(valid) {reasoning_texts[i] = valid[i].reasoning}

	edge_embeddings, batch_ok := p.embed_texts(p, reasoning_texts)
	pt.edge_embeddings = edge_embeddings
	pt.edge_emb_ok = batch_ok
	if !batch_ok {log.warn("[ingest:link] edge reasoning embedding failed")}
}


link :: proc(p: ^provider_pkg.Provider, pa: ^Prepared_Article, cfg: Config = DEFAULT_CONFIG) {
	threads := make([dynamic]^thread.Thread, 0, len(pa.thoughts))
	defer delete(threads)

	for &pt in pa.thoughts {
		if pt.is_duplicate {continue}
		task := new(_Link_Task)
		task.pt = &pt
		task.p = p
		task.cfg = cfg

		t := thread.create(_link_thought_goroutine)
		t.user_args[0] = task
		t.init_context = context
		thread.start(t)
		append(&threads, t)
	}

	for t in threads {
		thread.join(t)
		thread.destroy(t)
	}
	log.info("[ingest:link] link phase complete for %d thoughts", len(pa.thoughts))
}


commit :: proc(g: ^graph_pkg.Graph, pa: ^Prepared_Article, cfg: Config = DEFAULT_CONFIG, graph_path: string = "") -> int {
	mat := graph_pkg.maturity(g)
	log.info("[ingest:commit] committing %d thoughts (maturity=%.2f)", len(pa.thoughts), mat)

	added := 0
	now := time.time_to_unix(time.now())
	src := len(pa.source) > 0 ? pa.source : "ingest"

	for &pt in pa.thoughts {
		if pt.is_duplicate {
			log.info("[ingest:commit] skipping duplicate: %.60s", pt.text)
			continue
		}

		valid_link_count := 0
		if pt.links_ok {
			for &lnk in pt.links {
				if lnk.index >= 0 &&
				   lnk.index < len(pt.candidate_ids) &&
				   lnk.weight >= cfg.min_link_weight {
					valid_link_count += 1
				}
			}
		}

		has_links := valid_link_count > 0

		if !has_links && !_should_accept(mat, false) {
			graph_pkg.add_limbo(g, pt.text, fmt.aprintf("limbo:%s", src), pt.embedding, now)
			log.info("[ingest:commit] → limbo: %.60s", pt.text)
			continue
		}
		if has_links && !_should_accept(mat, true) {
			graph_pkg.add_limbo(g, pt.text, fmt.aprintf("limbo:%s", src), pt.embedding, now)
			log.info("[ingest:commit] → limbo (linked): %.60s", pt.text)
			continue
		}

		tid := graph_pkg.add_thought(g, pt.text, src, pt.embedding, now)
		if tid == 0 {
			log.warn("[ingest:commit] add_thought returned 0 (max_thoughts reached?)")
			continue
		}
		added += 1
		log.info("[ingest:commit] added thought %d: %.60s", tid, pt.text)

		// Append to WAL if path provided.
		if len(graph_path) > 0 {
			if t := graph_pkg.get_thought(g, tid); t != nil {
				graph_pkg.append_log_thought(graph_path, t)
			}
		}

		if pt.links_ok && pt.edge_emb_ok && pt.edge_embeddings != nil {
			emb_idx := 0
			for &lnk in pt.links {
				if lnk.index < 0 || lnk.index >= len(pt.candidate_ids) {continue}
				if lnk.weight < cfg.min_link_weight {continue}
				if emb_idx >= len(pt.edge_embeddings) {continue}
				target_id := pt.candidate_ids[lnk.index]
				ok := graph_pkg.add_edge(
					g,
					tid,
					target_id,
					lnk.weight,
					lnk.reasoning,
					pt.edge_embeddings[emb_idx],
					now,
				)
				if ok && len(graph_path) > 0 {
					ei := len(g.edges) - 1
					graph_pkg.append_log_edge(graph_path, &g.edges[ei])
				}
				log.info("[ingest:commit] edge %d → %d (w=%.2f)", tid, target_id, lnk.weight)
				emb_idx += 1
			}
		}
	}

	log.info("[ingest:commit] done: %d/%d committed", added, len(pa.thoughts))
	return added
}


@(private = "file")
_should_accept :: proc(mat: f32, has_links: bool) -> bool {
	if mat <= 0.0 {return true}
	base: f32 = has_links ? 0.5 : 0.05
	p := math.pow(base, mat)

	_rng_state = _rng_state * 6364136223846793005 + 1442695040888963407
	r := f32(_rng_state >> 33) / f32(1 << 31)
	return r < p
}

@(private = "file")
_rng_state: u64 = 0xcafebabe12345678


ingest :: proc(
	g: ^graph_pkg.Graph,
	p: ^provider_pkg.Provider,
	text: string,
	source: string = "",
	descriptor: string = "",
	cfg: Config = DEFAULT_CONFIG,
	graph_path: string = "",
) -> int {
	pa, ok := prepare(p, text, g.purpose, source, descriptor)
	if !ok {return -1}
	defer prepared_article_release(&pa)

	snapshot(g, &pa, cfg)
	dedup(g, &pa, cfg)
	link(p, &pa, cfg)
	return commit(g, &pa, cfg, graph_path)
}
