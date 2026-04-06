package ingest

import "core:fmt"
import "core:math"
import "core:os"
import "core:path/filepath"
import "core:strings"
import "core:thread"
import "core:time"

import "../graph"
import log "../logger"
import "../provider"
import "../registry"
import "../util"

Config :: struct {
	max_similar:        int,
	edge_threshold:     f32,
	maturity_threshold: int,
	max_tags:           int,
	min_link_weight:    f32,
	limbo_graph:        ^graph.Graph,
	limbo_cluster_min:  int,
	limbo_threshold:    f32,
}

DEFAULT_CONFIG :: Config {
	max_similar        = 5,
	edge_threshold     = 0.3,
	maturity_threshold = 50,
	max_tags           = 128,
	min_link_weight    = 0.1,
	limbo_graph        = nil,
	limbo_cluster_min  = 3,
	limbo_threshold    = 0.75,
}

// A single thought that has been decomposed and embedded but not yet
// linked into the graph.  Created by ingest_prepare, consumed by ingest_commit.
//
// After ingest_snapshot (phase 1.5) the candidate_* fields are populated with
// a snapshot of the nearest neighbours at snapshot time.
// After ingest_link (phase 1.75 — still no mutex) the links / edge_embeddings
// fields carry the pre-computed LLM results ready for ingest_commit.
Prepared_Thought :: struct {
	text:            string, // cloned — caller must free
	embedding:       graph.Embedding,

	// Populated by ingest_snapshot (brief mutex, graph read-only).
	candidate_ids:   [dynamic]u64,
	candidate_texts: [dynamic]string, // cloned from graph

	// Populated by ingest_link (no mutex, LLM calls).
	links:           []provider.Batch_Link_Result, // nil if no candidates / failed
	links_ok:        bool,
	edge_embeddings: []graph.Embedding,            // parallel to valid links
	edge_emb_ok:     bool,
}

// The result of the "prepare" phase for a single article.
// Contains all decomposed+embedded thoughts, ready for graph insertion.
Prepared_Article :: struct {
	thoughts:   [dynamic]Prepared_Thought,
	descriptor: string, // cloned — caller must free (empty string if none)
}

prepared_article_release :: proc(pa: ^Prepared_Article) {
	for &pt in pa.thoughts {
		delete(pt.text)
		delete(pt.candidate_ids)
		for s in pt.candidate_texts {delete(s)}
		delete(pt.candidate_texts)
		if pt.links != nil {
			for &link in pt.links {delete(link.reasoning)}
			delete(pt.links)
		}
		if pt.edge_embeddings != nil {delete(pt.edge_embeddings)}
	}
	delete(pa.thoughts)
	if len(pa.descriptor) > 0 {delete(pa.descriptor)}
}

// ingest_prepare runs the LLM-heavy portion of ingestion that does NOT touch
// the graph: decompose text into thoughts and embed each one.
// This is safe to call from multiple threads concurrently.
// Returns nil thoughts slice on failure.
ingest_prepare :: proc(
	p: ^provider.Provider,
	text: string,
	purpose: string,
	descriptor: string = "",
) -> (result: Prepared_Article, ok: bool) {
	if len(text) == 0 {
		return {}, false
	}

	raw_thoughts, decompose_ok := p.decompose_text(p, text, purpose, descriptor)
	if !decompose_ok {
		log.warn("[ingest:prepare] failed to decompose text")
		return {}, false
	}
	defer delete(raw_thoughts)

	log.info("[ingest:prepare] decomposed into %d candidate thoughts", len(raw_thoughts))

	embeddings, embed_ok := p.embed_texts(p, raw_thoughts)
	if !embed_ok {
		log.warn("[ingest:prepare] failed to batch-embed thoughts")
		for s in raw_thoughts {delete(s)}
		return {}, false
	}
	defer delete(embeddings)

	prepared := make([dynamic]Prepared_Thought, 0, len(raw_thoughts))

	for thought_text, i in raw_thoughts {
		append(&prepared, Prepared_Thought{
			text      = thought_text, // take ownership — raw_thoughts slice freed, strings kept
			embedding = embeddings[i],
		})
	}

	log.info("[ingest:prepare] prepared %d thoughts (batch-embedded)", len(prepared))

	desc := len(descriptor) > 0 ? strings.clone(descriptor) : ""
	return Prepared_Article{thoughts = prepared, descriptor = desc}, true
}

// ingest_snapshot is a brief, graph-READ-ONLY pass that must be called under
// the handler mutex.  For each prepared thought it finds the nearest existing
// neighbours and clones their IDs and texts into the Prepared_Thought so that
// the subsequent ingest_link phase can call the LLM without holding any mutex.
ingest_snapshot :: proc(g: ^graph.Graph, pa: ^Prepared_Article, c: Config = DEFAULT_CONFIG) {
	for &pt in pa.thoughts {
		if graph.thought_count(g) < 1 {continue}

		results := graph.find_thoughts(g, &pt.embedding, c.max_similar + 1)
		defer delete(results)

		pt.candidate_ids   = make([dynamic]u64,   0, len(results))
		pt.candidate_texts = make([dynamic]string, 0, len(results))

		for result in results {
			if result.score < c.edge_threshold {continue}
			existing := graph.get_thought(g, result.id)
			if existing == nil {continue}
			append(&pt.candidate_ids,   result.id)
			append(&pt.candidate_texts, strings.clone(existing.text))
		}
	}
	log.info("[ingest:snapshot] snapshotted candidates for %d thoughts", len(pa.thoughts))
}

// _Link_Task is the per-thought context passed to each link goroutine.
@(private = "file")
_Link_Task :: struct {
	pt:  ^Prepared_Thought,
	p:   ^provider.Provider,
	cfg: Config,
}

@(private = "file")
_link_thought_goroutine :: proc(t: ^thread.Thread) {
	task := (^_Link_Task)(t.user_args[0])
	defer free(task)

	pt  := task.pt
	p   := task.p
	cfg := task.cfg

	if len(pt.candidate_texts) == 0 {return}

	log.info("[ingest:link] evaluating %d candidates for thought", len(pt.candidate_texts))
	links, links_ok := p.batch_link_reason(p, pt.text, pt.candidate_texts[:])
	pt.links    = links
	pt.links_ok = links_ok
	if !links_ok {
		log.warn("[ingest:link] batch_link_reason failed, treating as zero connections")
		return
	}

	// Pre-embed edge reasonings for valid links (also LLM, also no mutex).
	valid_links: [dynamic]^provider.Batch_Link_Result
	defer delete(valid_links)
	for &link in pt.links {
		if link.index < 0 || link.index >= len(pt.candidate_ids) {continue}
		if link.weight < cfg.min_link_weight {continue}
		append(&valid_links, &link)
	}

	if len(valid_links) == 0 {return}

	reasoning_texts := make([]string, len(valid_links))
	defer delete(reasoning_texts)
	for i in 0 ..< len(valid_links) {
		reasoning_texts[i] = valid_links[i].reasoning
	}

	edge_embeddings, batch_ok := p.embed_texts(p, reasoning_texts)
	pt.edge_embeddings = edge_embeddings
	pt.edge_emb_ok     = batch_ok
	if !batch_ok {
		log.warn("[ingest:link] failed to batch-embed edge reasonings")
	}
}

// ingest_link runs the LLM-heavy link-reasoning for all thoughts in pa IN PARALLEL.
// One goroutine is spawned per thought; all are joined before returning.
// Uses only pre-snapshotted candidate_texts — no graph access at all.
// Safe to call from multiple article goroutines concurrently.
ingest_link :: proc(p: ^provider.Provider, pa: ^Prepared_Article, c: Config = DEFAULT_CONFIG) {
	threads := make([dynamic]^thread.Thread, 0, len(pa.thoughts))
	defer delete(threads)

	for &pt in pa.thoughts {
		task := new(_Link_Task)
		task.pt  = &pt
		task.p   = p
		task.cfg = c

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

	log.info("[ingest:link] link phase complete for %d thoughts (parallel)", len(pa.thoughts))
}

// ingest_commit takes prepared+snapshotted+linked thoughts and writes them
// into the graph.  This phase ONLY does graph mutations — no LLM calls.
// Must be called under the handler mutex.
ingest_commit :: proc(
	g: ^graph.Graph,
	p: ^provider.Provider,
	pa: ^Prepared_Article,
	c: Config = DEFAULT_CONFIG,
) -> int {
	maturity := graph_maturity(g, c)
	log.info(
		"[ingest:commit] committing %d thoughts (graph maturity: %.2f)",
		len(pa.thoughts),
		maturity,
	)

	added := 0

	for &pt in pa.thoughts {
		// Count valid links from the pre-computed LLM results.
		valid_link_count := 0
		if pt.links_ok {
			for &link in pt.links {
				if link.index >= 0 &&
				   link.index < len(pt.candidate_ids) &&
				   link.weight >= c.min_link_weight {
					valid_link_count += 1
				}
			}
		}

		if valid_link_count == 0 && !should_keep_unconnected(g, c) {
			log.info("[ingest:commit] unconnected thought (graph mature): %.60s...", pt.text)
			if c.limbo_graph != nil {
				now := time.time_to_unix(time.now())
				src := fmt.aprintf("limbo:%d", now)
				defer delete(src)
				log.info("[ingest:commit] routing to limbo (current limbo size: %d)", graph.thought_count(c.limbo_graph))
				graph.add_thought(c.limbo_graph, pt.text, src, pt.embedding, now)
				log.info(
					"[ingest:commit] thought routed to limbo (%d total)",
					graph.thought_count(c.limbo_graph),
				)
			} else {
				log.info("[ingest:commit] dropping thought (no limbo configured)")
			}
			continue
		}

		now := time.time_to_unix(time.now())
		source_id := fmt.tprintf("ingest:%d", now)
		thought_id := graph.add_thought(g, pt.text, source_id, pt.embedding, now)
		added += 1

		log.info("[ingest:commit] added thought %d: %.60s...", thought_id, pt.text)

		if pt.links_ok && pt.edge_emb_ok && pt.edge_embeddings != nil {
			emb_idx := 0
			for &link in pt.links {
				if link.index < 0 || link.index >= len(pt.candidate_ids) {continue}
				if link.weight < c.min_link_weight {continue}
				if emb_idx >= len(pt.edge_embeddings) {continue}
				target_id := pt.candidate_ids[link.index]
				graph.add_edge(
					g,
					thought_id,
					target_id,
					link.weight,
					link.reasoning,
					pt.edge_embeddings[emb_idx],
					now,
				)
				log.info(
					"[ingest:commit] added edge %d -> %d (weight=%.2f)",
					thought_id,
					target_id,
					link.weight,
				)
				emb_idx += 1
			}
		}
	}

	log.info("[ingest:commit] commit complete: %d thoughts added", added)

	if added > 0 {
		update_tags(g, p, c)
	}

	return added
}

@(private = "file")
graph_maturity :: proc(g: ^graph.Graph, c: Config) -> f32 {
	if c.maturity_threshold <= 0 {return 1.0}
	fill := f32(graph.thought_count(g)) / f32(c.maturity_threshold)
	return min(fill, 1.0)
}

@(private = "file")
should_keep_unconnected :: proc(g: ^graph.Graph, c: Config) -> bool {
	maturity := graph_maturity(g, c)
	if maturity < 0.1 {return true}
	acceptance := 1.0 * math.pow(f32(0.05), maturity)
	return acceptance > 0.5
}


// ingest is the original single-phase entry point.  It runs prepare + snapshot
// + link + commit sequentially, holding no external mutex during LLM calls.
// Kept for backward compatibility and non-queued callers.
// NOTE: The caller must ensure the graph is not mutated concurrently between
// snapshot and commit (i.e. this is not safe to call from multiple threads
// against the same graph without external locking).
ingest :: proc(
	g: ^graph.Graph,
	p: ^provider.Provider,
	text: string,
	c: Config = DEFAULT_CONFIG,
	descriptor: string = "",
) -> int {
	if len(text) == 0 {
		return 0
	}

	pa, ok := ingest_prepare(p, text, g.purpose, descriptor)
	if !ok {
		return -1
	}
	defer prepared_article_release(&pa)

	ingest_snapshot(g, &pa, c)
	ingest_link(p, &pa, c)
	return ingest_commit(g, p, &pa, c)
}

@(private = "file")
Thought_Entry :: struct {
	text:  string,
	value: f32,
}

@(private = "file")
update_tags :: proc(g: ^graph.Graph, p: ^provider.Provider, c: Config) {
	dims := graph.top_dimensions(g, c.max_tags)
	if len(dims) == 0 {return}
	defer delete(dims)

	new_dims := graph.changed_dimensions(g, dims)
	if len(new_dims) == 0 {return}
	defer delete(new_dims)

	log.info("[ingest] %d new tag dimensions to label", len(new_dims))

	dim_thoughts := make([][]string, len(new_dims))
	defer {
		for &group in dim_thoughts {
			delete(group)
		}
		delete(dim_thoughts)
	}

	for nd, i in new_dims {
		entries := make([dynamic]Thought_Entry, 0, graph.thought_count(g))
		defer delete(entries)

		for _, &t in g.thoughts {
			val := t.embedding[nd]
			if val > 0.0 {
				append(&entries, Thought_Entry{text = t.text, value = val})
			}
		}

		for j in 1 ..< len(entries) {
			k := j
			for k > 0 && entries[k].value > entries[k - 1].value {
				entries[k], entries[k - 1] = entries[k - 1], entries[k]
				k -= 1
			}
		}

		take := min(5, len(entries))
		result := make([]string, take)
		for j in 0 ..< take {
			result[j] = entries[j].text
		}
		dim_thoughts[i] = result
	}

	if p.label_dimensions == nil {
		log.warn("[ingest] provider does not support label_dimensions, skipping tag update")
		return
	}

	LABEL_BATCH_SIZE :: 16

	all_labels := make([dynamic]string, 0, len(new_dims))
	defer {
		for &l in all_labels {
			delete(l)
		}
		delete(all_labels)
	}

	batch_failed := false
	for start := 0; start < len(new_dims); start += LABEL_BATCH_SIZE {
		end := min(start + LABEL_BATCH_SIZE, len(new_dims))
		batch := dim_thoughts[start:end]
		batch_labels, ok := p.label_dimensions(p, g.purpose, batch)
		if !ok {
			log.warn("[ingest] failed to label dimensions (batch %d-%d)", start, end)
			batch_failed = true
			break
		}
		for l in batch_labels {
			append(&all_labels, l)
		}
		delete(batch_labels)
	}

	if batch_failed {
		return
	}

	labels := all_labels[:]

	existing := graph.get_tags(g)
	merged := make([dynamic]graph.Tag, 0, len(existing) + len(new_dims))
	defer delete(merged)

	consumed := make([]bool, len(new_dims))
	defer delete(consumed)

	for &tag in existing {
		still_top := false
		for &dim in dims {
			if dim.index == tag.dim_index {
				still_top = true
				break
			}
		}
		if still_top {
			relabeled := false
			for nd, i in new_dims {
				if nd == tag.dim_index {
					if i < len(labels) {
						append(&merged, graph.Tag{dim_index = nd, label = labels[i]})
					} else {
						append(&merged, graph.Tag{dim_index = tag.dim_index, label = tag.label})
					}
					consumed[i] = true
					relabeled = true
					break
				}
			}
			if !relabeled {
				append(&merged, graph.Tag{dim_index = tag.dim_index, label = tag.label})
			}
		}
	}

	for nd, i in new_dims {
		if consumed[i] {continue}
		if i >= len(labels) {continue}
		append(&merged, graph.Tag{dim_index = nd, label = labels[i]})
	}

	graph.set_tags(g, merged[:])

	log.info("[ingest] updated tags: %d total", len(g.tags))
}

scan_limbo :: proc(
	limbo_g: ^graph.Graph,
	limbo_path: string,
	p: ^provider.Provider,
	min_cluster: int,
	threshold: f32,
) {
	if limbo_g == nil || min_cluster <= 0 {
		return
	}
	if graph.thought_count(limbo_g) < min_cluster {
		return
	}
	if p == nil || p.suggest_store == nil {
		return
	}

	cluster, found := find_cluster(limbo_g, min_cluster, threshold)
	if !found {
		return
	}
	defer delete(cluster)

	texts := make([dynamic]string, 0, len(cluster))
	defer delete(texts)
	embeddings := make([dynamic]graph.Embedding, 0, len(cluster))
	defer delete(embeddings)

	for id in cluster {
		t := graph.get_thought(limbo_g, id)
		if t != nil {
			append(&texts, t.text)
			append(&embeddings, t.embedding)
		}
	}
	if len(texts) == 0 {
		return
	}

	store_name, store_purpose, ok := p.suggest_store(p, texts[:])
	if !ok {
		log.warn("[limbo] suggest_store failed, skipping promotion")
		return
	}
	defer delete(store_name)
	defer delete(store_purpose)

	log.info("[limbo] promoting cluster of %d thoughts as '%s'", len(texts), store_name)

	final_name, final_path := unique_store_name(store_name)
	defer delete(final_name)
	defer delete(final_path)

	new_g: graph.Graph
	graph.create(&new_g)
	defer graph.release(&new_g)
	graph.set_purpose(&new_g, store_purpose)

	dir := filepath.dir(final_path)
	defer delete(dir)
	ensure_dir(dir)

	if !graph.save(&new_g, final_path) {
		log.err("[limbo] could not create store file at %s", final_path)
		return
	}
	if !graph.stream_open(&new_g, final_path) {
		log.err("[limbo] could not open stream for new store at %s", final_path)
		return
	}

	now := time.time_to_unix(time.now())
	for i in 0 ..< len(texts) {
		src := fmt.tprintf("limbo-promote:%d", now)
		graph.add_thought(&new_g, texts[i], src, embeddings[i], now)
	}
	if !graph.save(&new_g, final_path) {
		log.warn("[limbo] could not persist promoted store at %s", final_path)
	}

	reg, _ := registry.load()
	defer registry.release(&reg)
	registry.add_store(&reg, final_name, final_path)
	if !registry.save(&reg) {
		log.warn("[limbo] could not update registry with new store '%s'", final_name)
	}

	log.info("[limbo] created store '%s' at %s", final_name, final_path)

	remove_thoughts(limbo_g, cluster[:])
	if !graph.save(limbo_g, limbo_path) {
		log.warn("[limbo] could not rewrite limbo file after promotion")
	}
	if limbo_g.stream_handle != os.INVALID_HANDLE {
		os.close(limbo_g.stream_handle)
		limbo_g.stream_handle = os.INVALID_HANDLE
	}
	if !graph.stream_open(limbo_g, limbo_path) {
		log.warn("[limbo] could not reopen limbo stream after promotion")
	}

	log.info(
		"[limbo] promoted %d thoughts to '%s'; limbo now has %d thoughts",
		len(texts),
		final_name,
		graph.thought_count(limbo_g),
	)
}

@(private = "file")
find_cluster :: proc(g: ^graph.Graph, min_size: int, threshold: f32) -> ([]u64, bool) {
	if graph.thought_count(g) < min_size {
		return nil, false
	}

	best: [dynamic]u64
	best_len := 0

	for seed_id, &seed_t in g.thoughts {
		members := make([dynamic]u64, 0, min_size)
		append(&members, seed_id)

		for other_id, &other_t in g.thoughts {
			if other_id == seed_id {continue}
			sim := graph.cosine_similarity(&seed_t.embedding, &other_t.embedding)
			if sim >= threshold {
				append(&members, other_id)
			}
		}

		if len(members) >= min_size && len(members) > best_len {
			delete(best)
			best = members
			best_len = len(members)
		} else {
			delete(members)
		}
	}

	if best_len >= min_size {
		return best[:], true
	}
	return nil, false
}

@(private = "file")
remove_thoughts :: proc(g: ^graph.Graph, ids: []u64) {
	for id in ids {
		if t, ok := g.thoughts[id]; ok {
			delete(t.text)
			delete(t.source_id)
			delete_key(&g.thoughts, id)
		}
		if list, ok := g.outgoing[id]; ok {
			delete(list)
			delete_key(&g.outgoing, id)
		}
		if list, ok := g.incoming[id]; ok {
			delete(list)
			delete_key(&g.incoming, id)
		}
	}

	g.profile = {}
	g.profile_count = 0
	for _, &t in g.thoughts {
		n := f32(g.profile_count)
		for i in 0 ..< graph.EMBEDDING_DIM {
			g.profile[i] = (g.profile[i] * n + t.embedding[i]) / (n + 1.0)
		}
		g.profile_count += 1
	}
}

@(private = "file")
unique_store_name :: proc(base: string) -> (name: string, path: string) {
	reg, _ := registry.load()
	defer registry.release(&reg)

	try_name :: proc(reg: ^registry.Registry, n: string) -> (string, string, bool) {
		p := store_path_for(n)
		if registry.find_store(reg, n) == nil && !os.exists(p) {
			return strings.clone(n), p, true
		}
		delete(p)
		return "", "", false
	}

	if n, p, ok := try_name(&reg, base); ok {
		return n, p
	}
	for i := 2; i < 1000; i += 1 {
		candidate := fmt.tprintf("%s-%d", base, i)
		if n, p, ok := try_name(&reg, candidate); ok {
			return n, p
		}
	}
	ts := fmt.tprintf("%s-%d", base, time.time_to_unix(time.now()))
	return strings.clone(ts), store_path_for(ts)
}

@(private = "file")
store_path_for :: proc(name: string) -> string {
	home := home_dir_ingest()
	if len(home) == 0 {
		return strings.clone(fmt.tprintf("%s.graph", name))
	}
	defer delete(home)
	return filepath.join({home, ".config", "knod", "data", fmt.tprintf("%s.graph", name)})
}

@(private = "file")
home_dir_ingest :: proc() -> string {
	return util.home_dir()
}

@(private = "file")
ensure_dir :: proc(path: string) {
	util.ensure_dir(path)
}
