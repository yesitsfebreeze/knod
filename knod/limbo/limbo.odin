package limbo


import "core:fmt"
import "core:path/filepath"
import "core:strings"
import "core:time"

import gnn_pkg "../gnn"
import graph_pkg "../graph"
import ingest_pkg "../ingest"
import log "../logger"
import provider_pkg "../provider"
import registry_pkg "../registry"
import util_pkg "../util"


Config :: struct {
	cluster_min:                int,
	cluster_threshold:          f32,
	specialist_match_threshold: f32,
}

DEFAULT_CONFIG :: Config {
	cluster_min                = 3,
	cluster_threshold          = 0.75,
	specialist_match_threshold = 0.8,
}


Specialist_Profile :: struct {
	name:    string,
	profile: ^graph_pkg.Embedding,
}


scan :: proc(
	limbo_g: ^graph_pkg.Graph,
	limbo_path: string,
	p: ^provider_pkg.Provider,
	specialists: []Specialist_Profile,
	cfg: Config = DEFAULT_CONFIG,
	ingest_cfg: ingest_pkg.Config = ingest_pkg.DEFAULT_CONFIG,
	on_spawn: proc(name, path, purpose: string) = nil,
) -> int {
	if p == nil || p.suggest_store == nil {return 0}
	if graph_pkg.limbo_count(limbo_g) < cfg.cluster_min {return 0}

	clusters := find_clusters(limbo_g, cfg)
	defer {
		for &cl in clusters {delete(cl)}
		delete(clusters)
	}

	if len(clusters) == 0 {return 0}

	total_promoted := 0

	for &cluster in clusters {
		n := promote_cluster(limbo_g, limbo_path, cluster[:], specialists, p, cfg, ingest_cfg, on_spawn)
		total_promoted += n
	}

	return total_promoted
}


find_clusters :: proc(g: ^graph_pkg.Graph, cfg: Config) -> [dynamic][]int {
	n := len(g.limbo)
	if n < cfg.cluster_min {
		return make([dynamic][]int)
	}


	sim := make([]f32, n * n)
	defer delete(sim)

	for i in 0 ..< n {
		for j in 0 ..< n {
			if i == j {
				sim[i * n + j] = 1.0
				continue
			}
			s := graph_pkg.cosine_similarity(&g.limbo[i].embedding, &g.limbo[j].embedding)
			sim[i * n + j] = s
			sim[j * n + i] = s
		}
	}

	visited := make([]bool, n)
	defer delete(visited)

	result := make([dynamic][]int)

	for i in 0 ..< n {
		if visited[i] {continue}
		cluster := make([dynamic]int)
		append(&cluster, i)
		visited[i] = true

		stack := make([dynamic]int)
		defer delete(stack)
		append(&stack, i)

		for len(stack) > 0 {
			current := stack[len(stack) - 1]
			pop(&stack)
			for j in 0 ..< n {
				if !visited[j] && sim[current * n + j] >= cfg.cluster_threshold {
					visited[j] = true
					append(&cluster, j)
					append(&stack, j)
				}
			}
		}

		if len(cluster) >= cfg.cluster_min {
			append(&result, cluster[:])
		} else {
			delete(cluster)
		}
	}

	return result
}


@(private = "file")
promote_cluster :: proc(
	limbo_g: ^graph_pkg.Graph,
	limbo_path: string,
	indices: []int,
	specialists: []Specialist_Profile,
	p: ^provider_pkg.Provider,
	cfg: Config,
	ingest_cfg: ingest_pkg.Config,
	on_spawn: proc(name, path, purpose: string),
) -> int {

	texts := make([]string, len(indices))
	defer delete(texts)
	for i, slot in indices {texts[i] = limbo_g.limbo[slot].text}

	store_name, purpose, ok := p.suggest_store(p, texts)
	if !ok {
		log.warn("[limbo] suggest_store failed, skipping cluster")
		return 0
	}
	defer delete(store_name)
	defer delete(purpose)


	best_name := ""
	best_sim := f32(0.0)

	if len(specialists) > 0 {
		purpose_emb, emb_ok := p.embed_text(p, purpose)
		if emb_ok {
			for &sp in specialists {
				if sp.profile == nil {continue}
				sim := graph_pkg.cosine_similarity(&purpose_emb, sp.profile)
				if sim > best_sim {
					best_sim = sim
					best_name = sp.name
				}
			}
		}
	}

	if best_sim >= cfg.specialist_match_threshold && len(best_name) > 0 {


		log.info(
			"[limbo] promoting %d thoughts → existing specialist '%s' (sim=%.2f)",
			len(indices),
			best_name,
			best_sim,
		)

		sorted := make([]int, len(indices))
		copy(sorted, indices)
		_sort_desc(sorted)
		graph_pkg.remove_limbo_indices(limbo_g, sorted)
		delete(sorted)
		graph_pkg.save(limbo_g, limbo_path)
		if on_spawn != nil {on_spawn(best_name, "", purpose)}
		return len(indices)
	}


	return _spawn_specialist(limbo_g, limbo_path, indices, store_name, purpose, p, ingest_cfg, on_spawn)
}

@(private = "file")
_spawn_specialist :: proc(
	limbo_g: ^graph_pkg.Graph,
	limbo_path: string,
	indices: []int,
	name: string,
	purpose: string,
	p: ^provider_pkg.Provider,
	ingest_cfg: ingest_pkg.Config,
	on_spawn: proc(name, path, purpose: string),
) -> int {
	safe_name := _safe_store_name(name)
	defer delete(safe_name)

	graph_path := _store_path(safe_name)
	defer delete(graph_path)


	dir := filepath.dir(graph_path)
	defer delete(dir)
	util_pkg.ensure_dir(dir)

	new_g: graph_pkg.Graph
	graph_pkg.create(&new_g)
	defer graph_pkg.release(&new_g)
	graph_pkg.set_purpose(&new_g, purpose)

	now := time.time_to_unix(time.now())
	for slot in indices {
		lt := &limbo_g.limbo[slot]
		src := fmt.aprintf("limbo-promote:%d", now)
		defer delete(src)
		graph_pkg.add_thought(&new_g, lt.text, src, lt.embedding, now)
	}

	// Bootstrap: build a Prepared_Article from the limbo thoughts (already embedded)
	// and run the link phase to create edges, then GNN training.
	if graph_pkg.thought_count(&new_g) >= 2 {
		pa: ingest_pkg.Prepared_Article
		pa.thoughts = make([dynamic]ingest_pkg.Prepared_Thought, 0, len(indices))
		pa.source = "limbo-bootstrap"

		for slot in indices {
			lt := &limbo_g.limbo[slot]
			append(&pa.thoughts, ingest_pkg.Prepared_Thought{
				text      = lt.text,
				embedding = lt.embedding,
			})
		}

		ingest_pkg.snapshot(&new_g, &pa, ingest_cfg)
		ingest_pkg.link(p, &pa, ingest_cfg)

		// Commit only edges (thoughts already added above).
		for &pt in pa.thoughts {
			if !pt.links_ok || pt.edge_embeddings == nil {continue}
			// Find the thought id matching this text.
			tid_match: u64 = 0
			for tid, &t in new_g.thoughts {
				if t.text == pt.text {
					tid_match = tid
					break
				}
			}
			if tid_match == 0 {continue}
			emb_idx := 0
			for &lnk in pt.links {
				if lnk.index < 0 || lnk.index >= len(pt.candidate_ids) {continue}
				if lnk.weight < ingest_cfg.min_link_weight {continue}
				if emb_idx >= len(pt.edge_embeddings) {continue}
				target_id := pt.candidate_ids[lnk.index]
				graph_pkg.add_edge(
					&new_g, tid_match, target_id, lnk.weight, lnk.reasoning,
					pt.edge_embeddings[emb_idx], now,
				)
				emb_idx += 1
			}
		}

		// Cleanup pa (we borrowed text/embedding pointers, don't free them).
		for &pt in pa.thoughts {
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

		// GNN training on the new specialist graph.
		if graph_pkg.edge_count(&new_g) > 0 {
			model: gnn_pkg.MPNN
			gnn_pkg.create(&model)
			defer gnn_pkg.release(&model)

			strand: gnn_pkg.StrandMPNN
			gnn_pkg.strand_create(&strand, gnn_pkg.DEFAULT_HIDDEN_DIM)
			defer gnn_pkg.strand_release(&strand)

			n := graph_pkg.thought_count(&new_g)
			strand_steps := gnn_pkg.adaptive_steps(n, gnn_pkg.STRAND_TRAIN_STEPS_MAX, gnn_pkg.STRAND_TRAIN_STEPS_MIN)
			base_steps   := gnn_pkg.adaptive_steps(n, gnn_pkg.BASE_REFINE_STEPS_MAX, gnn_pkg.BASE_REFINE_STEPS_MIN)
			gnn_pkg.train_strand(&model, &strand, &new_g, strand_steps)
			gnn_pkg.train_base_refine(&model, &strand, &new_g, base_steps)

			strand_bytes := gnn_pkg.strand_save_bytes(&strand)
			if strand_bytes != nil {
				graph_pkg.save_with_strand(&new_g, graph_path, strand_bytes)
				delete(strand_bytes)
			} else {
				graph_pkg.save(&new_g, graph_path)
			}
			log.info("[limbo] bootstrap: %d edges, GNN trained for '%s'", graph_pkg.edge_count(&new_g), safe_name)
		} else {
			if !graph_pkg.save(&new_g, graph_path) {
				log.err("[limbo] failed to save new specialist at %s", graph_path)
				return 0
			}
		}
	} else {
		if !graph_pkg.save(&new_g, graph_path) {
			log.err("[limbo] failed to save new specialist at %s", graph_path)
			return 0
		}
	}


	reg, reg_ok := registry_pkg.load()
	if reg_ok {
		registry_pkg.add_store(&reg, safe_name, graph_path)
		registry_pkg.save(&reg)
		registry_pkg.release(&reg)
	}

	log.info(
		"[limbo] spawned new specialist '%s' at %s (%d thoughts)",
		safe_name,
		graph_path,
		len(indices),
	)


	sorted := make([]int, len(indices))
	copy(sorted, indices)
	_sort_desc(sorted)
	graph_pkg.remove_limbo_indices(limbo_g, sorted)
	delete(sorted)
	graph_pkg.save(limbo_g, limbo_path)

	if on_spawn != nil {on_spawn(safe_name, graph_path, purpose)}
	return len(indices)
}


@(private = "file")
_safe_store_name :: proc(name: string) -> string {
	b := strings.builder_make()
	for ch in name {
		switch {
		case ch >= 'a' && ch <= 'z':
			strings.write_rune(&b, ch)
		case ch >= 'A' && ch <= 'Z':
			strings.write_rune(&b, rune(ch - 'A' + 'a'))
		case ch >= '0' && ch <= '9':
			strings.write_rune(&b, ch)
		case ch == ' ' || ch == '-' || ch == '_':
			strings.write_rune(&b, '-')
		}
	}
	s := strings.to_string(b)
	if len(s) == 0 {s = "specialist"}
	return s
}

@(private = "file")
_store_path :: proc(name: string) -> string {
	home := util_pkg.home_dir()
	if len(home) == 0 {
		return fmt.aprintf("%s.strand", name)
	}
	defer delete(home)
	return filepath.join({home, ".config", "knod", "data", fmt.aprintf("%s.strand", name)})
}

@(private = "file")
_sort_desc :: proc(a: []int) {
	for i in 1 ..< len(a) {
		j := i
		for j > 0 && a[j] > a[j - 1] {
			a[j], a[j - 1] = a[j - 1], a[j]
			j -= 1
		}
	}
}
