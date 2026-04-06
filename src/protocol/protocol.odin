package protocol

import "core:strings"

import "../gnn"
import "../graph"
import "../ingest"
import log "../logger"
import "../provider"
import "../util"

Handler :: struct {
	g:                    ^graph.Graph,
	p:                    ^provider.Provider,
	model:                ^gnn.MPNN,
	strand:               ^gnn.StrandMPNN,
	graph_path:           string,
	base_checkpoint_path: string,
	ingest_cfg:           ingest.Config,
	limbo:                ^graph.Graph,
}

IngestResult :: struct {
	added:    int,
	thoughts: int,
	edges:    int,
}

handle_ingest :: proc(h: ^Handler, text: string, descriptor: string = "") -> IngestResult {
	added := ingest.ingest(h.g, h.p, text, h.ingest_cfg, descriptor)

	if added > 0 {
		log.info(
			"[ingest] added %d thoughts (total: %d thoughts, %d edges)",
			added,
			graph.thought_count(h.g),
			graph.edge_count(h.g),
		)
		gnn.train_strand(h.model, h.strand, h.g, gnn.STRAND_TRAIN_STEPS)
		gnn.train_base_refine(h.model, h.strand, h.g, gnn.BASE_REFINE_STEPS)

		gnn.save_checkpoint(h.model, h.base_checkpoint_path)

		strand_bytes := gnn.strand_save_bytes(h.strand)
		if strand_bytes != nil {
			graph.save_strand(h.g, strand_bytes, h.graph_path)
			delete(strand_bytes)
		}
	}

	return IngestResult {
		added = added,
		thoughts = graph.thought_count(h.g),
		edges = graph.edge_count(h.g),
	}
}

handle_set_purpose :: proc(h: ^Handler, purpose: string) {
	graph.set_purpose(h.g, purpose)
	save_graph(h)
	log.info("[purpose] set to: \"%s\"", h.g.purpose)
}

handle_descriptor_add :: proc(h: ^Handler, name: string, text: string) {
	graph.set_descriptor(h.g, name, text)
	save_graph(h)
	log.info("[descriptor] added \"%s\" (%d bytes)", name, len(text))
}

handle_descriptor_remove :: proc(h: ^Handler, name: string) -> bool {
	ok := graph.remove_descriptor(h.g, name)
	if ok {
		save_graph(h)
		log.info("[descriptor] removed \"%s\"", name)
	}
	return ok
}

handle_descriptor_list :: proc(h: ^Handler) -> string {
	b := strings.builder_make()
	first := true
	for _, &d in h.g.descriptors {
		if !first {strings.write_byte(&b, '\n')}
		strings.write_string(&b, d.name)
		first = false
	}
	return strings.to_string(b)
}

get_descriptor_text :: proc(h: ^Handler, name: string) -> string {
	d := graph.get_descriptor(h.g, name)
	if d == nil {return ""}
	return d.text
}

handle_ask :: proc(h: ^Handler, query: string) -> (answer: string, ok: bool) {

	query_embedding, embed_ok := h.p.embed_text(h.p, query)
	if !embed_ok {
		log.warn("[ask] failed to embed query")
		return "", false
	}

	K := graph.cfg.default_find_k

	seen: map[u64]f32
	defer delete(seen)

	if graph.thought_count(h.g) >= 2 && graph.edge_count(h.g) > 0 {
		snap := gnn.build_snapshot(h.g)
		defer gnn.release_snapshot(&snap)

		gnn_emb: gnn.Embedding
		for i in 0 ..< gnn.EMBEDDING_DIM {gnn_emb[i] = query_embedding[i]}

		gnn_results := gnn.score_nodes(h.model, h.strand, &snap, &gnn_emb, K)
		defer delete(gnn_results)

		for result in gnn_results {
			existing, found := seen[result.node_id]
			if !found || result.score > existing {
				seen[result.node_id] = result.score
			}
		}
	}

	cosine_results := graph.find_thoughts(h.g, &query_embedding, K)
	defer delete(cosine_results)

	for result in cosine_results {
		existing, found := seen[result.id]
		if !found || result.score > existing {
			seen[result.id] = result.score
		}
	}

	edge_results := graph.find_edges(h.g, &query_embedding, K)
	defer delete(edge_results)

	for er in edge_results {
		edge := &h.g.edges[er.edge_index]
		edge_score := er.score * util.EDGE_SCORE_DISCOUNT

		src_existing, src_found := seen[edge.source_id]
		if !src_found || edge_score > src_existing {
			seen[edge.source_id] = edge_score
		}
		dst_existing, dst_found := seen[edge.target_id]
		if !dst_found || edge_score > dst_existing {
			seen[edge.target_id] = edge_score
		}
	}

	if len(seen) == 0 {
		log.info("[ask] no relevant thoughts found")
		return "", false
	}

	ranked := make([dynamic]graph.findResult, 0, len(seen))
	defer delete(ranked)
	for id, score in seen {
		append(&ranked, graph.findResult{id = id, score = score})
	}

	for i in 1 ..< len(ranked) {
		j := i
		for j > 0 && ranked[j].score > ranked[j - 1].score {
			ranked[j], ranked[j - 1] = ranked[j - 1], ranked[j]
			j -= 1
		}
	}

	n := min(K, len(ranked))

	context_parts: [dynamic]string
	defer delete(context_parts)

	for i in 0 ..< n {
		thought := graph.get_thought(h.g, ranked[i].id)
		if thought != nil {
			append(&context_parts, thought.text)
		}
	}

	edge_context_count := 0
	for er in edge_results {
		if edge_context_count >= graph.cfg.max_context_edges {break}
		edge := &h.g.edges[er.edge_index]
		if len(edge.reasoning) > 0 {
			append(&context_parts, edge.reasoning)
			edge_context_count += 1
		}
	}

	context_text := strings.join(context_parts[:], "\n")
	defer delete(context_text)

	result, answer_ok := h.p.generate_answer(h.p, query, context_text)
	if !answer_ok {
		log.warn("[ask] failed to generate answer")
		return "", false
	}

	log.info(
		"[ask] responded with %d bytes (sources: %d thoughts, %d edges)",
		len(result),
		n,
		len(edge_results),
	)

	return result, true
}

// save_graph persists the graph in the correct format based on the file extension.
// For .strand files, it wraps the graph data in a container with the current strand checkpoint.
// For other files, it writes the raw graph format.
@(private)
save_graph :: proc(h: ^Handler) {
	if strings.has_suffix(h.graph_path, util.STRAND_EXTENSION) {
		strand_bytes := gnn.strand_save_bytes(h.strand)
		if strand_bytes == nil {
			strand_bytes = make([]u8, 0)
		}
		graph.save_strand(h.g, strand_bytes, h.graph_path)
		delete(strand_bytes)
	} else {
		graph.save(h.g, h.graph_path)
	}
}
