package protocols

// handler.odin — core Handler struct and command dispatch procs.
//
// Adapts archive/src/protocol/protocol.odin to the new package structure:
//   - Uses knod/ingest (prepare/snapshot/link/commit, no tags)
//   - Uses knod/query  (retrieve + answer)
//   - Uses knod/graph  (save/save_with_strand)
//   - No limbo field on Handler — limbo lives inside graph.Graph

import "core:encoding/endian"
import "core:fmt"
import "core:net"
import "core:strings"
import "core:sync"
import "core:time"

import gnn_pkg    "../gnn"
import graph_pkg  "../graph"
import ingest_pkg "../ingest"
import log        "../logger"
import prov       "../provider"
import query_pkg  "../query"
import util       "../util"

Handler :: struct {
	g:                       ^graph_pkg.Graph,
	p:                       ^prov.Provider,
	model:                   ^gnn_pkg.MPNN,
	strand:                  ^gnn_pkg.StrandMPNN,
	graph_path:              string,
	base_checkpoint_path:    string,
	ingest_cfg:              ingest_pkg.Config,
	query_cfg:               query_pkg.Config,
	mu:                      sync.Mutex,
	queue:                   Ingest_Queue,
	queue_ok:                bool,
	subs_mu:                 sync.Mutex,
	subs:                    [dynamic]net.TCP_Socket,
	query_routing_threshold: f32,
	edge_decay:              f32,
}

// handler_start_queue creates and starts the async ingest queue.
handler_start_queue :: proc(h: ^Handler) -> bool {
	if !queue_init(&h.queue, h, &h.mu) {return false}
	h.queue_ok = true
	return true
}

// handler_stop_queue shuts down the ingest queue and joins the worker.
handler_stop_queue :: proc(h: ^Handler) {
	if h.queue_ok {
		queue_destroy(&h.queue)
		h.queue_ok = false
	}
}

// handler_enqueue pushes text into the async ingest queue.
handler_enqueue :: proc(h: ^Handler, text: string, descriptor: string = "") -> bool {
	if !h.queue_ok {return false}
	return queue_push(&h.queue, text, descriptor)
}

IngestResult :: struct {
	added:    int,
	thoughts: int,
	edges:    int,
}

handle_ingest :: proc(h: ^Handler, text: string, descriptor: string = "") -> IngestResult {
	added := ingest_pkg.ingest(h.g, h.p, text, "", descriptor, h.ingest_cfg, h.graph_path)

	if added > 0 {
		log.info(
			"[ingest] added %d thoughts (total: %d thoughts, %d edges)",
			added,
			graph_pkg.thought_count(h.g),
			graph_pkg.edge_count(h.g),
		)
		_run_gnn_and_save(h, added)
	}

	return IngestResult{
		added    = added,
		thoughts = graph_pkg.thought_count(h.g),
		edges    = graph_pkg.edge_count(h.g),
	}
}

handle_set_purpose :: proc(h: ^Handler, purpose: string) {
	graph_pkg.set_purpose(h.g, purpose)
	_save_graph(h)
	log.info("[purpose] set to: \"%s\"", h.g.purpose)
}

handle_descriptor_add :: proc(h: ^Handler, name: string, text: string) {
	graph_pkg.set_descriptor(h.g, name, text)
	_save_graph(h)
	log.info("[descriptor] added \"%s\" (%d bytes)", name, len(text))
}

handle_descriptor_remove :: proc(h: ^Handler, name: string) -> bool {
	ok := graph_pkg.remove_descriptor(h.g, name)
	if ok {
		_save_graph(h)
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
	d := graph_pkg.get_descriptor(h.g, name)
	if d == nil {return ""}
	return d.text
}

handle_ask :: proc(h: ^Handler, query_text: string) -> (answer: string, ok: bool) {
	query_embedding, embed_ok := h.p.embed_text(h.p, query_text)
	if !embed_ok {
		log.warn("[ask] failed to embed query")
		return "", false
	}

	qcfg := h.query_cfg
	if qcfg.top_k == 0 {qcfg = query_pkg.DEFAULT_CONFIG}

	scored := query_pkg.retrieve(
		h.g,
		&query_embedding,
		h.model,
		h.strand,
		qcfg,
	)
	defer delete(scored)

	if len(scored) == 0 {
		log.info("[ask] no relevant thoughts found")
		return "", false
	}

	// Confidence gate: if the top result exceeds the threshold, return its text
	// directly without invoking the LLM.
	if qcfg.confidence_threshold > 0.0 && scored[0].score >= qcfg.confidence_threshold {
		b := strings.builder_make()
		for st, i in scored {
			if i > 0 {strings.write_string(&b, "\n\n")}
			strings.write_string(&b, st.thought.text)
		}
		answer_text := strings.to_string(b)
		log.info("[ask] confidence gate triggered (score=%.3f >= %.3f), skipping LLM", scored[0].score, qcfg.confidence_threshold)
		return answer_text, true
	}

	result, answer_ok := query_pkg.answer(h.g, h.p, query_text, scored)
	if !answer_ok {
		log.warn("[ask] failed to generate answer")
		return "", false
	}
	defer delete(result.sources)

	log.info(
		"[ask] responded with %d bytes (%d sources)",
		len(result.answer_text),
		len(result.sources),
	)
	return result.answer_text, true
}

// handle_register_specialist upserts a meta-node for a specialist store into the
// handler's graph.  Called when a specialist is spawned by limbo or loaded externally.
// specialist_profile is the specialist graph's running profile embedding.
handle_register_specialist :: proc(h: ^Handler, store_name: string, specialist_profile: ^graph_pkg.Embedding) {
	now := time.time_to_unix(time.now())
	tid := graph_pkg.upsert_registry_node(h.g, store_name, specialist_profile, now)
	if tid != 0 {
		log.info("[handler] upserted meta-node %d for specialist '%s'", tid, store_name)
		_save_graph(h)
	}
}

handle_status :: proc(h: ^Handler) -> string {
	thoughts   := graph_pkg.thought_count(h.g)
	edges      := graph_pkg.edge_count(h.g)
	gnn_step   := h.model != nil ? h.model.adam_t : 0
	strand_step := h.strand != nil ? h.strand.adam_t : 0
	queued     := h.queue_ok ? queue_len(&h.queue) : 0
	in_flight  := h.queue_ok ? queue_in_flight(&h.queue) : 0
	return fmt.aprintf(
		"thoughts:%d edges:%d gnn_step:%d strand_step:%d queued:%d in_flight:%d graph:%s",
		thoughts,
		edges,
		gnn_step,
		strand_step,
		queued,
		in_flight,
		h.graph_path,
	)
}

// handler_subscribe registers sock as a status-event subscriber.
handler_subscribe :: proc(h: ^Handler, sock: net.TCP_Socket) {
	sync.lock(&h.subs_mu)
	defer sync.unlock(&h.subs_mu)
	for s in h.subs {
		if s == sock {return}
	}
	append(&h.subs, sock)
	log.info("[subs] socket subscribed (%d total)", len(h.subs))
}

// handler_unsubscribe removes sock from the subscriber list.
handler_unsubscribe :: proc(h: ^Handler, sock: net.TCP_Socket) {
	sync.lock(&h.subs_mu)
	defer sync.unlock(&h.subs_mu)
	for i := 0; i < len(h.subs); i += 1 {
		if h.subs[i] == sock {
			ordered_remove(&h.subs, i)
			log.info("[subs] socket unsubscribed (%d remaining)", len(h.subs))
			return
		}
	}
}

// handler_push_event sends a status frame to every subscriber.
// Must NOT be called with h.mu held.
handler_push_event :: proc(h: ^Handler) {
	status := handle_status(h)
	defer delete(status)

	data := transmute([]u8)status
	hdr: [4]u8
	endian.put_u32(hdr[:], .Big, u32(len(data)))

	sync.lock(&h.subs_mu)
	defer sync.unlock(&h.subs_mu)

	dead: [dynamic]int
	defer delete(dead)

	for i := 0; i < len(h.subs); i += 1 {
		_, err1 := net.send_tcp(h.subs[i], hdr[:])
		_, err2 := net.send_tcp(h.subs[i], data)
		if err1 != nil || err2 != nil {
			append(&dead, i)
		}
	}
	#reverse for i in dead {
		ordered_remove(&h.subs, i)
	}
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// _run_gnn_and_save runs GNN training and saves checkpoints.
// Must be called with handler mutex held (or from inside handle_ingest before returning).
@(private)
_run_gnn_and_save :: proc(h: ^Handler, added: int) {
	if added <= 0 || h.model == nil {return}

	// Apply edge decay before training so stale edges don't bias the GNN.
	if h.edge_decay > 0.0 {
		now := time.time_to_unix(time.now())
		graph_pkg.apply_edge_decay(h.g, h.edge_decay, now)
	}

	n := graph_pkg.thought_count(h.g)
	log.info("[gnn] training (%d thoughts, %d edges)", n, graph_pkg.edge_count(h.g))

	strand_steps := gnn_pkg.adaptive_steps(
		n,
		gnn_pkg.STRAND_TRAIN_STEPS_MAX,
		gnn_pkg.STRAND_TRAIN_STEPS_MIN,
	)
	base_steps := gnn_pkg.adaptive_steps(
		n,
		gnn_pkg.BASE_REFINE_STEPS_MAX,
		gnn_pkg.BASE_REFINE_STEPS_MIN,
	)
	gnn_pkg.train_strand(h.model, h.strand, h.g, strand_steps)
	gnn_pkg.train_base_refine(h.model, h.strand, h.g, base_steps)

	gnn_pkg.save_checkpoint(h.model, h.base_checkpoint_path)

	if h.strand != nil {
		strand_bytes := gnn_pkg.strand_save_bytes(h.strand)
		if strand_bytes != nil {
			graph_pkg.save_with_strand(h.g, h.graph_path, strand_bytes)
			delete(strand_bytes)
		}
	}
}

@(private)
_save_graph :: proc(h: ^Handler) {
	if strings.has_suffix(h.graph_path, util.STRAND_EXTENSION) {
		strand_bytes: []u8
		if h.strand != nil {
			strand_bytes = gnn_pkg.strand_save_bytes(h.strand)
		}
		if strand_bytes == nil {
			strand_bytes = make([]u8, 0)
		}
		graph_pkg.save_with_strand(h.g, h.graph_path, strand_bytes)
		delete(strand_bytes)
	} else {
		graph_pkg.save(h.g, h.graph_path)
	}
}
