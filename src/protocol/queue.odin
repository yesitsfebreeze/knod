package protocol

import "core:strings"
import "core:sync"
import "core:sync/chan"
import "core:thread"

import "../gnn"
import "../graph"
import "../ingest"
import log "../logger"
import "../provider"

// Ingest_Item holds a single queued ingestion request.
// Text and descriptor are cloned — the queue owns them.
Ingest_Item :: struct {
	text:       string,
	descriptor: string,
}

// Ingest_Queue is a thread-safe queue backed by an Odin channel.
// A dedicated worker thread dispatches each item to its own goroutine
// immediately on receipt.  All LLM work (prepare + link) happens in those
// goroutines with no mutex held; the mutex is acquired only for the two brief
// graph-access windows (snapshot and commit).
Ingest_Queue :: struct {
	ch:        chan.Chan(^Ingest_Item),
	handler:   ^Handler,
	mu:        ^sync.Mutex, // shared mutex protecting handler/graph access
	worker:    ^thread.Thread,
	in_flight: i64, // number of article goroutines currently running (atomic)
}

QUEUE_CAPACITY :: 128

// queue_init initialises the ingest queue in-place and starts its worker thread.
// q must be a stable pointer (e.g. &handler.queue) that outlives the worker.
queue_init :: proc(q: ^Ingest_Queue, handler: ^Handler, mu: ^sync.Mutex) -> bool {
	c, err := chan.create(chan.Chan(^Ingest_Item), QUEUE_CAPACITY, context.allocator)
	if err != .None {
		log.err("[queue] could not create channel: %v", err)
		return false
	}

	q.ch = c
	q.handler = handler
	q.mu = mu
	q.worker = thread.create_and_start_with_poly_data2(q, mu, _queue_worker, context)
	if q.worker == nil {
		chan.destroy(q.ch)
		log.err("[queue] could not start worker thread")
		return false
	}

	log.info("[queue] ingest worker started (capacity %d, fully parallel)", QUEUE_CAPACITY)
	return true
}

// queue_destroy closes the channel and joins the worker thread.
queue_destroy :: proc(q: ^Ingest_Queue) {
	chan.close(q.ch)
	if q.worker != nil {
		thread.join(q.worker)
		thread.destroy(q.worker)
		q.worker = nil
	}
	chan.destroy(q.ch)
	log.info("[queue] ingest worker stopped")
}

// queue_push enqueues an ingestion request.  Text and descriptor are cloned.
// Returns false if the queue is full or closed.
queue_push :: proc(q: ^Ingest_Queue, text: string, descriptor: string = "") -> bool {
	item := new(Ingest_Item)
	item.text = strings.clone(text)
	item.descriptor = len(descriptor) > 0 ? strings.clone(descriptor) : ""

	ok := chan.send(q.ch, item)
	if !ok {
		// Channel closed or full — should not normally happen.
		_free_item(item)
		return false
	}
	return true
}

// queue_len returns the number of items waiting in the queue.
queue_len :: proc(q: ^Ingest_Queue) -> int {
	return chan.len(q.ch)
}

// queue_in_flight returns the number of article goroutines currently running.
queue_in_flight :: proc(q: ^Ingest_Queue) -> int {
	return int(sync.atomic_load(&q.in_flight))
}

// --- internal ---

// _Article_Job is heap-allocated per item and owned by the article goroutine.
// It carries everything the goroutine needs without referencing the queue struct
// after dispatch (the queue may move on to the next item immediately).
@(private)
_Article_Job :: struct {
	item:       ^Ingest_Item,
	p:          ^provider.Provider,  // shared stateless provider
	handler:    ^Handler,            // for graph/model access under mutex
	mu:         ^sync.Mutex,
	purpose:    string,              // cloned at dispatch time
	ingest_cfg: ingest.Config,       // value copy — safe for concurrent reads
	self:       ^thread.Thread,      // goroutine destroys its own handle on exit
	in_flight:  ^i64,                // shared counter; decremented on goroutine exit
}

// _queue_worker is a pure dispatcher.  It receives items as fast as they
// arrive and immediately spawns a goroutine for each one.  No batching,
// no waiting for LLM calls to finish before accepting the next item.
@(private)
_queue_worker :: proc(q: ^Ingest_Queue, mu: ^sync.Mutex) {
	// Read purpose once; if it could ever change, re-read per-item inside mutex.
	// For now purpose is set at startup and never changes — safe to clone once
	// but we clone per-job to keep the goroutine fully self-contained.

	for {
		item, recv_ok := chan.recv(q.ch)
		if !recv_ok {
			// Channel closed — exit.
			break
		}

		// Snapshot mutable state needed by the goroutine while we hold nothing.
		// Purpose lives on the graph struct; read under a brief lock.
		sync.lock(mu)
		purpose := strings.clone(q.handler.g.purpose)
		cfg     := q.handler.ingest_cfg // value copy
		sync.unlock(mu)

		job := new(_Article_Job)
		job.item       = item
		job.p          = q.handler.p
		job.handler    = q.handler
		job.mu         = mu
		job.purpose    = purpose
		job.ingest_cfg = cfg
		job.self       = nil // filled in below after thread.create
		job.in_flight  = &q.in_flight

		sync.atomic_add(&q.in_flight, 1)

		t := thread.create(_article_goroutine)
		if t == nil {
			// Thread creation failed — run inline as fallback (blocks dispatcher
			// for this item, but keeps correctness).
			log.warn("[queue] thread creation failed, running inline (%d bytes)", len(item.text))
			// Build a fake thread.Thread on the stack so _article_goroutine can read user_args[0].
			fake_t: thread.Thread
			fake_t.user_args[0] = job
			job.self = nil // no handle to destroy in fallback
			_article_goroutine(&fake_t)
		} else {
			t.user_args[0]  = job
			t.init_context  = context
			job.self        = t   // goroutine will destroy its own handle
			thread.start(t)
		}

		log.info("[queue] dispatched article (%d bytes, %d still queued)", len(item.text), chan.len(q.ch))
	}
}

// _article_goroutine runs the full four-phase pipeline for a single article.
// It is self-contained and self-cleaning: it frees job and item on exit,
// and destroys its own thread handle so the dispatcher never blocks.
@(private)
_article_goroutine :: proc(t: ^thread.Thread) {
	job := (^_Article_Job)(t.user_args[0])
	defer {
		sync.atomic_sub(job.in_flight, 1)
		delete(job.purpose)
		_free_item(job.item)
		if job.self != nil {thread.destroy(job.self)}
		free(job)
	}

	text_preview := job.item.text
	if len(text_preview) > 40 {text_preview = text_preview[:40]}
	log.info("[goroutine] start: %.40s...", text_preview)

	// --- Phase 1: Prepare — decompose + embed (no mutex, LLM calls) ---
	log.info("[goroutine] phase1:prepare start (%.40s...)", text_preview)
	pa, ok := ingest.ingest_prepare(job.p, job.item.text, job.purpose, job.item.descriptor)
	if !ok {
		log.warn("[goroutine] prepare failed (%.40s...)", text_preview)
		return
	}
	defer ingest.prepared_article_release(&pa)
	log.info("[goroutine] phase1:prepare done: %d thoughts (%.40s...)", len(pa.thoughts), text_preview)

	// --- Phase 1.5: Snapshot — brief mutex, graph read-only ---
	log.info("[goroutine] phase1.5:snapshot start")
	sync.lock(job.mu)
	ingest.ingest_snapshot(job.handler.g, &pa, job.ingest_cfg)
	sync.unlock(job.mu)
	log.info("[goroutine] phase1.5:snapshot done")

	// --- Phase 1.75: Link — batch_link_reason + edge embeddings (no mutex, LLM calls) ---
	log.info("[goroutine] phase1.75:link start")
	ingest.ingest_link(job.p, &pa, job.ingest_cfg)
	log.info("[goroutine] phase1.75:link done")

	// --- Phase 2: Commit + GNN (mutex — pure graph writes, no LLM) ---
	log.info("[goroutine] phase2:commit start")
	sync.lock(job.mu)
	added := ingest.ingest_commit(job.handler.g, job.handler.p, &pa, job.ingest_cfg)
	log.info("[goroutine] phase2:commit done: %d added", added)
	_run_gnn_and_save(job.handler, added)
	sync.unlock(job.mu)

	log.info("[goroutine] complete: %d thoughts added (%.40s...)", added, text_preview)
}

// _run_gnn_and_save runs GNN training and saves checkpoints if any thoughts
// were added.  Must be called with handler mutex held.
@(private)
_run_gnn_and_save :: proc(h: ^Handler, added: int) {
	if added <= 0 {return}

	log.info(
		"[queue] training GNN (%d thoughts, %d edges)",
		graph.thought_count(h.g),
		graph.edge_count(h.g),
	)

	n := graph.thought_count(h.g)
	strand_steps := gnn.adaptive_steps(n, gnn.STRAND_TRAIN_STEPS_MAX, gnn.STRAND_TRAIN_STEPS_MIN)
	base_steps := gnn.adaptive_steps(n, gnn.BASE_REFINE_STEPS_MAX, gnn.BASE_REFINE_STEPS_MIN)
	gnn.train_strand(h.model, h.strand, h.g, strand_steps)
	gnn.train_base_refine(h.model, h.strand, h.g, base_steps)

	gnn.save_checkpoint(h.model, h.base_checkpoint_path)

	strand_bytes := gnn.strand_save_bytes(h.strand)
	if strand_bytes != nil {
		graph.save_strand(h.g, strand_bytes, h.graph_path)
		delete(strand_bytes)
	}
}

@(private)
_free_item :: proc(item: ^Ingest_Item) {
	if len(item.text) > 0 {delete(item.text)}
	if len(item.descriptor) > 0 {delete(item.descriptor)}
	free(item)
}
