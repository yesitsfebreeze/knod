package protocols

// queue.odin — async ingest queue.
//
// Adapted from archive/src/protocol/queue.odin.
// Key differences:
//   - Calls ingest.prepare / ingest.snapshot / ingest.link / ingest.commit
//     (archive called ingest_prepare / ingest_snapshot / ingest_link / ingest_commit)
//   - No tags phase (py_knod has no tags)
//   - graph.save_with_strand instead of graph.save_strand

import "base:intrinsics"
import "core:mem/virtual"
import "core:strings"
import "core:sync"
import "core:sync/chan"
import "core:thread"

import gnn_pkg    "../gnn"
import graph_pkg  "../graph"
import ingest_pkg "../ingest"
import log        "../logger"

// Ingest_Item holds a single queued ingestion request.
// Text and descriptor are cloned — the queue owns them.
Ingest_Item :: struct {
	text:       string,
	descriptor: string,
}

// Ingest_Queue is a thread-safe queue backed by an Odin channel.
Ingest_Queue :: struct {
	ch:        chan.Chan(^Ingest_Item),
	handler:   ^Handler,
	mu:        ^sync.Mutex,
	worker:    ^thread.Thread,
	in_flight: i64, // atomic
}

QUEUE_CAPACITY :: 128

// queue_init initialises the ingest queue in-place and starts its worker thread.
queue_init :: proc(q: ^Ingest_Queue, handler: ^Handler, mu: ^sync.Mutex) -> bool {
	c, err := chan.create(chan.Chan(^Ingest_Item), QUEUE_CAPACITY, context.allocator)
	if err != .None {
		log.err("[queue] could not create channel: %v", err)
		return false
	}

	q.ch      = c
	q.handler = handler
	q.mu      = mu
	q.worker  = thread.create_and_start_with_poly_data2(q, mu, _queue_worker, context)
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
queue_push :: proc(q: ^Ingest_Queue, text: string, descriptor: string = "") -> bool {
	item          := new(Ingest_Item)
	item.text      = strings.clone(text)
	item.descriptor = len(descriptor) > 0 ? strings.clone(descriptor) : ""

	ok := chan.send(q.ch, item)
	if !ok {
		_free_item(item)
		return false
	}
	return true
}

queue_len :: proc(q: ^Ingest_Queue) -> int {
	return chan.len(q.ch)
}

queue_in_flight :: proc(q: ^Ingest_Queue) -> int {
	return int(sync.atomic_load(&q.in_flight))
}

// ---------------------------------------------------------------------------
// Internal
// ---------------------------------------------------------------------------

@(private)
_Article_Job :: struct {
	item:       ^Ingest_Item,
	handler:    ^Handler,
	mu:         ^sync.Mutex,
	purpose:    string,       // cloned at dispatch time
	ingest_cfg: ingest_pkg.Config,
	in_flight:  ^i64,
}

@(private)
_queue_worker :: proc(q: ^Ingest_Queue, mu: ^sync.Mutex) {
	for {
		item, recv_ok := chan.recv(q.ch)
		if !recv_ok {break}

		sync.lock(mu)
		purpose := strings.clone(q.handler.g.purpose)
		cfg     := q.handler.ingest_cfg
		sync.unlock(mu)

		job            := new(_Article_Job)
		job.item        = item
		job.handler     = q.handler
		job.mu          = mu
		job.purpose     = purpose
		job.ingest_cfg  = cfg
		job.in_flight   = &q.in_flight

		sync.atomic_add(&q.in_flight, 1)

		t := thread.create(_article_goroutine)
		if t == nil {
			log.warn("[queue] thread creation failed, running inline (%d bytes)", len(item.text))
			fake_t: thread.Thread
			fake_t.user_args[0] = job
			_article_goroutine(&fake_t)
		} else {
			t.user_args[0]  = job
			t.init_context  = context
			intrinsics.atomic_or(&t.flags, {.Self_Cleanup})
			thread.start(t)
		}

		log.info("[queue] dispatched article (%d bytes, %d still queued)", len(item.text), chan.len(q.ch))
	}
}

@(private)
_article_goroutine :: proc(t: ^thread.Thread) {
	job := (^_Article_Job)(t.user_args[0])

	temp_arena: virtual.Arena
	_ = virtual.arena_init_growing(&temp_arena)
	context.temp_allocator = virtual.arena_allocator(&temp_arena)
	defer virtual.arena_destroy(&temp_arena)

	defer {
		sync.atomic_sub(job.in_flight, 1)
		handler_push_event(job.handler)
		delete(job.purpose)
		_free_item(job.item)
		free(job)
	}

	text_preview := job.item.text
	if len(text_preview) > 40 {text_preview = text_preview[:40]}
	log.info("[goroutine] start: %.40s...", text_preview)

	// --- Phase 1: Prepare (no mutex, LLM calls) ---
	log.info("[goroutine] phase1:prepare start (%.40s...)", text_preview)
	pa, ok := ingest_pkg.prepare(
		job.handler.p,
		job.item.text,
		job.purpose,
		"",                    // source — left blank; ingest.commit defaults to "ingest"
		job.item.descriptor,
	)
	if !ok {
		log.warn("[goroutine] prepare failed (%.40s...)", text_preview)
		return
	}
	defer ingest_pkg.prepared_article_release(&pa)
	log.info("[goroutine] phase1:prepare done: %d thoughts (%.40s...)", len(pa.thoughts), text_preview)

	// --- Phase 1.5: Snapshot (brief mutex, graph read-only) ---
	log.info("[goroutine] phase1.5:snapshot start")
	sync.lock(job.mu)
	ingest_pkg.snapshot(job.handler.g, &pa, job.ingest_cfg)
	sync.unlock(job.mu)
	log.info("[goroutine] phase1.5:snapshot done")

	// --- Phase 1.75: Link (no mutex, LLM calls) ---
	log.info("[goroutine] phase1.75:link start")
	ingest_pkg.link(job.handler.p, &pa, job.ingest_cfg)
	log.info("[goroutine] phase1.75:link done")

	// --- Phase 2: Commit + GNN (mutex, pure graph writes) ---
	log.info("[goroutine] phase2:commit start")
	sync.lock(job.mu)
	added := ingest_pkg.commit(job.handler.g, &pa, job.ingest_cfg)
	log.info("[goroutine] phase2:commit done: %d added", added)
	_run_gnn_and_save(job.handler, added)
	sync.unlock(job.mu)

	log.info("[goroutine] complete: %d thoughts added (%.40s...)", added, text_preview)
}

@(private)
_free_item :: proc(item: ^Ingest_Item) {
	if len(item.text) > 0       {delete(item.text)}
	if len(item.descriptor) > 0 {delete(item.descriptor)}
	free(item)
}
