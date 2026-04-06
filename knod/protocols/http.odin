package protocols

// http.odin — HTTP server.
// Near-verbatim copy of archive/src/protocol/http.odin.
// Explore endpoints (GET /graph, /thoughts, /thought/<id>, /find) are in explore.odin.

import "core:fmt"
import "core:net"
import "core:strings"
import "core:sync"
import "core:thread"

import http_pkg "../http"
import log      "../logger"

HTTP :: struct {
	server:  http_pkg.Server,
	handler: ^Handler,
	port:    int,
	t:       ^thread.Thread,
}

http_create :: proc(port: int, handler: ^Handler) -> (h: HTTP, ok: bool) {
	h.handler = handler
	h.port    = port
	h.t       = thread.create_and_start_with_poly_data2(&h, port, _http_server_thread, context)
	if h.t == nil {
		log.err("http: could not start server thread")
		return {}, false
	}
	log.info("http: listening on :%d", port)
	return h, true
}

http_destroy :: proc(h: ^HTTP) {
	http_pkg.server_shutdown(&h.server)
	if h.t != nil {
		thread.join(h.t)
		thread.destroy(h.t)
		h.t = nil
	}
}

@(private)
_http_server_thread :: proc(h: ^HTTP, port: int) {
	endpoint := net.Endpoint{
		address = net.IP4_Address{0, 0, 0, 0},
		port    = port,
	}
	handler := http_pkg.Handler{
		user_data = h,
		handle    = _http_dispatch,
	}
	err := http_pkg.listen_and_serve(&h.server, handler, endpoint)
	if err != nil {
		log.err("http: server error: %v", err)
	}
}

@(private)
HTTP_Context :: struct {
	h:    ^HTTP,
	req:  ^http_pkg.Request,
	res:  ^http_pkg.Response,
	path: string,
}

@(private)
_http_dispatch :: proc(handler: ^http_pkg.Handler, req: ^http_pkg.Request, res: ^http_pkg.Response) {
	h := cast(^HTTP)handler.user_data

	rline  := req.line.(http_pkg.Requestline)
	method := rline.method
	path   := req.url.path

	// CORS
	http_pkg.headers_set_unsafe(&res.headers, "access-control-allow-origin", "*")
	if method == .Options {
		http_pkg.headers_set_unsafe(&res.headers, "access-control-allow-methods", "GET, POST, DELETE, OPTIONS")
		http_pkg.headers_set_unsafe(&res.headers, "access-control-allow-headers", "content-type")
		res.status = .No_Content
		http_pkg.respond(res)
		return
	}

	if method == .Post && (path == "/ingest" || strings.has_prefix(path, "/ingest")) {
		ctx      := new(HTTP_Context)
		ctx.h    = h
		ctx.req  = req
		ctx.res  = res
		ctx.path = path
		http_pkg.body(req, 4 * 1024 * 1024, ctx, _handle_ingest_body)
		return
	}

	if method == .Post && path == "/ask" {
		ctx      := new(HTTP_Context)
		ctx.h    = h
		ctx.req  = req
		ctx.res  = res
		ctx.path = path
		http_pkg.body(req, 1 * 1024 * 1024, ctx, _handle_ask_body)
		return
	}

	if method == .Post && path == "/purpose" {
		ctx      := new(HTTP_Context)
		ctx.h    = h
		ctx.req  = req
		ctx.res  = res
		ctx.path = path
		http_pkg.body(req, 4096, ctx, _handle_purpose_body)
		return
	}

	if method == .Post && strings.has_prefix(path, "/descriptor/add") {
		ctx      := new(HTTP_Context)
		ctx.h    = h
		ctx.req  = req
		ctx.res  = res
		ctx.path = path
		http_pkg.body(req, 64 * 1024, ctx, _handle_descriptor_add_body)
		return
	}

	if method == .Delete && strings.has_prefix(path, "/descriptor/") {
		name := path[len("/descriptor/"):]
		if len(name) == 0 {
			http_pkg.respond(res, http_pkg.Status.Bad_Request)
			return
		}
		sync.lock(&h.handler.mu)
		ok := handle_descriptor_remove(h.handler, name)
		sync.unlock(&h.handler.mu)
		if ok {
			http_pkg.respond(res, http_pkg.Status.OK)
		} else {
			http_pkg.respond(res, http_pkg.Status.Not_Found)
		}
		return
	}

	if method == .Get && (path == "/descriptor" || path == "/descriptor/") {
		sync.lock(&h.handler.mu)
		list := handle_descriptor_list(h.handler)
		sync.unlock(&h.handler.mu)
		res.status = .OK
		http_pkg.body_set(res, list)
		delete(list)
		http_pkg.respond(res)
		return
	}

	if method == .Get && (path == "/health" || path == "/") {
		res.status = .OK
		http_pkg.body_set(res, "ok")
		http_pkg.respond(res)
		return
	}

	if method == .Get && path == "/status" {
		status := handle_status(h.handler)
		res.status = .OK
		http_pkg.body_set(res, status)
		delete(status)
		http_pkg.respond(res)
		return
	}

	// --- explore API ---

	if method == .Get && (path == "/graph" || path == "/graph/") {
		sync.lock(&h.handler.mu)
		body := handle_graph_info(h.handler)
		sync.unlock(&h.handler.mu)
		res.status = .OK
		http_pkg.headers_set_unsafe(&res.headers, "content-type", "application/json")
		http_pkg.body_set(res, body)
		delete(body)
		http_pkg.respond(res)
		return
	}

	if method == .Get && (path == "/thoughts" || path == "/thoughts/") {
		sync.lock(&h.handler.mu)
		body := handle_thoughts_list(h.handler)
		sync.unlock(&h.handler.mu)
		res.status = .OK
		http_pkg.headers_set_unsafe(&res.headers, "content-type", "application/json")
		http_pkg.body_set(res, body)
		delete(body)
		http_pkg.respond(res)
		return
	}

	if method == .Get && strings.has_prefix(path, "/thought/") {
		id_str := path[len("/thought/"):]
		if len(id_str) == 0 {
			http_pkg.respond(res, http_pkg.Status.Bad_Request)
			return
		}
		sync.lock(&h.handler.mu)
		body, ok := handle_thought_detail(h.handler, id_str)
		sync.unlock(&h.handler.mu)
		if !ok {
			http_pkg.respond(res, http_pkg.Status.Not_Found)
			return
		}
		res.status = .OK
		http_pkg.headers_set_unsafe(&res.headers, "content-type", "application/json")
		http_pkg.body_set(res, body)
		delete(body)
		http_pkg.respond(res)
		return
	}

	if method == .Get && strings.has_prefix(path, "/find") {
		_handle_find_dispatch(h, req, res)
		return
	}

	http_pkg.respond(res, http_pkg.Status.Not_Found)
}

@(private)
_handle_ingest_body :: proc(user_data: rawptr, body: http_pkg.Body, err: http_pkg.Body_Error) {
	ctx := cast(^HTTP_Context)user_data
	defer free(ctx)

	if err != nil {
		http_pkg.respond(ctx.res, http_pkg.body_error_status(err))
		return
	}
	if len(body) == 0 {
		http_pkg.respond(ctx.res, http_pkg.Status.Bad_Request)
		return
	}

	query        := ctx.req.url.query
	desc_name    := _query_param(query, "descriptor")
	descriptor_text := ""
	if len(desc_name) > 0 {
		sync.lock(&ctx.h.handler.mu)
		descriptor_text = get_descriptor_text(ctx.h.handler, desc_name)
		sync.unlock(&ctx.h.handler.mu)
		if len(descriptor_text) == 0 {
			log.warn("[http/ingest] descriptor \"%s\" not found, ingesting without descriptor", desc_name)
		}
	}

	log.info("[http/ingest] received %d bytes (descriptor: \"%s\")", len(body), desc_name)

	handler := ctx.h.handler
	if handler_enqueue(handler, string(body), descriptor_text) {
		pending := queue_len(&handler.queue)
		reply   := fmt.aprintf("queued (%d pending)", pending)
		res     := ctx.res
		res.status = .OK
		http_pkg.body_set(res, reply)
		delete(reply)
		http_pkg.respond(res)
	} else {
		sync.lock(&handler.mu)
		result := handle_ingest(handler, string(body), descriptor_text)
		sync.unlock(&handler.mu)

		reply := fmt.aprintf(
			"added %d thoughts (%d total, %d edges)",
			result.added,
			result.thoughts,
			result.edges,
		)
		res := ctx.res
		res.status = .OK
		http_pkg.body_set(res, reply)
		delete(reply)
		http_pkg.respond(res)
	}
}

@(private)
_handle_ask_body :: proc(user_data: rawptr, body: http_pkg.Body, err: http_pkg.Body_Error) {
	ctx := cast(^HTTP_Context)user_data
	defer free(ctx)

	if err != nil {
		http_pkg.respond(ctx.res, http_pkg.body_error_status(err))
		return
	}
	if len(body) == 0 {
		http_pkg.respond(ctx.res, http_pkg.Status.Bad_Request)
		return
	}

	query_text := string(body)
	log.info("[http/ask] query: \"%s\"", query_text)

	sync.lock(&ctx.h.handler.mu)
	answer, ok := handle_ask(ctx.h.handler, query_text)
	sync.unlock(&ctx.h.handler.mu)

	res := ctx.res
	res.status = .OK
	if ok {
		http_pkg.body_set(res, answer)
		delete(answer)
	}
	http_pkg.respond(res)
}

@(private)
_handle_purpose_body :: proc(user_data: rawptr, body: http_pkg.Body, err: http_pkg.Body_Error) {
	ctx := cast(^HTTP_Context)user_data
	defer free(ctx)

	if err != nil {
		http_pkg.respond(ctx.res, http_pkg.body_error_status(err))
		return
	}
	if len(body) == 0 {
		http_pkg.respond(ctx.res, http_pkg.Status.Bad_Request)
		return
	}

	purpose := strings.trim_space(string(body))
	sync.lock(&ctx.h.handler.mu)
	handle_set_purpose(ctx.h.handler, purpose)
	sync.unlock(&ctx.h.handler.mu)

	res := ctx.res
	res.status = .OK
	http_pkg.body_set(res, "ok")
	http_pkg.respond(res)
}

@(private)
_handle_descriptor_add_body :: proc(user_data: rawptr, body: http_pkg.Body, err: http_pkg.Body_Error) {
	ctx := cast(^HTTP_Context)user_data
	defer free(ctx)

	if err != nil {
		http_pkg.respond(ctx.res, http_pkg.body_error_status(err))
		return
	}
	if len(body) == 0 {
		http_pkg.respond(ctx.res, http_pkg.Status.Bad_Request)
		return
	}

	name := _query_param(ctx.req.url.query, "name")
	if len(name) == 0 {
		http_pkg.respond(ctx.res, http_pkg.Status.Bad_Request)
		return
	}

	sync.lock(&ctx.h.handler.mu)
	handle_descriptor_add(ctx.h.handler, name, string(body))
	sync.unlock(&ctx.h.handler.mu)

	res := ctx.res
	res.status = .OK
	http_pkg.body_set(res, "ok")
	http_pkg.respond(res)
}

@(private)
_query_param :: proc(query: string, key: string) -> string {
	q := query
	for {
		entry, ok := http_pkg.query_iter(&q)
		if !ok {break}
		if entry.key == key {return entry.value}
	}
	return ""
}
