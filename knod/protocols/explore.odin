package protocols

// explore.odin — JSON API for graph inspection.
//
// Adapted from archive/src/protocol/explore.odin.
// Differences:
//   - Thought.source  (not source_id)
//   - No tags (py_knod has no tags)
//   - Uses query_pkg.retrieve for /find (more accurate than raw cosine+edge scan)

import "core:fmt"
import "core:strconv"
import "core:strings"
import "core:sync"

import graph_pkg "../graph"
import http_pkg  "../http"
import query_pkg "../query"

// GET /graph
// Returns JSON: { purpose, thought_count, edge_count }
handle_graph_info :: proc(h: ^Handler) -> string {
	b := strings.builder_make()
	strings.write_string(&b, "{")
	strings.write_string(&b, `"purpose":`)
	json_string(&b, h.g.purpose)
	fmt.sbprintf(&b, `,"thought_count":%d`, graph_pkg.thought_count(h.g))
	fmt.sbprintf(&b, `,"edge_count":%d}`, graph_pkg.edge_count(h.g))
	return strings.to_string(b)
}

// GET /thoughts
// Returns JSON array sorted by access_count desc, then created_at desc.
handle_thoughts_list :: proc(h: ^Handler) -> string {
	b := strings.builder_make()
	strings.write_byte(&b, '[')

	Entry :: struct {
		id:            u64,
		text:          string,
		source:        string,
		created_at:    i64,
		access_count:  u32,
		last_accessed: i64,
	}
	entries := make([dynamic]Entry, 0, len(h.g.thoughts))
	defer delete(entries)

	for id, &t in h.g.thoughts {
		append(
			&entries,
			Entry{
				id            = id,
				text          = t.text,
				source        = t.source,
				created_at    = t.created_at,
				access_count  = t.access_count,
				last_accessed = t.last_accessed,
			},
		)
	}

	// Sort descending by access_count, then created_at.
	for i in 1 ..< len(entries) {
		j := i
		for j > 0 {
			a  := entries[j]
			bv := entries[j - 1]
			if a.access_count > bv.access_count ||
			   (a.access_count == bv.access_count && a.created_at > bv.created_at) {
				entries[j], entries[j - 1] = entries[j - 1], entries[j]
				j -= 1
			} else {
				break
			}
		}
	}

	for e, i in entries {
		if i > 0 {strings.write_byte(&b, ',')}
		fmt.sbprintf(&b, `{"id":%d,"source":`, e.id)
		json_string(&b, e.source)
		strings.write_string(&b, `,"text":`)
		json_string(&b, e.text)
		fmt.sbprintf(
			&b,
			`,"created_at":%d,"access_count":%d,"last_accessed":%d}`,
			e.created_at,
			e.access_count,
			e.last_accessed,
		)
	}

	strings.write_byte(&b, ']')
	return strings.to_string(b)
}

// GET /thought/<id>
// Returns JSON with thought fields and outgoing/incoming edge lists.
handle_thought_detail :: proc(h: ^Handler, id_str: string) -> (body: string, ok: bool) {
	id_u64, parse_ok := strconv.parse_u64(id_str)
	if !parse_ok {return "", false}

	t := graph_pkg.get_thought(h.g, id_u64)
	if t == nil {return "", false}

	b := strings.builder_make()
	fmt.sbprintf(&b, `{"id":%d,"text":`, t.id)
	json_string(&b, t.text)
	strings.write_string(&b, `,"source":`)
	json_string(&b, t.source)
	fmt.sbprintf(
		&b,
		`,"created_at":%d,"access_count":%d,"last_accessed":%d`,
		t.created_at,
		t.access_count,
		t.last_accessed,
	)

	// Outgoing edges
	out_edges := graph_pkg.outgoing_edges(h.g, id_u64)
	defer delete(out_edges)
	strings.write_string(&b, `,"outgoing":[`)
	for e, i in out_edges {
		if i > 0 {strings.write_byte(&b, ',')}
		fmt.sbprintf(&b, `{"target_id":%d,"weight":%f,"reasoning":`, e.target_id, e.weight)
		json_string(&b, e.reasoning)
		fmt.sbprintf(&b, `,"created_at":%d}`, e.created_at)
	}
	strings.write_string(&b, "]")

	// Incoming edges
	in_edges := graph_pkg.incoming_edges(h.g, id_u64)
	defer delete(in_edges)
	strings.write_string(&b, `,"incoming":[`)
	for e, i in in_edges {
		if i > 0 {strings.write_byte(&b, ',')}
		fmt.sbprintf(&b, `{"source_id":%d,"weight":%f,"reasoning":`, e.source_id, e.weight)
		json_string(&b, e.reasoning)
		fmt.sbprintf(&b, `,"created_at":%d}`, e.created_at)
	}
	strings.write_string(&b, "]}")

	return strings.to_string(b), true
}

// GET /find?q=<text>
// Returns JSON array of { id, score, text, source } sorted by score desc.
handle_find :: proc(h: ^Handler, query_text: string) -> (body: string, ok: bool) {
	if len(query_text) == 0 {return "", false}

	query_embedding, embed_ok := h.p.embed_text(h.p, query_text)
	if !embed_ok {return "", false}

	scored := query_pkg.retrieve(h.g, &query_embedding, h.model, h.strand)
	defer delete(scored)

	b := strings.builder_make()
	strings.write_byte(&b, '[')
	for st, i in scored {
		if i > 0 {strings.write_byte(&b, ',')}
		fmt.sbprintf(&b, `{"id":%d,"score":%f,"text":`, st.thought.id, st.score)
		json_string(&b, st.thought.text)
		strings.write_string(&b, `,"source":`)
		json_string(&b, st.thought.source)
		strings.write_byte(&b, '}')
	}
	strings.write_byte(&b, ']')
	return strings.to_string(b), true
}

// json_string writes a JSON-encoded, escaped string literal into a builder.
@(private)
json_string :: proc(b: ^strings.Builder, s: string) {
	strings.write_byte(b, '"')
	for c in s {
		switch c {
		case '"':  strings.write_string(b, `\"`)
		case '\\': strings.write_string(b, `\\`)
		case '\n': strings.write_string(b, `\n`)
		case '\r': strings.write_string(b, `\r`)
		case '\t': strings.write_string(b, `\t`)
		case:      strings.write_rune(b, c)
		}
	}
	strings.write_byte(b, '"')
}

// _handle_find_dispatch is called from the HTTP dispatch loop.
_handle_find_dispatch :: proc(h: ^HTTP, req: ^http_pkg.Request, res: ^http_pkg.Response) {
	query_text := _query_param(req.url.query, "q")
	if len(query_text) == 0 {
		http_pkg.respond(res, http_pkg.Status.Bad_Request)
		return
	}

	sync.lock(&h.handler.mu)
	body, ok := handle_find(h.handler, query_text)
	sync.unlock(&h.handler.mu)

	if !ok {
		http_pkg.respond(res, http_pkg.Status.Internal_Server_Error)
		if len(body) > 0 {delete(body)}
		return
	}

	res.status = .OK
	http_pkg.headers_set_unsafe(&res.headers, "content-type", "application/json")
	http_pkg.body_set(res, body)
	delete(body)
	http_pkg.respond(res)
}
