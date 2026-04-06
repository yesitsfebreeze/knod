package protocol

import "core:fmt"
import "core:strconv"
import "core:strings"
import "core:sync"

import "../graph"
import http "../http"

// GET /graph
// Returns JSON: { purpose, thought_count, edge_count, tags: [{dim_index, label}] }
handle_graph_info :: proc(h: ^Handler) -> string {
	b := strings.builder_make()
	strings.write_string(&b, "{")

	// purpose
	strings.write_string(&b, `"purpose":`)
	json_string(&b, h.g.purpose)
	strings.write_string(&b, `,`)

	// counts
	fmt.sbprintf(&b, `"thought_count":%d,`, graph.thought_count(h.g))
	fmt.sbprintf(&b, `"edge_count":%d,`, graph.edge_count(h.g))

	// tags
	strings.write_string(&b, `"tags":[`)
	tags := graph.get_tags(h.g)
	for tag, i in tags {
		if i > 0 {strings.write_byte(&b, ',')}
		fmt.sbprintf(&b, `{"dim_index":%d,"label":`, tag.dim_index)
		json_string(&b, tag.label)
		strings.write_byte(&b, '}')
	}
	strings.write_string(&b, "]}")

	return strings.to_string(b)
}

// GET /thoughts
// Returns JSON array of all thoughts (id, text, source_id, created_at, access_count, last_accessed)
// Sorted by access_count desc, then created_at desc.
handle_thoughts_list :: proc(h: ^Handler) -> string {
	b := strings.builder_make()
	strings.write_byte(&b, '[')

	// Collect into a slice so we can sort
	Entry :: struct {
		id:            u64,
		text:          string,
		source_id:     string,
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
				source_id     = t.source_id,
				created_at    = t.created_at,
				access_count  = t.access_count,
				last_accessed = t.last_accessed,
			},
		)
	}

	// Sort by access_count desc, then created_at desc
	for i in 1 ..< len(entries) {
		j := i
		for j > 0 {
			a := entries[j]
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
		fmt.sbprintf(
			&b,
			`{"id":%d,"source_id":`,
			e.id,
		)
		json_string(&b, e.source_id)
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
// Returns JSON: { thought fields, outgoing: [{target_id, weight, reasoning, created_at}], incoming: [...] }
handle_thought_detail :: proc(h: ^Handler, id_str: string) -> (body: string, ok: bool) {
	id_u64, parse_ok := strconv.parse_u64(id_str)
	if !parse_ok {
		return "", false
	}

	t := graph.get_thought(h.g, id_u64)
	if t == nil {
		return "", false
	}

	b := strings.builder_make()
	strings.write_string(&b, `{"id":`)
	fmt.sbprintf(&b, "%d", t.id)
	strings.write_string(&b, `,"text":`)
	json_string(&b, t.text)
	strings.write_string(&b, `,"source_id":`)
	json_string(&b, t.source_id)
	fmt.sbprintf(&b, `,"created_at":%d,"access_count":%d,"last_accessed":%d`, t.created_at, t.access_count, t.last_accessed)

	// outgoing edges
	out_edges := graph.outgoing(h.g, id_u64)
	defer delete(out_edges)
	strings.write_string(&b, `,"outgoing":[`)
	for e, i in out_edges {
		if i > 0 {strings.write_byte(&b, ',')}
		fmt.sbprintf(&b, `{"target_id":%d,"weight":%f,"reasoning":`, e.target_id, e.weight)
		json_string(&b, e.reasoning)
		fmt.sbprintf(&b, `,"created_at":%d}`, e.created_at)
	}
	strings.write_string(&b, "]")

	// incoming edges
	in_edges := graph.incoming(h.g, id_u64)
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
// Returns JSON array of { id, score, text, source_id } sorted by score desc.
// Uses the same cosine + edge search as /ask but returns raw results instead of an LLM answer.
handle_find :: proc(h: ^Handler, query: string) -> (body: string, ok: bool) {
	if len(query) == 0 {
		return "", false
	}

	query_embedding, embed_ok := h.p.embed_text(h.p, query)
	if !embed_ok {
		return "", false
	}

	K := graph.cfg.default_find_k

	seen: map[u64]f32
	defer delete(seen)

	cosine_results := graph.find_thoughts(h.g, &query_embedding, K)
	defer delete(cosine_results)
	for r in cosine_results {
		existing, found := seen[r.id]
		if !found || r.score > existing {
			seen[r.id] = r.score
		}
	}

	edge_results := graph.find_edges(h.g, &query_embedding, K)
	defer delete(edge_results)
	for er in edge_results {
		edge := &h.g.edges[er.edge_index]
		for id in ([]u64{edge.source_id, edge.target_id}) {
			existing, found := seen[id]
			if !found || er.score > existing {
				seen[id] = er.score
			}
		}
	}

	if len(seen) == 0 {
		return "[]", true
	}

	Ranked :: struct {
		id:    u64,
		score: f32,
	}
	ranked := make([dynamic]Ranked, 0, len(seen))
	defer delete(ranked)
	for id, score in seen {
		append(&ranked, Ranked{id = id, score = score})
	}
	for i in 1 ..< len(ranked) {
		j := i
		for j > 0 && ranked[j].score > ranked[j - 1].score {
			ranked[j], ranked[j - 1] = ranked[j - 1], ranked[j]
			j -= 1
		}
	}

	b := strings.builder_make()
	strings.write_byte(&b, '[')
	for r, i in ranked {
		t := graph.get_thought(h.g, r.id)
		if t == nil {continue}
		if i > 0 {strings.write_byte(&b, ',')}
		fmt.sbprintf(&b, `{"id":%d,"score":%f,"text":`, r.id, r.score)
		json_string(&b, t.text)
		strings.write_string(&b, `,"source_id":`)
		json_string(&b, t.source_id)
		strings.write_byte(&b, '}')
	}
	strings.write_byte(&b, ']')
	return strings.to_string(b), true
}

// json_string writes a JSON-encoded string (with escaping) into a builder.
@(private)
json_string :: proc(b: ^strings.Builder, s: string) {
	strings.write_byte(b, '"')
	for c in s {
		switch c {
		case '"':
			strings.write_string(b, `\"`)
		case '\\':
			strings.write_string(b, `\\`)
		case '\n':
			strings.write_string(b, `\n`)
		case '\r':
			strings.write_string(b, `\r`)
		case '\t':
			strings.write_string(b, `\t`)
		case:
			strings.write_rune(b, c)
		}
	}
	strings.write_byte(b, '"')
}

// _handle_find_dispatch is called from the HTTP dispatch loop.
// path must already be verified to start with "/find"
_handle_find_dispatch :: proc(h: ^HTTP, req: ^http.Request, res: ^http.Response) {
	query := _query_param(req.url.query, "q")
	if len(query) == 0 {
		http.respond(res, http.Status.Bad_Request)
		return
	}

	sync.lock(&h.handler.mu)
	body, ok := handle_find(h.handler, query)
	sync.unlock(&h.handler.mu)

	if !ok {
		http.respond(res, http.Status.Internal_Server_Error)
		if len(body) > 0 {delete(body)}
		return
	}

	res.status = .OK
	http.headers_set_unsafe(&res.headers, "content-type", "application/json")
	http.body_set(res, body)
	delete(body)
	http.respond(res)
}
