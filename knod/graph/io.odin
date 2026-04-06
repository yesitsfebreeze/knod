package graph

import "core:mem"
import "core:os"

save :: proc(g: ^Graph, path: string) -> bool {
	fd, err := os.open(path, os.O_WRONLY | os.O_CREATE | os.O_TRUNC, 0o644)
	if err != os.ERROR_NONE {return false}
	defer os.close(fd)

	if !write_container_header(fd) {return false}


	if !write_section_graph(fd, g) {return false}


	if len(g.limbo) > 0 {
		if !write_section_limbo(fd, g) {return false}
	}

	return true
}


load :: proc(g: ^Graph, path: string) -> (strand_offset: int, ok: bool) {
	data, read_ok := os.read_entire_file(path)
	if !read_ok {return 0, false}
	defer delete(data)

	if len(data) < 8 {return 0, false}

	off := 0
	magic := read_i32(data, &off)
	if magic != KNOD_MAGIC {return 0, false}
	version := read_i32(data, &off)
	if version != KNOD_VERSION {return 0, false}

	strand_off := 0
	for off < len(data) {
		if off + 9 > len(data) {break}
		section_tag := data[off]; off += 1
		section_len := int(read_i64(data, &off))
		section_start := off

		switch section_tag {
		case SECTION_GRAPH:
			if !load_graph_section(g, data[section_start:section_start + section_len]) {
				return 0, false
			}
			off = section_start + section_len
		case SECTION_LIMBO:
			load_limbo_section(g, data[section_start:section_start + section_len])
			off = section_start + section_len
		case SECTION_STRAND:
			strand_off = section_start
			off = section_start + section_len
		case:
			off = section_start + section_len
		}
	}

	return strand_off, true
}


save_with_strand :: proc(g: ^Graph, path: string, strand_bytes: []u8) -> bool {
	fd, err := os.open(path, os.O_WRONLY | os.O_CREATE | os.O_TRUNC, 0o644)
	if err != os.ERROR_NONE {return false}
	defer os.close(fd)

	if !write_container_header(fd) {return false}
	if !write_section_graph(fd, g) {return false}
	if len(g.limbo) > 0 {
		if !write_section_limbo(fd, g) {return false}
	}


	sec := SECTION_STRAND
	if !write_val(fd, &sec) {return false}
	slen := i64(len(strand_bytes))
	if !write_val(fd, &slen) {return false}
	if len(strand_bytes) > 0 {
		_, werr := os.write(fd, strand_bytes)
		if werr != os.ERROR_NONE {return false}
	}
	return true
}


load_strand_bytes :: proc(path: string) -> (strand_bytes: []u8, ok: bool) {
	data, read_ok := os.read_entire_file(path)
	if !read_ok {return nil, false}
	defer delete(data)

	if len(data) < 8 {return nil, false}

	off := 0
	magic := read_i32(data, &off)
	if magic != KNOD_MAGIC {return nil, false}
	off += 4

	for off < len(data) {
		if off + 9 > len(data) {break}
		section_tag := data[off]; off += 1
		section_len := int(read_i64(data, &off))
		section_start := off

		if section_tag == SECTION_STRAND {
			if section_len == 0 {return nil, true}
			buf := make([]u8, section_len)
			copy(buf, data[section_start:section_start + section_len])
			return buf, true
		}
		off = section_start + section_len
	}

	return nil, true
}


@(private = "file")
write_container_header :: proc(fd: os.Handle) -> bool {
	magic := KNOD_MAGIC; if !write_val(fd, &magic) {return false}
	ver := KNOD_VERSION; if !write_val(fd, &ver) {return false}
	return true
}

@(private = "file")
write_section_graph :: proc(fd: os.Handle, g: ^Graph) -> bool {

	sec := SECTION_GRAPH
	if !write_val(fd, &sec) {return false}


	len_offset, seek_err := os.seek(fd, 0, os.SEEK_CUR)
	if seek_err != os.ERROR_NONE {return false}
	zero_len: i64 = 0
	if !write_val(fd, &zero_len) {return false}

	content_start, _ := os.seek(fd, 0, os.SEEK_CUR)


	if !write_graph_inner(fd, g) {return false}

	content_end, _ := os.seek(fd, 0, os.SEEK_CUR)
	section_len := content_end - content_start


	os.seek(fd, len_offset, os.SEEK_SET)
	if !write_val(fd, &section_len) {return false}
	os.seek(fd, content_end, os.SEEK_SET)
	return true
}

@(private = "file")
write_section_limbo :: proc(fd: os.Handle, g: ^Graph) -> bool {
	sec := SECTION_LIMBO
	if !write_val(fd, &sec) {return false}

	len_offset, _ := os.seek(fd, 0, os.SEEK_CUR)
	zero_len: i64 = 0
	if !write_val(fd, &zero_len) {return false}
	content_start, _ := os.seek(fd, 0, os.SEEK_CUR)

	count := u32(len(g.limbo))
	if !write_val(fd, &count) {return false}
	for &lt in g.limbo {
		if !write_string(fd, lt.text) {return false}
		if !write_string(fd, lt.source) {return false}
		if !write_embedding(fd, &lt.embedding) {return false}
		ca := lt.created_at
		if !write_val(fd, &ca) {return false}
	}

	content_end, _ := os.seek(fd, 0, os.SEEK_CUR)
	section_len := content_end - content_start
	os.seek(fd, len_offset, os.SEEK_SET)
	if !write_val(fd, &section_len) {return false}
	os.seek(fd, content_end, os.SEEK_SET)
	return true
}

@(private = "file")
write_graph_inner :: proc(fd: os.Handle, g: ^Graph) -> bool {
	magic := LOG_MAGIC; if !write_val(fd, &magic) {return false}
	ver := LOG_VERSION; if !write_val(fd, &ver) {return false}
	nid := g.next_id; if !write_val(fd, &nid) {return false}
	if !write_string(fd, g.purpose) {return false}

	pc := g.profile_count
	if !write_val(fd, &pc) {return false}
	if g.profile_count > 0 {
		if !write_embedding(fd, &g.profile) {return false}
	}

	dc := u16(len(g.descriptors))
	if !write_val(fd, &dc) {return false}
	for _, &d in g.descriptors {
		if !write_string(fd, d.name) {return false}
		if !write_string(fd, d.text) {return false}
	}

	for _, &t in g.thoughts {
		if !stream_thought(fd, &t) {return false}
	}
	for i in 0 ..< len(g.edges) {
		if !stream_edge(fd, &g.edges[i]) {return false}
	}
	return true
}


@(private = "file")
load_graph_section :: proc(g: ^Graph, data: []u8) -> bool {
	if len(data) < 16 {return false}
	off := 0

	magic := read_i32(data, &off)
	if magic != LOG_MAGIC {return false}
	version := read_i32(data, &off)
	if version != LOG_VERSION && version != 1 {return false}
	g.next_id = read_u64(data, &off)

	purpose := read_str(data, &off)
	if len(purpose) > 0 {
		set_purpose(g, purpose)
		delete(purpose)
	}

	g.profile_count = read_u64(data, &off)
	if g.profile_count > 0 {
		g.profile = read_embedding(data, &off)
	}

	if version >= 2 {
		desc_count := int(read_u16(data, &off))
		for _ in 0 ..< desc_count {
			name := read_str(data, &off)
			text := read_str(data, &off)
			if len(name) > 0 {set_descriptor(g, name, text)}
			if len(name) > 0 {delete(name)}
			if len(text) > 0 {delete(text)}
		}
	}

	for off < len(data) {
		if off + 1 > len(data) {break}
		rt := RecordType(data[off]); off += 1
		switch rt {
		case .THOUGHT:
			if !read_thought(g, data, &off) {return false}
		case .EDGE:
			if !read_edge(g, data, &off) {return false}
		case .LIMBO_THOUGHT:
			_ = read_str(data, &off)
			_ = read_str(data, &off)
			_ = read_embedding(data, &off)
			_ = read_i64(data, &off)
		}
	}
	return true
}

@(private = "file")
load_limbo_section :: proc(g: ^Graph, data: []u8) {
	off := 0
	count := int(read_u32(data, &off))
	for _ in 0 ..< count {
		text := read_str(data, &off)
		source := read_str(data, &off)
		embedding := read_embedding(data, &off)
		created_at := read_i64(data, &off)
		append(
			&g.limbo,
			LimboThought {
				text = text,
				embedding = embedding,
				source = source,
				created_at = created_at,
			},
		)
	}
}


stream_thought :: proc(fd: os.Handle, t: ^Thought) -> bool {
	rt := RecordType.THOUGHT; if !write_val(fd, &rt) {return false}
	id := t.id; if !write_val(fd, &id) {return false}
	if !write_string(fd, t.text) {return false}
	if !write_string(fd, t.source) {return false}
	if !write_embedding(fd, &t.embedding) {return false}
	ca := t.created_at; if !write_val(fd, &ca) {return false}
	ac := t.access_count; if !write_val(fd, &ac) {return false}
	la := t.last_accessed; if !write_val(fd, &la) {return false}
	return true
}

stream_edge :: proc(fd: os.Handle, e: ^Edge) -> bool {
	rt := RecordType.EDGE; if !write_val(fd, &rt) {return false}
	sid := e.source_id; if !write_val(fd, &sid) {return false}
	tid := e.target_id; if !write_val(fd, &tid) {return false}
	w := e.weight; if !write_val(fd, &w) {return false}
	if !write_string(fd, e.reasoning) {return false}
	if !write_embedding(fd, &e.embedding) {return false}
	ca := e.created_at; if !write_val(fd, &ca) {return false}
	return true
}


@(private = "file")
read_i32 :: proc(data: []u8, off: ^int) -> i32 {
	if off^ + 4 > len(data) {return 0}
	val := (^i32)(raw_data(data[off^:]))^; off^ += 4; return val
}
@(private = "file")
read_u16 :: proc(data: []u8, off: ^int) -> u16 {
	if off^ + 2 > len(data) {return 0}
	val := (^u16)(raw_data(data[off^:]))^; off^ += 2; return val
}
@(private = "file")
read_u32 :: proc(data: []u8, off: ^int) -> u32 {
	if off^ + 4 > len(data) {return 0}
	val := (^u32)(raw_data(data[off^:]))^; off^ += 4; return val
}
@(private = "file")
read_u64 :: proc(data: []u8, off: ^int) -> u64 {
	if off^ + 8 > len(data) {return 0}
	val := (^u64)(raw_data(data[off^:]))^; off^ += 8; return val
}
@(private = "file")
read_i64 :: proc(data: []u8, off: ^int) -> i64 {
	if off^ + 8 > len(data) {return 0}
	val := (^i64)(raw_data(data[off^:]))^; off^ += 8; return val
}
@(private = "file")
read_str :: proc(data: []u8, off: ^int) -> string {
	slen := int(read_i32(data, off))
	if slen <= 0 || off^ + slen > len(data) {return ""}
	buf := make([]u8, slen)
	copy(buf, data[off^:off^ + slen])
	off^ += slen
	return string(buf)
}
@(private = "file")
read_embedding :: proc(data: []u8, off: ^int) -> Embedding {
	emb: Embedding
	size := EMBEDDING_DIM * size_of(f32)
	if off^ + size > len(data) {return emb}
	src := mem.slice_ptr((^f32)(raw_data(data[off^:])), EMBEDDING_DIM)
	for i in 0 ..< EMBEDDING_DIM {emb[i] = src[i]}
	off^ += size
	return emb
}
@(private = "file")
read_thought :: proc(g: ^Graph, data: []u8, off: ^int) -> bool {
	if off^ + 8 > len(data) {return false}
	id := read_u64(data, off)
	text := read_str(data, off)
	source := read_str(data, off)
	embedding := read_embedding(data, off)
	created_at := read_i64(data, off)
	access_count := read_u32(data, off)
	last_accessed := read_i64(data, off)
	t := Thought {
		id            = id,
		text          = text,
		embedding     = embedding,
		source        = source,
		created_at    = created_at,
		access_count  = access_count,
		last_accessed = last_accessed,
	}
	g.thoughts[id] = t
	return true
}
@(private = "file")
read_edge :: proc(g: ^Graph, data: []u8, off: ^int) -> bool {
	if off^ + 16 > len(data) {return false}
	source_id := read_u64(data, off)
	target_id := read_u64(data, off)
	weight := (^f32)(raw_data(data[off^:]))^; off^ += 4
	reasoning := read_str(data, off)
	embedding := read_embedding(data, off)
	created_at := read_i64(data, off)

	edge_idx := len(g.edges)
	e := Edge {
		source_id  = source_id,
		target_id  = target_id,
		weight     = weight,
		reasoning  = reasoning,
		embedding  = embedding,
		created_at = created_at,
	}
	append(&g.edges, e)

	if source_id not_in g.outgoing {g.outgoing[source_id] = make([dynamic]int)}
	out := g.outgoing[source_id]; append(&out, edge_idx); g.outgoing[source_id] = out

	if target_id not_in g.incoming {g.incoming[target_id] = make([dynamic]int)}
	inc := g.incoming[target_id]; append(&inc, edge_idx); g.incoming[target_id] = inc

	return true
}


write_val :: proc(fd: os.Handle, val: ^$T) -> bool {
	bytes := mem.slice_ptr(cast(^u8)val, size_of(T))
	_, err := os.write(fd, bytes)
	return err == os.ERROR_NONE
}
@(private = "file")
write_string :: proc(fd: os.Handle, s: string) -> bool {
	slen := i32(len(s))
	if !write_val(fd, &slen) {return false}
	if slen > 0 {
		_, err := os.write(fd, transmute([]u8)s)
		if err != os.ERROR_NONE {return false}
	}
	return true
}
write_embedding :: proc(fd: os.Handle, emb: ^Embedding) -> bool {
	bytes := mem.slice_ptr(cast(^u8)emb, EMBEDDING_DIM * size_of(f32))
	_, err := os.write(fd, bytes)
	return err == os.ERROR_NONE
}
