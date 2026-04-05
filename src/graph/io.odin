package graph

import "core:mem"
import "core:os"

save :: proc(g: ^Graph, path: string) -> bool {
	fd, err := os.open(path, os.O_WRONLY | os.O_CREATE | os.O_TRUNC, 0o644)
	if err != os.ERROR_NONE {
		return false
	}
	defer os.close(fd)

	if !write_header(fd, g) {return false}

	for _, &t in g.thoughts {
		if !stream_thought(fd, &t) {return false}
	}

	for i in 0 ..< len(g.edges) {
		if !stream_edge(fd, &g.edges[i]) {return false}
	}

	return true
}

load :: proc(g: ^Graph, path: string) -> bool {
	data, ok := os.read_entire_file(path)
	if !ok {
		return false
	}
	defer delete(data)

	if len(data) < 16 {
		return false
	}

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

	tag_count := int(read_u16(data, &off))
	if tag_count > 0 {
		tags := make([]Tag, tag_count)
		defer {
			for &tag in tags {
				delete(tag.label)
			}
			delete(tags)
		}
		for i in 0 ..< tag_count {
			tags[i].dim_index = read_u16(data, &off)
			tags[i].label = read_str(data, &off)
		}
		set_tags(g, tags)
	}

	if version >= 2 {
		desc_count := int(read_u16(data, &off))
		for _ in 0 ..< desc_count {
			name := read_str(data, &off)
			text := read_str(data, &off)
			if len(name) > 0 {
				set_descriptor(g, name, text)
			}
			if len(name) > 0 {delete(name)}
			if len(text) > 0 {delete(text)}
		}
	}

	for off < len(data) {
		if off + 1 > len(data) {break}
		rt := RecordType(data[off])
		off += 1

		switch rt {
		case .THOUGHT:
			if !read_thought(g, data, &off) {return false}
		case .EDGE:
			if !read_edge(g, data, &off) {return false}
		}
	}

	return true
}

@(private = "package")
write_header :: proc(fd: os.Handle, g: ^Graph) -> bool {
	magic: i32 = LOG_MAGIC
	version := LOG_VERSION
	nid := g.next_id
	if !write_val(fd, &magic) {return false}
	if !write_val(fd, &version) {return false}
	if !write_val(fd, &nid) {return false}
	if !write_string(fd, g.purpose) {return false}

	pc := g.profile_count
	if !write_val(fd, &pc) {return false}
	if g.profile_count > 0 {
		if !write_embedding(fd, &g.profile) {return false}
	}

	tc := u16(len(g.tags))
	if !write_val(fd, &tc) {return false}
	for &tag in g.tags {
		dim := tag.dim_index
		if !write_val(fd, &dim) {return false}
		if !write_string(fd, tag.label) {return false}
	}

	dc := u16(len(g.descriptors))
	if !write_val(fd, &dc) {return false}
	for _, &desc in g.descriptors {
		if !write_string(fd, desc.name) {return false}
		if !write_string(fd, desc.text) {return false}
	}

	return true
}

@(private = "package")
write_val :: proc(fd: os.Handle, val: ^$T) -> bool {
	bytes := mem.slice_ptr(cast(^u8)val, size_of(T))
	_, err := os.write(fd, bytes)
	return err == os.ERROR_NONE
}

@(private = "package")
write_string :: proc(fd: os.Handle, s: string) -> bool {
	slen := i32(len(s))
	if !write_val(fd, &slen) {return false}
	if slen > 0 {
		_, err := os.write(fd, transmute([]u8)s)
		if err != os.ERROR_NONE {return false}
	}
	return true
}

@(private = "package")
write_embedding :: proc(fd: os.Handle, emb: ^Embedding) -> bool {
	bytes := mem.slice_ptr(cast(^u8)emb, EMBEDDING_DIM * size_of(f32))
	_, err := os.write(fd, bytes)
	return err == os.ERROR_NONE
}

@(private = "file")
read_embedding :: proc(data: []u8, off: ^int) -> Embedding {
	emb: Embedding
	size := EMBEDDING_DIM * size_of(f32)
	if off^ + size > len(data) {return emb}
	src := mem.slice_ptr((^f32)(raw_data(data[off^:])), EMBEDDING_DIM)
	for i in 0 ..< EMBEDDING_DIM {
		emb[i] = src[i]
	}
	off^ += size
	return emb
}

@(private = "file")
read_thought :: proc(g: ^Graph, data: []u8, off: ^int) -> bool {
	if off^ + 8 > len(data) {return false}
	id := read_u64(data, off)
	text := read_str(data, off)
	source_id := read_str(data, off)
	embedding := read_embedding(data, off)
	created_at := read_i64(data, off)
	access_count := read_u32(data, off)
	last_accessed := read_i64(data, off)

	t := Thought {
		id            = id,
		text          = text,
		embedding     = embedding,
		source_id     = source_id,
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
	weight := read_f32(data, off)
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

	if source_id not_in g.outgoing {
		g.outgoing[source_id] = make([dynamic]int)
	}
	out := &g.outgoing[source_id]
	append(out, edge_idx)

	if target_id not_in g.incoming {
		g.incoming[target_id] = make([dynamic]int)
	}
	inc := &g.incoming[target_id]
	append(inc, edge_idx)

	return true
}

save_strand :: proc(g: ^Graph, strand_bytes: []u8, path: string) -> bool {
	fd, err := os.open(path, os.O_WRONLY | os.O_CREATE | os.O_TRUNC, 0o644)
	if err != os.ERROR_NONE {
		return false
	}
	defer os.close(fd)

	magic: i32 = KNOD_MAGIC
	version: i32 = KNOD_VERSION
	if !write_val(fd, &magic) {return false}
	if !write_val(fd, &version) {return false}

	sec_graph := SECTION_GRAPH
	if !write_val(fd, &sec_graph) {return false}

	graph_len_placeholder: i64 = 0
	if !write_val(fd, &graph_len_placeholder) {return false}

	graph_start: i64 = 17

	if !write_header(fd, g) {return false}
	for _, &t in g.thoughts {
		if !stream_thought(fd, &t) {return false}
	}
	for i in 0 ..< len(g.edges) {
		if !stream_edge(fd, &g.edges[i]) {return false}
	}

	graph_end, seek_err1 := os.seek(fd, 0, os.SEEK_CUR)
	if seek_err1 != os.ERROR_NONE {return false}
	graph_section_len := graph_end - graph_start


	_, seek_err2 := os.seek(fd, 9, os.SEEK_SET)
	if seek_err2 != os.ERROR_NONE {return false}
	if !write_val(fd, &graph_section_len) {return false}


	_, seek_err3 := os.seek(fd, graph_end, os.SEEK_SET)
	if seek_err3 != os.ERROR_NONE {return false}


	sec_strand := SECTION_STRAND
	if !write_val(fd, &sec_strand) {return false}
	strand_len := i64(len(strand_bytes))
	if !write_val(fd, &strand_len) {return false}
	if len(strand_bytes) > 0 {
		_, werr := os.write(fd, strand_bytes)
		if werr != os.ERROR_NONE {return false}
	}

	return true
}

load_strand :: proc(g: ^Graph, path: string) -> (strand_offset: int, ok: bool) {
	data, read_ok := os.read_entire_file(path)
	if !read_ok {
		return 0, false
	}
	defer delete(data)

	if len(data) < 8 {
		return 0, false
	}

	off := 0
	magic := read_i32(data, &off)
	if magic != KNOD_MAGIC {
		return 0, false
	}
	version := read_i32(data, &off)
	if version != KNOD_VERSION {
		return 0, false
	}


	for off < len(data) {
		if off + 1 > len(data) {break}
		section_tag := data[off]
		off += 1

		if off + 8 > len(data) {break}
		section_len := int((^i64)(raw_data(data[off:]))^)
		off += 8

		section_start := off

		switch section_tag {
		case SECTION_GRAPH:
			if !load_graph_section(g, data[section_start:section_start + section_len]) {
				return 0, false
			}
			off = section_start + section_len
		case SECTION_STRAND:
			return section_start, true
		case:
			off = section_start + section_len
		}
	}

	return 0, true
}

@(private = "file")
load_graph_section :: proc(g: ^Graph, data: []u8) -> bool {
	if len(data) < 16 {
		return false
	}

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

	tag_count := int(read_u16(data, &off))
	if tag_count > 0 {
		tags := make([]Tag, tag_count)
		defer {
			for &tag in tags {delete(tag.label)}
			delete(tags)
		}
		for i in 0 ..< tag_count {
			tags[i].dim_index = read_u16(data, &off)
			tags[i].label = read_str(data, &off)
		}
		set_tags(g, tags)
	}

	if version >= 2 {
		desc_count := int(read_u16(data, &off))
		for _ in 0 ..< desc_count {
			name := read_str(data, &off)
			text := read_str(data, &off)
			if len(name) > 0 {
				set_descriptor(g, name, text)
			}
			if len(name) > 0 {delete(name)}
			if len(text) > 0 {delete(text)}
		}
	}

	for off < len(data) {
		if off + 1 > len(data) {break}
		rt := RecordType(data[off])
		off += 1

		switch rt {
		case .THOUGHT:
			if !read_thought(g, data, &off) {return false}
		case .EDGE:
			if !read_edge(g, data, &off) {return false}
		}
	}

	return true
}

load_strand_bytes :: proc(path: string) -> (data: []u8, strand_offset: int, ok: bool) {
	file_data, read_ok := os.read_entire_file(path)
	if !read_ok {
		return nil, 0, false
	}

	if len(file_data) < 8 {
		delete(file_data)
		return nil, 0, false
	}

	off := 0
	magic := (^i32)(raw_data(file_data[off:]))^
	off += 4
	if magic != KNOD_MAGIC {
		delete(file_data)
		return nil, 0, false
	}
	off += 4

	for off < len(file_data) {
		if off + 1 > len(file_data) {break}
		section_tag := file_data[off]
		off += 1

		if off + 8 > len(file_data) {break}
		section_len := int((^i64)(raw_data(file_data[off:]))^)
		off += 8

		if section_tag == SECTION_STRAND {
			return file_data, off, true
		}
		off += section_len
	}

	return file_data, 0, true
}
