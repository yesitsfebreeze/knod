package graph

import "core:os"
import "core:strings"

import "../util"

stream_thought :: proc(fd: os.Handle, t: ^Thought) -> bool {
	rt := RecordType.THOUGHT
	if !write_val(fd, &rt) {return false}
	id := t.id
	if !write_val(fd, &id) {return false}
	if !write_string(fd, t.text) {return false}
	if !write_string(fd, t.source_id) {return false}
	if !write_embedding(fd, &t.embedding) {return false}
	ca := t.created_at
	if !write_val(fd, &ca) {return false}
	ac := t.access_count
	if !write_val(fd, &ac) {return false}
	la := t.last_accessed
	if !write_val(fd, &la) {return false}
	return true
}

stream_edge :: proc(fd: os.Handle, e: ^Edge) -> bool {
	rt := RecordType.EDGE
	if !write_val(fd, &rt) {return false}
	sid := e.source_id
	if !write_val(fd, &sid) {return false}
	tid := e.target_id
	if !write_val(fd, &tid) {return false}
	w := e.weight
	if !write_val(fd, &w) {return false}
	if !write_string(fd, e.reasoning) {return false}
	if !write_embedding(fd, &e.embedding) {return false}
	ca := e.created_at
	if !write_val(fd, &ca) {return false}
	return true
}

stream_open :: proc(g: ^Graph, path: string) -> bool {
	exists := os.exists(path)
	is_strand := strings.has_suffix(path, util.STRAND_EXTENSION)

	flags := os.O_WRONLY | os.O_CREATE | os.O_APPEND
	fd, err := os.open(path, flags, 0o644)
	if err != os.ERROR_NONE {
		return false
	}

	if !exists {
		if !write_header(fd, g) {
			os.close(fd)
			return false
		}
	} else if is_strand {
		end_off, off_ok := strand_graph_end_offset(path)
		if off_ok && end_off > 0 {
			os.seek(fd, end_off, os.SEEK_SET)
		}
	}

	g.stream_handle = fd
	return true
}

strand_graph_end_offset :: proc(path: string) -> (i64, bool) {
	data, ok := os.read_entire_file(path)
	if !ok {
		return 0, false
	}
	defer delete(data)

	if len(data) < 8 {
		return 0, false
	}

	off := 0
	magic := (^i32)(raw_data(data[off:]))^
	off += 4
	if magic != KNOD_MAGIC {
		return 0, false
	}
	off += 4

	for off < len(data) {
		if off + 1 > len(data) {break}
		section_tag := data[off]
		off += 1

		if off + 8 > len(data) {break}
		section_len := int((^i64)(raw_data(data[off:]))^)
		off += 8

		if section_tag == SECTION_GRAPH {
			return i64(off + section_len), true
		}
		off += section_len
	}

	return 0, false
}
