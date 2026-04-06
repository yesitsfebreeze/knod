package http

import "core:bufio"
import "core:io"
import "core:log"
import "core:net"
import "core:strconv"
import "core:strings"

Body :: string

Body_Callback :: #type proc(user_data: rawptr, body: Body, err: Body_Error)

Body_Error :: bufio.Scanner_Error


body :: proc(req: ^Request, max_length: int = -1, user_data: rawptr, cb: Body_Callback) {
	assert(req._body_ok == nil, "you can only call body once per request")

	enc_header, ok := headers_get_unsafe(req.headers, "transfer-encoding")
	if ok && strings.has_suffix(enc_header, "chunked") {
		_body_chunked(req, max_length, user_data, cb)
	} else {
		_body_length(req, max_length, user_data, cb)
	}
}


body_url_encoded :: proc(plain: Body, allocator := context.temp_allocator) -> (res: map[string]string, ok: bool) {

	insert :: proc(m: ^map[string]string, plain: string, keys: int, vals: int, end: int, allocator := context.temp_allocator) -> bool {
		has_value := vals != -1
		key_end   := vals - 1 if has_value else end
		key       := plain[keys:key_end]
		val       := plain[vals:end] if has_value else ""

		// PERF: this could be a hot spot and I don't like that we allocate the decoded key and value here.
		keye := (net.percent_decode(key, allocator) or_return) if strings.index_byte(key, '%') > -1 else key
		vale := (net.percent_decode(val, allocator) or_return) if has_value && strings.index_byte(val, '%') > -1 else val

		m[keye] = vale
		return true
	}

	count := 1
	for b in plain {
		if b == '&' { count += 1 }
	}

	queries := make(map[string]string, count, allocator)

	keys := 0
	vals := -1
	for b, i in plain {
		switch b {
		case '=':
			vals = i + 1
		case '&':
			insert(&queries, plain, keys, vals, i) or_return
			keys = i + 1
			vals = -1
		}
	}

	insert(&queries, plain, keys, vals, len(plain)) or_return

	return queries, true
}

body_error_status :: proc(e: Body_Error) -> Status {
	switch t in e {
	case bufio.Scanner_Extra_Error:
		switch t {
		case .Too_Long:                            return .Payload_Too_Large
		case .Too_Short, .Bad_Read_Count:          return .Bad_Request
		case .Negative_Advance, .Advanced_Too_Far: return .Internal_Server_Error
		case .None:                                return .OK
		case:
			return .Internal_Server_Error
		}
	case io.Error:
		switch t {
		case .EOF, .Unknown, .No_Progress, .Unexpected_EOF:
			return .Bad_Request
		case .Empty, .Short_Write, .Buffer_Full, .Short_Buffer,
		     .Invalid_Write, .Negative_Read, .Invalid_Whence, .Invalid_Offset,
		     .Invalid_Unread, .Negative_Write, .Negative_Count,
		     .Permission_Denied, .No_Size, .Closed:
			return .Internal_Server_Error
		case .None:
			return .OK
		case:
			return .Internal_Server_Error
		}
	case: unreachable()
	}
}


_body_length :: proc(req: ^Request, max_length: int = -1, user_data: rawptr, cb: Body_Callback) {
	req._body_ok = false

	len, ok := headers_get_unsafe(req.headers, "content-length")
	if !ok {
		cb(user_data, "", nil)
		return
	}

	ilen, lenok := strconv.parse_int(len, 10)
	if !lenok {
		cb(user_data, "", .Bad_Read_Count)
		return
	}

	if max_length > -1 && ilen > max_length {
		cb(user_data, "", .Too_Long)
		return
	}

	if ilen == 0 {
		req._body_ok = true
		cb(user_data, "", nil)
		return
	}

	req._scanner.max_token_size = ilen

	req._scanner.split          = scan_num_bytes
	req._scanner.split_data     = rawptr(uintptr(ilen))

	req._body_ok = true
	scanner_scan(req._scanner, user_data, cb)
}

_body_chunked :: proc(req: ^Request, max_length: int = -1, user_data: rawptr, cb: Body_Callback) {
	req._body_ok = false

	on_scan :: proc(s: rawptr, size_line: string, err: bufio.Scanner_Error) {
		s := cast(^Chunked_State)s
		size_line := size_line

		if err != nil {
			s.cb(s.user_data, "", err)
			return
		}

		if semi := strings.index_byte(size_line, ';'); semi > -1 {
			size_line = size_line[:semi]
		}

		size, ok := strconv.parse_int(string(size_line), 16)
		if !ok {
			log.infof("Encountered an invalid chunk size when decoding a chunked body: %q", string(size_line))
			s.cb(s.user_data, "", .Bad_Read_Count)
			return
		}

		if size == 0 {
			scanner_scan(s.req._scanner, s, on_scan_trailer)
			return
		}

		if s.max_length > -1 && strings.builder_len(s.buf) + size > s.max_length {
			s.cb(s.user_data, "", .Too_Long)
			return
		}

		s.req._scanner.max_token_size = size

		s.req._scanner.split          = scan_num_bytes
		s.req._scanner.split_data     = rawptr(uintptr(size))

		scanner_scan(s.req._scanner, s, on_scan_chunk)
	}

	on_scan_chunk :: proc(s: rawptr, token: string, err: bufio.Scanner_Error) {
		s := cast(^Chunked_State)s

		if err != nil {
			s.cb(s.user_data, "", err)
			return
		}

		s.req._scanner.max_token_size = bufio.DEFAULT_MAX_SCAN_TOKEN_SIZE
		s.req._scanner.split          = scan_lines

		strings.write_string(&s.buf, token)

		on_scan_empty_line :: proc(s: rawptr, token: string, err: bufio.Scanner_Error) {
			s := cast(^Chunked_State)s

			if err != nil {
				s.cb(s.user_data, "", err)
				return
			}
			assert(len(token) == 0)

			scanner_scan(s.req._scanner, s, on_scan)
		}

		scanner_scan(s.req._scanner, s, on_scan_empty_line)
	}

	on_scan_trailer :: proc(s: rawptr, line: string, err: bufio.Scanner_Error) {
		s := cast(^Chunked_State)s

		if err != nil || len(line) == 0 {
			headers_delete_unsafe(&s.req.headers, "trailer")

			te_header := headers_get_unsafe(s.req.headers, "transfer-encoding")
			new_te_header := strings.trim_suffix(te_header, "chunked")

			s.req.headers.readonly = false
			headers_set_unsafe(&s.req.headers, "transfer-encoding", new_te_header)
			s.req.headers.readonly = true

			s.req._body_ok = true
			s.cb(s.user_data, strings.to_string(s.buf), nil)
			return
		}

		key, ok := header_parse(&s.req.headers, string(line))
		if !ok {
			log.infof("Invalid header when decoding chunked body: %q", string(line))
			s.cb(s.user_data, "", .Unknown)
			return
		}

		if !header_allowed_trailer(key) {
			log.infof("Invalid trailer header received, discarding it: %q", key)
			headers_delete(&s.req.headers, key)
		}

		scanner_scan(s.req._scanner, s, on_scan_trailer)
	}

	Chunked_State :: struct {
		req:        ^Request,
		max_length: int,
		user_data:  rawptr,
		cb:         Body_Callback,

		buf:        strings.Builder,
	}

	s := new(Chunked_State, context.temp_allocator)

	s.buf.buf.allocator = context.temp_allocator

	s.req        = req
	s.max_length = max_length
	s.user_data  = user_data
	s.cb         = cb

	s.req._scanner.split = scan_lines
	scanner_scan(s.req._scanner, s, on_scan)
}
