// package provides a very simple (for now) HTTP/1.1 client.
package client

import "core:bufio"
import "core:bytes"
import "core:c"
import "core:encoding/json"
import "core:io"
import "core:log"
import "core:net"
import "core:strconv"
import "core:strings"

import http ".."
import openssl "../openssl"

Request :: struct {
	method:  http.Method,
	headers: http.Headers,
	cookies: [dynamic]http.Cookie,
	body:    bytes.Buffer,
}

request_init :: proc(r: ^Request, method := http.Method.Get, allocator := context.allocator) {
	r.method = method
	http.headers_init(&r.headers, allocator)
	r.cookies = make([dynamic]http.Cookie, allocator)
	bytes.buffer_init_allocator(&r.body, 0, 0, allocator)
}

request_destroy :: proc(r: ^Request) {
	delete(r.headers._kv)
	delete(r.cookies)
	bytes.buffer_destroy(&r.body)
}

with_json :: proc(r: ^Request, v: any, opt: json.Marshal_Options = {}) -> json.Marshal_Error {
	if r.method == .Get {r.method = .Post}
	http.headers_set_content_type(&r.headers, http.mime_to_content_type(.Json))

	stream := bytes.buffer_to_stream(&r.body)
	opt := opt
	json.marshal_to_writer(io.to_writer(stream), v, &opt) or_return
	return nil
}

get :: proc(target: string, allocator := context.allocator) -> (Response, Error) {
	r: Request
	request_init(&r, .Get, allocator)
	defer request_destroy(&r)

	return request(&r, target, allocator)
}

Request_Error :: enum {
	Ok,
	Invalid_Response_HTTP_Version,
	Invalid_Response_Method,
	Invalid_Response_Header,
	Invalid_Response_Cookie,
}

SSL_Error :: enum {
	Ok,
	Controlled_Shutdown,
	Fatal_Shutdown,
	SSL_Write_Failed,
}

Error :: union #shared_nil {
	net.Dial_Error,
	net.Parse_Endpoint_Error,
	net.Network_Error,
	net.TCP_Send_Error,
	bufio.Scanner_Error,
	Request_Error,
	SSL_Error,
}

request :: proc(
	request: ^Request,
	target: string,
	allocator := context.allocator,
) -> (
	res: Response,
	err: Error,
) {
	url, endpoint := parse_endpoint(target) or_return

	http.headers_set_close(&request.headers)

	req_buf := format_request(url, request, allocator)
	defer bytes.buffer_destroy(&req_buf)

	socket := net.dial_tcp(endpoint) or_return

	if url.scheme == "https" {
		ctx := openssl.SSL_CTX_new(openssl.TLS_client_method())
		ssl := openssl.SSL_new(ctx)
		openssl.SSL_set_fd(ssl, c.int(socket))

		chostname := strings.clone_to_cstring(url.host, allocator)
		defer delete(chostname, allocator)
		openssl.SSL_set_tlsext_host_name(ssl, chostname)

		switch openssl.SSL_connect(ssl) {
		case 2:
			err = SSL_Error.Controlled_Shutdown
			return
		case 1:
		case:
			err = SSL_Error.Fatal_Shutdown
			return
		}

		buf := bytes.buffer_to_bytes(&req_buf)
		to_write := len(buf)
		for to_write > 0 {
			ret := openssl.SSL_write(ssl, raw_data(buf), c.int(to_write))
			if ret <= 0 {
				err = SSL_Error.SSL_Write_Failed
				return
			}

			to_write -= int(ret)
		}

		return parse_response(SSL_Communication{ssl = ssl, ctx = ctx, socket = socket}, allocator)
	}

	net.send_tcp(socket, bytes.buffer_to_bytes(&req_buf)) or_return
	return parse_response(socket, allocator)
}

Response :: struct {
	status:    http.Status,
	headers:   http.Headers,
	cookies:   [dynamic]http.Cookie,
	_socket:   Communication,
	_body:     bufio.Scanner,
	_body_err: Body_Error,
}

response_destroy :: proc(
	res: ^Response,
	body: Maybe(Body_Type) = nil,
	was_allocation := false,
	body_allocator := context.allocator,
) {
	for k, v in res.headers._kv {
		delete(v, res.headers._kv.allocator)
		delete(k, res.headers._kv.allocator)
	}

	delete(res.headers._kv)

	bufio.scanner_destroy(&res._body)

	for cookie in res.cookies {
		delete(cookie._raw)
	}
	delete(res.cookies)

	if body != nil {
		body_destroy(body.(Body_Type), was_allocation, body_allocator)
	}

	switch comm in res._socket {
	case net.TCP_Socket:
		net.close(comm)
	case SSL_Communication:
		openssl.SSL_free(comm.ssl)
		openssl.SSL_CTX_free(comm.ctx)
		net.close(comm.socket)
	}
}

Body_Error :: enum {
	None,
	No_Length,
	Invalid_Length,
	Too_Long,
	Scan_Failed,
	Invalid_Chunk_Size,
	Invalid_Trailer_Header,
}

Body_Plain :: string

Body_Url_Encoded :: map[string]string

Body_Type :: union #no_nil {
	Body_Plain,
	Body_Url_Encoded,
	Body_Error,
}

body_destroy :: proc(body: Body_Type, was_allocation: bool, allocator := context.allocator) {
	switch b in body {
	case Body_Plain:
		if was_allocation {delete(b, allocator)}
	case Body_Url_Encoded:
		for k, v in b {
			delete(k, b.allocator)
			delete(v, b.allocator)
		}
		delete(b)
	case Body_Error:
	}
}

response_body :: proc(
	res: ^Response,
	max_length := -1,
	allocator := context.allocator,
) -> (
	body: Body_Type,
	was_allocation: bool,
	err: Body_Error,
) {
	defer res._body_err = err
	assert(res._body_err == nil)
	body, was_allocation, err = _parse_body(&res.headers, &res._body, max_length, allocator)
	return
}

_parse_body :: proc(
	headers: ^http.Headers,
	_body: ^bufio.Scanner,
	max_length := -1,
	allocator := context.allocator,
) -> (
	body: Body_Type,
	was_allocation: bool,
	err: Body_Error,
) {

	enc, has_enc := http.headers_get_unsafe(headers^, "transfer-encoding")
	length, has_length := http.headers_get_unsafe(headers^, "content-length")
	switch {
	case has_enc && strings.has_suffix(enc, "chunked"):
		was_allocation = true
		body = _response_body_chunked(headers, _body, max_length, allocator) or_return

	case has_length:
		body = _response_body_length(_body, max_length, length) or_return

	case:
		body = _response_till_close(_body, max_length) or_return
	}

	if typ, ok := http.headers_get_unsafe(headers^, "content-type");
	   ok && typ == "application/x-www-form-urlencoded" {
		plain := body.(Body_Plain)
		defer if was_allocation {delete(plain, allocator)}

		keyvalues := strings.split(plain, "&", allocator)
		defer delete(keyvalues, allocator)

		queries := make(Body_Url_Encoded, len(keyvalues), allocator)
		for keyvalue in keyvalues {
			seperator := strings.index(keyvalue, "=")
			if seperator == -1 {
				queries[keyvalue] = ""
				continue
			}

			key, key_decoded_ok := net.percent_decode(keyvalue[:seperator], allocator)
			if !key_decoded_ok {
				log.warnf("url encoded body key %q could not be decoded", keyvalue[:seperator])
				continue
			}

			val, val_decoded_ok := net.percent_decode(keyvalue[seperator + 1:], allocator)
			if !val_decoded_ok {
				log.warnf(
					"url encoded body value %q for key %q could not be decoded",
					keyvalue[seperator + 1:],
					key,
				)
				continue
			}

			queries[key] = val
		}

		body = queries
	}

	return
}

_response_till_close :: proc(_body: ^bufio.Scanner, max_length: int) -> (string, Body_Error) {
	_body.max_token_size = max_length
	defer _body.max_token_size = bufio.DEFAULT_MAX_SCAN_TOKEN_SIZE

	_body.split = proc(
		data: []byte,
		at_eof: bool,
	) -> (
		advance: int,
		token: []byte,
		err: bufio.Scanner_Error,
		final_token: bool,
	) {
		if at_eof {
			return len(data), data, nil, true
		}

		return
	}
	defer _body.split = bufio.scan_lines

	if !bufio.scanner_scan(_body) {
		if bufio.scanner_error(_body) == .Too_Long {
			return "", .Too_Long
		}

		return "", .Scan_Failed
	}

	return bufio.scanner_text(_body), .None
}

_response_body_length :: proc(
	_body: ^bufio.Scanner,
	max_length: int,
	len: string,
) -> (
	string,
	Body_Error,
) {
	ilen, lenok := strconv.parse_int(len, 10)
	if !lenok {
		return "", .Invalid_Length
	}

	if max_length > -1 && ilen > max_length {
		return "", .Too_Long
	}

	if ilen == 0 {
		return "", nil
	}

	context.user_index = ilen

	_body.max_token_size = ilen
	defer _body.max_token_size = bufio.DEFAULT_MAX_SCAN_TOKEN_SIZE

	_body.split = scan_num_bytes
	defer _body.split = bufio.scan_lines

	log.debugf("scanning %i bytes body", ilen)

	if !bufio.scanner_scan(_body) {
		return "", .Scan_Failed
	}

	return bufio.scanner_text(_body), .None
}

_response_body_chunked :: proc(
	headers: ^http.Headers,
	_body: ^bufio.Scanner,
	max_length: int,
	allocator := context.allocator,
) -> (
	body: string,
	err: Body_Error,
) {
	body_buff: bytes.Buffer

	bytes.buffer_init_allocator(&body_buff, 0, 0, allocator)
	defer if err != nil {bytes.buffer_destroy(&body_buff)}

	for {
		if !bufio.scanner_scan(_body) {
			return "", .Scan_Failed
		}

		size_line := bufio.scanner_bytes(_body)

		if semi := bytes.index_byte(size_line, ';'); semi > -1 {
			size_line = size_line[:semi]
		}

		size, ok := strconv.parse_int(string(size_line), 16)
		if !ok {
			err = .Invalid_Chunk_Size
			return
		}
		if size == 0 {break}

		if max_length > -1 && bytes.buffer_length(&body_buff) + size > max_length {
			return "", .Too_Long
		}

		context.user_index = size

		_body.max_token_size = size
		_body.split = scan_num_bytes

		if !bufio.scanner_scan(_body) {
			return "", .Scan_Failed
		}

		_body.max_token_size = bufio.DEFAULT_MAX_SCAN_TOKEN_SIZE
		_body.split = bufio.scan_lines

		bytes.buffer_write(&body_buff, bufio.scanner_bytes(_body))

		if !bufio.scanner_scan(_body) {
			return "", .Scan_Failed
		}
		assert(bufio.scanner_text(_body) == "")
	}

	if !bufio.scanner_scan(_body) || bufio.scanner_text(_body) != "" {
		return "", .Scan_Failed
	}

	for {
		if !bufio.scanner_scan(_body) {
			break
		}

		line := bufio.scanner_text(_body)

		if line == "" {
			break
		}

		key, ok := http.header_parse(headers, line)
		if !ok {
			return "", .Invalid_Trailer_Header
		}

		if !http.header_allowed_trailer(key) {
			http.headers_delete(headers, key)
		}
	}

	if http.headers_has_unsafe(headers^, "trailer") {
		http.headers_delete_unsafe(headers, "trailer")
	}

	te := strings.trim_suffix(http.headers_get_unsafe(headers^, "transfer-encoding"), "chunked")

	headers.readonly = false
	http.headers_set_unsafe(headers, "transfer-encoding", te)
	headers.readonly = true

	return bytes.buffer_to_string(&body_buff), .None
}

@(private)
scan_num_bytes :: proc(
	data: []byte,
	at_eof: bool,
) -> (
	advance: int,
	token: []byte,
	err: bufio.Scanner_Error,
	final_token: bool,
) {
	n := context.user_index
	if at_eof && len(data) < n {
		return
	}

	if len(data) < n {
		return
	}

	return n, data[:n], nil, false
}
