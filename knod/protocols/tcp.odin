package protocols

// tcp.odin — TCP server.
// Near-verbatim copy of archive/src/protocol/tcp.odin.
// Frame format (both directions): [uint32 big-endian length][payload bytes]

import "base:intrinsics"
import "core:encoding/endian"
import "core:fmt"
import "core:mem/virtual"
import "core:net"
import "core:strings"
import "core:sync"
import "core:thread"

import log "../logger"

TCP :: struct {
	socket:  net.TCP_Socket,
	handler: ^Handler,
}

tcp_create :: proc(port: int, handler: ^Handler) -> (tcp: TCP, ok: bool) {
	endpoint := net.Endpoint{
		address = net.IP4_Address{0, 0, 0, 0},
		port    = port,
	}

	socket, listen_err := net.listen_tcp(endpoint)
	if listen_err != nil {
		log.err("could not listen on port %d: %v", port, listen_err)
		return {}, false
	}

	set_err := net.set_blocking(socket, false)
	if set_err != nil {
		log.warn("could not set non-blocking: %v", set_err)
	}

	log.info("tcp: listening on :%d", port)
	return TCP{socket = socket, handler = handler}, true
}

tcp_destroy :: proc(tcp: ^TCP) {
	net.close(tcp.socket)
}

@(private)
_Conn_Job :: struct {
	client:  net.TCP_Socket,
	handler: ^Handler,
}

// tcp_poll drains every pending incoming connection, spawning a goroutine per connection.
tcp_poll :: proc(tcp: ^TCP) -> bool {
	accepted_any := false

	for {
		client_socket, _, accept_err := net.accept_tcp(tcp.socket)
		if accept_err != .None {break}
		accepted_any = true

		job         := new(_Conn_Job)
		job.client   = client_socket
		job.handler  = tcp.handler

		t := thread.create(_conn_goroutine)
		t.user_args[0] = job
		t.init_context = context
		intrinsics.atomic_or(&t.flags, {.Self_Cleanup})
		thread.start(t)
	}

	return accepted_any
}

@(private)
_conn_goroutine :: proc(t: ^thread.Thread) {
	job := (^_Conn_Job)(t.user_args[0])
	defer free(job)

	temp_arena: virtual.Arena
	_ = virtual.arena_init_growing(&temp_arena)
	context.temp_allocator = virtual.arena_allocator(&temp_arena)
	defer virtual.arena_destroy(&temp_arena)

	client_socket := job.client
	h             := job.handler

	net.set_blocking(client_socket, true)

	// Peek at first byte to decide protocol:
	//   >= 0x20 (printable ASCII) → legacy single-shot
	//   < 0x20                    → framed persistent session
	peek: [4]u8
	n_peek, peek_err := net.recv_tcp(client_socket, peek[:1])
	if peek_err != nil || n_peek == 0 {
		net.close(client_socket)
		return
	}

	if peek[0] >= 0x20 {
		rest := read_all_tcp(client_socket)
		defer delete(rest)
		full: [dynamic]u8
		defer delete(full)
		append(&full, peek[0])
		append(&full, ..rest[:])
		_dispatch_once(client_socket, h, full[:])
		net.shutdown(client_socket, .Send)
		net.close(client_socket)
	} else {
		_, err := _recv_exact(client_socket, peek[1:4])
		if err {
			net.close(client_socket)
			return
		}
		_conn_session(client_socket, h, peek[:4])
		net.close(client_socket)
	}
}

@(private)
_conn_session :: proc(sock: net.TCP_Socket, h: ^Handler, first_len_buf: []u8) {
	defer handler_unsubscribe(h, sock)

	len_buf: [4]u8
	copy(len_buf[:], first_len_buf)

	for {
		msg_len, _ := endian.get_u32(len_buf[:], .Big)

		if msg_len == 0 {
			reply_hdr: [4]u8
			endian.put_u32(reply_hdr[:], .Big, 0)
			net.send_tcp(sock, reply_hdr[:])
			_, err := _recv_exact(sock, len_buf[:])
			if err {return}
			continue
		}

		body := make([]u8, msg_len)
		defer delete(body)
		if !_recv_exact_slice(sock, body) {return}

		reply := _dispatch_framed(sock, h, string(body))
		defer delete(reply)

		reply_hdr: [4]u8
		endian.put_u32(reply_hdr[:], .Big, u32(len(reply)))
		net.send_tcp(sock, reply_hdr[:])
		if len(reply) > 0 {
			net.send_tcp(sock, transmute([]u8)reply)
		}

		_, err := _recv_exact(sock, len_buf[:])
		if err {return}
	}
}

@(private)
_dispatch_framed :: proc(sock: net.TCP_Socket, h: ^Handler, text: string) -> string {
	ASK_PREFIX             :: "ASK:"
	STATUS_CMD             :: "STATUS"
	SUBSCRIBE_CMD          :: "SUBSCRIBE"
	UNSUBSCRIBE_CMD        :: "UNSUBSCRIBE"
	PURPOSE_PREFIX         :: "PURPOSE:"
	DESCRIPTOR_ADD_PREFIX  :: "DESCRIPTOR_ADD:"
	DESCRIPTOR_RM_PREFIX   :: "DESCRIPTOR_REMOVE:"
	DESCRIPTOR_LIST_PREFIX :: "DESCRIPTOR_LIST:"
	INGEST_D_PREFIX        :: "INGEST_D:"

	if strings.trim_space(text) == STATUS_CMD {
		return handle_status(h)
	}

	if strings.trim_space(text) == SUBSCRIBE_CMD {
		handler_subscribe(h, sock)
		return strings.clone("subscribed")
	}

	if strings.trim_space(text) == UNSUBSCRIBE_CMD {
		handler_unsubscribe(h, sock)
		return strings.clone("unsubscribed")
	}

	if strings.has_prefix(text, PURPOSE_PREFIX) {
		purpose := strings.trim_space(text[len(PURPOSE_PREFIX):])
		sync.lock(&h.mu)
		handle_set_purpose(h, purpose)
		sync.unlock(&h.mu)
		return strings.clone("ok")
	}

	if strings.has_prefix(text, ASK_PREFIX) {
		query := text[len(ASK_PREFIX):]
		log.info("[ask] query: \"%s\"", query)
		sync.lock(&h.mu)
		answer, ok := handle_ask(h, query)
		sync.unlock(&h.mu)
		if ok {return answer}
		return strings.clone("")
	}

	if strings.has_prefix(text, DESCRIPTOR_ADD_PREFIX) {
		payload := text[len(DESCRIPTOR_ADD_PREFIX):]
		newline  := strings.index(payload, "\n")
		if newline >= 0 {
			name      := strings.trim_space(payload[:newline])
			desc_text := payload[newline + 1:]
			if len(name) > 0 && len(desc_text) > 0 {
				sync.lock(&h.mu)
				handle_descriptor_add(h, name, desc_text)
				sync.unlock(&h.mu)
				return strings.clone("ok")
			}
			return strings.clone("error: name and text required")
		}
		return strings.clone("error: missing newline separator")
	}

	if strings.has_prefix(text, DESCRIPTOR_RM_PREFIX) {
		name := strings.trim_space(text[len(DESCRIPTOR_RM_PREFIX):])
		sync.lock(&h.mu)
		ok := handle_descriptor_remove(h, name)
		sync.unlock(&h.mu)
		if ok {return strings.clone("ok")}
		return strings.clone("not found")
	}

	if strings.has_prefix(text, DESCRIPTOR_LIST_PREFIX) {
		sync.lock(&h.mu)
		list := handle_descriptor_list(h)
		sync.unlock(&h.mu)
		return list
	}

	if strings.has_prefix(text, INGEST_D_PREFIX) {
		payload  := text[len(INGEST_D_PREFIX):]
		newline  := strings.index(payload, "\n")
		if newline >= 0 {
			desc_name   := strings.trim_space(payload[:newline])
			ingest_text := payload[newline + 1:]
			sync.lock(&h.mu)
			descriptor_text := get_descriptor_text(h, desc_name)
			sync.unlock(&h.mu)
			if len(desc_name) > 0 && len(descriptor_text) == 0 {
				log.warn("[ingest] descriptor \"%s\" not found", desc_name)
			}
			log.info("[ingest] received %d bytes (descriptor: \"%s\")", len(ingest_text), desc_name)
			if handler_enqueue(h, ingest_text, descriptor_text) {
				pending := queue_len(&h.queue)
				return fmt.aprintf("queued (%d pending)", pending)
			}
			sync.lock(&h.mu)
			handle_ingest(h, ingest_text, descriptor_text)
			sync.unlock(&h.mu)
			return strings.clone("ok")
		}
		return strings.clone("error: missing newline separator")
	}

	if len(text) > 0 {
		log.info("[ingest] received %d bytes", len(text))
		if handler_enqueue(h, text) {
			pending := queue_len(&h.queue)
			return fmt.aprintf("queued (%d pending)", pending)
		}
		sync.lock(&h.mu)
		handle_ingest(h, text)
		sync.unlock(&h.mu)
		return strings.clone("ok")
	}

	return strings.clone("")
}

@(private)
_dispatch_once :: proc(sock: net.TCP_Socket, h: ^Handler, data: []u8) {
	text := string(data)

	ASK_PREFIX             :: "ASK:"
	STATUS_CMD             :: "STATUS"
	PURPOSE_PREFIX         :: "PURPOSE:"
	DESCRIPTOR_ADD_PREFIX  :: "DESCRIPTOR_ADD:"
	DESCRIPTOR_RM_PREFIX   :: "DESCRIPTOR_REMOVE:"
	DESCRIPTOR_LIST_PREFIX :: "DESCRIPTOR_LIST:"
	INGEST_D_PREFIX        :: "INGEST_D:"

	if strings.has_prefix(text, PURPOSE_PREFIX) {
		purpose := strings.trim_space(text[len(PURPOSE_PREFIX):])
		sync.lock(&h.mu)
		handle_set_purpose(h, purpose)
		sync.unlock(&h.mu)
		net.send_tcp(sock, transmute([]u8)string("ok"))

	} else if strings.has_prefix(text, ASK_PREFIX) {
		query := text[len(ASK_PREFIX):]
		log.info("[ask] query: \"%s\"", query)
		sync.lock(&h.mu)
		answer, ok := handle_ask(h, query)
		sync.unlock(&h.mu)
		if ok {
			net.send_tcp(sock, transmute([]u8)answer)
			delete(answer)
		}

	} else if strings.trim_space(text) == STATUS_CMD {
		status := handle_status(h)
		net.send_tcp(sock, transmute([]u8)status)
		delete(status)

	} else if strings.has_prefix(text, DESCRIPTOR_ADD_PREFIX) {
		payload := text[len(DESCRIPTOR_ADD_PREFIX):]
		newline  := strings.index(payload, "\n")
		if newline >= 0 {
			name      := strings.trim_space(payload[:newline])
			desc_text := payload[newline + 1:]
			if len(name) > 0 && len(desc_text) > 0 {
				sync.lock(&h.mu)
				handle_descriptor_add(h, name, desc_text)
				sync.unlock(&h.mu)
				net.send_tcp(sock, transmute([]u8)string("ok"))
			} else {
				net.send_tcp(sock, transmute([]u8)string("error: name and text required"))
			}
		} else {
			net.send_tcp(sock, transmute([]u8)string("error: missing newline separator"))
		}

	} else if strings.has_prefix(text, DESCRIPTOR_RM_PREFIX) {
		name := strings.trim_space(text[len(DESCRIPTOR_RM_PREFIX):])
		sync.lock(&h.mu)
		ok := handle_descriptor_remove(h, name)
		sync.unlock(&h.mu)
		if ok {
			net.send_tcp(sock, transmute([]u8)string("ok"))
		} else {
			net.send_tcp(sock, transmute([]u8)string("not found"))
		}

	} else if strings.has_prefix(text, DESCRIPTOR_LIST_PREFIX) {
		sync.lock(&h.mu)
		list := handle_descriptor_list(h)
		sync.unlock(&h.mu)
		net.send_tcp(sock, transmute([]u8)list)
		delete(list)

	} else if strings.has_prefix(text, INGEST_D_PREFIX) {
		payload := text[len(INGEST_D_PREFIX):]
		newline  := strings.index(payload, "\n")
		if newline >= 0 {
			desc_name   := strings.trim_space(payload[:newline])
			ingest_text := payload[newline + 1:]
			sync.lock(&h.mu)
			descriptor_text := get_descriptor_text(h, desc_name)
			sync.unlock(&h.mu)
			if len(desc_name) > 0 && len(descriptor_text) == 0 {
				log.warn("[ingest] descriptor \"%s\" not found", desc_name)
			}
			log.info("[ingest] received %d bytes (descriptor: \"%s\")", len(ingest_text), desc_name)
			if handler_enqueue(h, ingest_text, descriptor_text) {
				pending := queue_len(&h.queue)
				reply   := fmt.aprintf("queued (%d pending)", pending)
				net.send_tcp(sock, transmute([]u8)reply)
				delete(reply)
			} else {
				sync.lock(&h.mu)
				handle_ingest(h, ingest_text, descriptor_text)
				sync.unlock(&h.mu)
			}
		} else {
			net.send_tcp(sock, transmute([]u8)string("error: missing newline separator"))
		}

	} else if len(data) > 0 {
		log.info("[ingest] received %d bytes", len(data))
		if handler_enqueue(h, text) {
			pending := queue_len(&h.queue)
			reply   := fmt.aprintf("queued (%d pending)", pending)
			net.send_tcp(sock, transmute([]u8)reply)
			delete(reply)
		} else {
			sync.lock(&h.mu)
			handle_ingest(h, text)
			sync.unlock(&h.mu)
		}
	}
}

@(private)
_recv_exact :: proc(sock: net.TCP_Socket, buf: []u8) -> (int, bool) {
	got := 0
	for got < len(buf) {
		n, err := net.recv_tcp(sock, buf[got:])
		if n > 0 {got += n}
		if err != nil || n == 0 {return got, true}
	}
	return got, false
}

@(private)
_recv_exact_slice :: proc(sock: net.TCP_Socket, buf: []u8) -> bool {
	_, err := _recv_exact(sock, buf)
	return !err
}

@(private)
read_all_tcp :: proc(socket: net.TCP_Socket) -> [dynamic]u8 {
	data: [dynamic]u8
	recv_buf: [4096]u8
	for {
		n, err := net.recv_tcp(socket, recv_buf[:])
		if n > 0 {append(&data, ..recv_buf[:n])}
		if err != nil || n == 0 {break}
	}
	return data
}
