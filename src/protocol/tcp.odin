package protocol

import "core:net"
import "core:strings"

import log "../logger"

TCP :: struct {
	socket:  net.TCP_Socket,
	handler: ^Handler,
}

tcp_create :: proc(port: int, handler: ^Handler) -> (tcp: TCP, ok: bool) {
	endpoint := net.Endpoint {
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

tcp_poll :: proc(tcp: ^TCP) -> bool {
	client_socket, _, accept_err := net.accept_tcp(tcp.socket)
	if accept_err != .None {
		return false
	}

	net.set_blocking(client_socket, true)
	data := read_all_tcp(client_socket)
	defer delete(data)

	ASK_PREFIX :: "ASK:"
	PURPOSE_PREFIX :: "PURPOSE:"
	DESCRIPTOR_ADD_PREFIX :: "DESCRIPTOR_ADD:"
	DESCRIPTOR_RM_PREFIX :: "DESCRIPTOR_REMOVE:"
	DESCRIPTOR_LIST_PREFIX :: "DESCRIPTOR_LIST:"
	INGEST_D_PREFIX :: "INGEST_D:"

	text := string(data[:])

	if strings.has_prefix(text, PURPOSE_PREFIX) {
		purpose := strings.trim_space(text[len(PURPOSE_PREFIX):])
		handle_set_purpose(tcp.handler, purpose)
		net.send_tcp(client_socket, transmute([]u8)string("ok"))

	} else if strings.has_prefix(text, ASK_PREFIX) {
		query := text[len(ASK_PREFIX):]
		log.info("[ask] query: \"%s\"", query)
		answer, ok := handle_ask(tcp.handler, query)
		if ok {
			net.send_tcp(client_socket, transmute([]u8)answer)
			delete(answer)
		} else {
			net.send_tcp(client_socket, transmute([]u8)string("error: failed to process query"))
		}

	} else if strings.has_prefix(text, DESCRIPTOR_ADD_PREFIX) {
		payload := text[len(DESCRIPTOR_ADD_PREFIX):]
		newline := strings.index(payload, "\n")
		if newline >= 0 {
			name := strings.trim_space(payload[:newline])
			desc_text := payload[newline + 1:]
			if len(name) > 0 && len(desc_text) > 0 {
				handle_descriptor_add(tcp.handler, name, desc_text)
				net.send_tcp(client_socket, transmute([]u8)string("ok"))
			} else {
				net.send_tcp(client_socket, transmute([]u8)string("error: name and text required"))
			}
		} else {
			net.send_tcp(client_socket, transmute([]u8)string("error: missing newline separator"))
		}

	} else if strings.has_prefix(text, DESCRIPTOR_RM_PREFIX) {
		name := strings.trim_space(text[len(DESCRIPTOR_RM_PREFIX):])
		if handle_descriptor_remove(tcp.handler, name) {
			net.send_tcp(client_socket, transmute([]u8)string("ok"))
		} else {
			net.send_tcp(client_socket, transmute([]u8)string("not found"))
		}

	} else if strings.has_prefix(text, DESCRIPTOR_LIST_PREFIX) {
		list := handle_descriptor_list(tcp.handler)
		net.send_tcp(client_socket, transmute([]u8)list)
		delete(list)

	} else if strings.has_prefix(text, INGEST_D_PREFIX) {
		payload := text[len(INGEST_D_PREFIX):]
		newline := strings.index(payload, "\n")
		if newline >= 0 {
			desc_name := strings.trim_space(payload[:newline])
			ingest_text := payload[newline + 1:]
			descriptor_text := get_descriptor_text(tcp.handler, desc_name)
			if len(desc_name) > 0 && len(descriptor_text) == 0 {
				log.warn(
					"[ingest] descriptor \"%s\" not found, ingesting without descriptor",
					desc_name,
				)
			}
			log.info(
				"[ingest] received %d bytes (descriptor: \"%s\")",
				len(ingest_text),
				desc_name,
			)
			handle_ingest(tcp.handler, ingest_text, descriptor_text)
		} else {
			net.send_tcp(client_socket, transmute([]u8)string("error: missing newline separator"))
		}

	} else if len(data) > 0 {
		log.info("[ingest] received %d bytes", len(data))
		handle_ingest(tcp.handler, text)
	}

	net.shutdown(client_socket, .Send)
	net.close(client_socket)
	return true
}

@(private)
read_all_tcp :: proc(socket: net.TCP_Socket) -> [dynamic]u8 {
	data: [dynamic]u8
	recv_buf: [4096]u8

	for {
		n, err := net.recv_tcp(socket, recv_buf[:])
		if n > 0 {
			append(&data, ..recv_buf[:n])
		}
		if err != nil || n == 0 {
			break
		}
	}
	return data
}
