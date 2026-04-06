"""TCP server — binary framed protocol matching Odin implementation.

Wire protocol:
  - Peek first byte: >= 0x20 = legacy single-shot (ASCII), < 0x20 = framed session
  - Framed: [uint32 big-endian length][payload]  (length=0 is keepalive ping)

Commands (framed session):
  STATUS                         → status string
  SUBSCRIBE                      → "subscribed" (receive push events)
  UNSUBSCRIBE                    → "unsubscribed"
  PURPOSE:<text>                 → "ok"
  ASK:<query>                    → answer text
  DESCRIPTOR_ADD:<name>\n<text>  → "ok"
  DESCRIPTOR_REMOVE:<name>       → "ok" | "not found"
  INGEST_D:<descriptor>\n<text>  → "queued (N pending)" | "ok"
  <plain text>                   → ingest as-is
"""

import logging
import socket
import struct
import threading

from ..handler import Handler

log = logging.getLogger(__name__)

_HEADER = struct.Struct("!I")  # uint32 big-endian


def _recv_exact(sock: socket.socket, n: int) -> bytes:
	"""Read exactly n bytes or raise ConnectionError."""
	buf = bytearray()
	while len(buf) < n:
		chunk = sock.recv(n - len(buf))
		if not chunk:
			raise ConnectionError("connection closed")
		buf.extend(chunk)
	return bytes(buf)


def _send_frame(sock: socket.socket, data: bytes):
	"""Send a length-prefixed frame."""
	sock.sendall(_HEADER.pack(len(data)) + data)


def _dispatch(sock: socket.socket, handler: Handler, text: str) -> str:
	"""Route a command string to the handler. Returns response string."""
	stripped = text.strip()

	if stripped == "STATUS":
		return handler.handle_status()

	if stripped == "SUBSCRIBE":
		handler.subscribe(sock)
		return "subscribed"

	if stripped == "UNSUBSCRIBE":
		handler.unsubscribe(sock)
		return "unsubscribed"

	if text.startswith("PURPOSE:"):
		purpose = text[len("PURPOSE:") :].strip()
		handler.handle_set_purpose(purpose)
		return "ok"

	if text.startswith("ASK:"):
		query = text[len("ASK:") :]
		answer, _ = handler.handle_ask(query)
		return answer

	if text.startswith("DESCRIPTOR_ADD:"):
		rest = text[len("DESCRIPTOR_ADD:") :]
		if "\n" in rest:
			name, desc_text = rest.split("\n", 1)
		else:
			name, desc_text = rest.strip(), ""
		handler.handle_descriptor_add(name.strip(), desc_text)
		return "ok"

	if text.startswith("DESCRIPTOR_REMOVE:"):
		name = text[len("DESCRIPTOR_REMOVE:") :].strip()
		return "ok" if handler.handle_descriptor_remove(name) else "not found"

	if text.startswith("INGEST_D:"):
		rest = text[len("INGEST_D:") :]
		if "\n" in rest:
			desc_name, ingest_text = rest.split("\n", 1)
		else:
			desc_name, ingest_text = "", rest
		desc_name = desc_name.strip()
		descriptor = handler.graph.descriptors.get(desc_name, "") if desc_name else ""
		return handler.handle_ingest_queued(ingest_text, descriptor=descriptor)

	# Plain text → ingest
	if text:
		return handler.handle_ingest_queued(text)

	return ""


def _handle_framed(sock: socket.socket, handler: Handler, first_hdr: bytes):
	"""Persistent framed session: read frames, dispatch, reply."""
	try:
		hdr = bytearray(first_hdr)
		while True:
			msg_len = _HEADER.unpack(hdr)[0]

			if msg_len == 0:
				# keepalive ping → echo empty frame
				_send_frame(sock, b"")
				hdr = bytearray(_recv_exact(sock, 4))
				continue

			body = _recv_exact(sock, msg_len).decode("utf-8", errors="replace")
			reply = _dispatch(sock, handler, body)
			_send_frame(sock, reply.encode("utf-8"))

			hdr = bytearray(_recv_exact(sock, 4))
	except (ConnectionError, OSError):
		pass
	finally:
		handler.unsubscribe(sock)
		sock.close()


def _handle_legacy(sock: socket.socket, handler: Handler, first_byte: bytes):
	"""Legacy single-shot: read all data, dispatch once, close."""
	try:
		sock.settimeout(1.0)
		parts = [first_byte]
		while True:
			try:
				chunk = sock.recv(4096)
				if not chunk:
					break
				parts.append(chunk)
			except socket.timeout:
				break
		text = b"".join(parts).decode("utf-8", errors="replace")
		reply = _dispatch(sock, handler, text)
		sock.sendall(reply.encode("utf-8"))
	except (ConnectionError, OSError):
		pass
	finally:
		sock.close()


def _handle_connection(sock: socket.socket, handler: Handler):
	"""Detect protocol mode from first byte, then dispatch."""
	try:
		first = _recv_exact(sock, 1)
	except ConnectionError:
		sock.close()
		return

	if first[0] >= 0x20:
		# ASCII → legacy single-shot
		_handle_legacy(sock, handler, first)
	else:
		# Binary → framed session, read remaining 3 bytes of header
		try:
			rest = _recv_exact(sock, 3)
		except ConnectionError:
			sock.close()
			return
		_handle_framed(sock, handler, first + rest)


def _tcp(handler: Handler, port: int = 7999) -> "TCPServer":
	"""Create and start a TCP server on the given port."""
	srv = TCPServer(handler, port)
	srv.start()
	return srv


class TCPServer:
	def __init__(self, handler: Handler, port: int = 7999):
		self.handler = handler
		self.port = port
		self._socket: socket.socket | None = None
		self._thread: threading.Thread | None = None
		self._running = False

	def start(self):
		self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		self._socket.bind(("0.0.0.0", self.port))
		self._socket.listen(64)
		self._socket.settimeout(0.5)  # allow periodic shutdown checks
		self._running = True
		self._thread = threading.Thread(target=self._accept_loop, daemon=True)
		self._thread.start()
		log.info("tcp: listening on :%d", self.port)

	def stop(self):
		self._running = False
		if self._socket:
			self._socket.close()

	def _accept_loop(self):
		while self._running:
			try:
				client, addr = self._socket.accept()
				t = threading.Thread(
					target=_handle_connection,
					args=(client, self.handler),
					daemon=True,
				)
				t.start()
			except socket.timeout:
				continue
			except OSError:
				break
