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
from ..Shard.types import GraphEvent

log = logging.getLogger(__name__)

_HEADER = struct.Struct("!I")  # uint32 big-endian


def recv_exact(sock: socket.socket, n: int) -> bytes:
	buf = bytearray()
	while len(buf) < n:
		chunk = sock.recv(n - len(buf))
		if not chunk:
			raise ConnectionError("connection closed")
		buf.extend(chunk)
	return bytes(buf)


def send_frame(sock: socket.socket, data: bytes):
	sock.sendall(_HEADER.pack(len(data)) + data)


def _dispatch(sock: socket.socket, handler: Handler, text: str, subscribers: set) -> str:
	"""Route a command string to the handler. Returns response string."""
	stripped = text.strip()

	if stripped == "STATUS":
		return handler.status()

	if stripped == "SUBSCRIBE":
		subscribers.add(sock)
		return "subscribed"

	if stripped == "UNSUBSCRIBE":
		subscribers.discard(sock)
		return "unsubscribed"

	if text.startswith("PURPOSE:"):
		purpose = text[len("PURPOSE:") :].strip()
		handler.set_purpose(purpose)
		return "ok"

	if text.startswith("ASK:"):
		query = text[len("ASK:") :]
		answer, _ = handler.ask(query)
		return answer

	if text.startswith("DESCRIPTOR_ADD:"):
		rest = text[len("DESCRIPTOR_ADD:") :]
		if "\n" in rest:
			name, desc_text = rest.split("\n", 1)
		else:
			name, desc_text = rest.strip(), ""
		handler.add_descriptor(name.strip(), desc_text)
		return "ok"

	if text.startswith("DESCRIPTOR_REMOVE:"):
		name = text[len("DESCRIPTOR_REMOVE:") :].strip()
		return "ok" if handler.remove_descriptor(name) else "not found"

	if text.startswith("INGEST_D:"):
		rest = text[len("INGEST_D:") :]
		if "\n" in rest:
			desc_name, ingest_text = rest.split("\n", 1)
		else:
			desc_name, ingest_text = "", rest
		desc_name = desc_name.strip()
		descriptor = handler.resolve_descriptor(desc_name) if desc_name else ""
		return handler.ingest(ingest_text, descriptor=descriptor)

	# Plain text → ingest
	if text:
		return handler.ingest(text)

	return ""


def _handle_framed(sock: socket.socket, handler: Handler, first_hdr: bytes, subscribers: set):
	"""Persistent framed session: read frames, dispatch, reply."""
	try:
		hdr = bytearray(first_hdr)
		while True:
			msg_len = _HEADER.unpack(hdr)[0]

			if msg_len == 0:
				# keepalive ping → echo empty frame
				send_frame(sock, b"")
				hdr = bytearray(recv_exact(sock, 4))
				continue

			body = recv_exact(sock, msg_len).decode("utf-8", errors="replace")
			reply = _dispatch(sock, handler, body, subscribers)
			send_frame(sock, reply.encode("utf-8"))

			hdr = bytearray(recv_exact(sock, 4))
	except (ConnectionError, OSError):
		pass
	finally:
		subscribers.discard(sock)
		sock.close()


def _handle_legacy(sock: socket.socket, handler: Handler, first_byte: bytes, subscribers: set):
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
		reply = _dispatch(sock, handler, text, subscribers)
		sock.sendall(reply.encode("utf-8"))
	except (ConnectionError, OSError):
		pass
	finally:
		sock.close()


def _handle_connection(sock: socket.socket, handler: Handler, subscribers: set):
	"""Detect protocol mode from first byte, then dispatch."""
	try:
		first = recv_exact(sock, 1)
	except ConnectionError:
		sock.close()
		return

	if first[0] >= 0x20:
		# ASCII → legacy single-shot
		_handle_legacy(sock, handler, first, subscribers)
	else:
		# Binary → framed session, read remaining 3 bytes of header
		try:
			rest = recv_exact(sock, 3)
		except ConnectionError:
			sock.close()
			return
		_handle_framed(sock, handler, first + rest, subscribers)


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
		# TCP owns its own subscriber list; sockets are transport-level state
		self._subscribers: set[socket.socket] = set()
		self._subs_lock = threading.Lock()
		# Wire up event listener to push events to all active subscribers
		handler.on("ingest_complete", self._on_event)
		handler.on("limbo_promoted", self._on_event)

	def _on_event(self, event: GraphEvent) -> None:
		"""Push a status update to all subscribed sockets."""
		msg = self.handler.status().encode("utf-8")
		hdr = _HEADER.pack(len(msg))
		with self._subs_lock:
			dead = set()
			for sock in self._subscribers:
				try:
					sock.sendall(hdr + msg)
				except OSError:
					dead.add(sock)
			self._subscribers -= dead

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
					args=(client, self.handler, self._subscribers),
					daemon=True,
				)
				t.start()
			except socket.timeout:
				continue
			except OSError:
				break
