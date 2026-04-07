import socket
import struct

_HEADER = struct.Struct("!I")

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
