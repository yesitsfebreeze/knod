"""
Smoke test: send one article to knod, wait for completion via push event, print result.
"""

import socket
import struct
import sys
import time

HOST = "127.0.0.1"
PORT = 7999


def recv_exact(sock, n):
	buf = bytearray()
	while len(buf) < n:
		chunk = sock.recv(n - len(buf))
		if not chunk:
			raise ConnectionError("connection closed")
		buf.extend(chunk)
	return bytes(buf)


def send_recv(sock, msg):
	data = msg.encode("utf-8")
	sock.sendall(struct.pack(">I", len(data)) + data)
	rlen = struct.unpack(">I", recv_exact(sock, 4))[0]
	return recv_exact(sock, rlen).decode("utf-8", errors="replace") if rlen else ""


def recv_frame(sock) -> str:
	"""Read one server-pushed frame (blocks until data arrives)."""
	hdr = recv_exact(sock, 4)
	length = struct.unpack(">I", hdr)[0]
	if length == 0:
		return ""
	return recv_exact(sock, length).decode("utf-8", errors="replace")


def parse_status(raw: str) -> dict:
	result = {}
	for part in raw.strip().split():
		if "=" in part:
			k, v = part.split("=", 1)
			try:
				result[k] = int(v)
			except ValueError:
				result[k] = v
	return result


sock = socket.socket()
sock.settimeout(600)
try:
	sock.connect((HOST, PORT))
except Exception as e:
	print(f"ERROR: could not connect to {HOST}:{PORT}: {e}")
	sys.exit(1)

# Set purpose
r = send_recv(sock, "PURPOSE:turtle specialist smoke test")
print(f"PURPOSE -> {r.strip()}")

# Use the first available corpus file
import os

corpus_dir = os.path.join(os.path.dirname(__file__), "corpus")
files = [f for f in os.listdir(corpus_dir) if f.endswith(".txt") and f != "manifest.txt"]
if not files:
	print("ERROR: no corpus files found")
	sys.exit(1)

fname = sorted(files)[0]
fpath = os.path.join(corpus_dir, fname)
with open(fpath, encoding="utf-8") as f:
	text = f.read()

print(f"Sending '{fname}' ({len(text):,} chars)...")
r = send_recv(sock, text)
print(f"INGEST -> {r.strip()}")

# Subscribe and wait for the push event instead of polling
resp = send_recv(sock, "SUBSCRIBE")
if resp.strip() != "subscribed":
	print(f"  WARNING: unexpected SUBSCRIBE reply: {resp!r}")

print("  Waiting for push event...", flush=True)
start = time.time()
completed = False

while True:
	try:
		raw = recv_frame(sock)
	except Exception as e:
		print(f"  recv error: {e}")
		break

	if not raw:
		continue  # keepalive / zero-length frame

	parts = parse_status(raw)
	if not parts:
		continue

	q = parts.get("queued", 0)
	inf = parts.get("in_flight", 0)
	t = parts.get("thoughts", 0)
	elapsed = int(time.time() - start)
	print(f"  [{elapsed:3d}s] queued={q} in_flight={inf} thoughts={t}")

	if q == 0 and inf == 0:
		print(f"SUCCESS: ingest completed. thoughts={t}")
		completed = True
		break

try:
	send_recv(sock, "UNSUBSCRIBE")
except Exception:
	pass

if not completed:
	print("TIMEOUT or ERROR: ingest did not complete within 600s")
	sys.exit(1)

sock.close()
