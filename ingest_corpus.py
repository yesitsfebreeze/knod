"""
Feed cached corpus files into a running knod instance via persistent TCP session.

Uses a single TCP connection with length-prefixed framing:
  send: [uint32 big-endian length][payload bytes]
  recv: [uint32 big-endian length][reply bytes]

This avoids per-article TCP handshake overhead and is faster than HTTP for
local testing.  Falls back to a plain STATUS check via a fresh connection if
needed (the server auto-detects framed vs legacy by the first byte).

Expects:
  - knod running on TCP port 7999
  - corpus/ directory populated by fetch_corpus.py

Usage: python ingest_corpus.py [--purpose PURPOSE] [--only on|off] [--delay SECONDS] [--passes N]
  --purpose   set the node purpose before ingesting (default: turtle specialist)
  --only      only ingest "on"-topic or "off"-topic files
  --delay     seconds to wait between articles (default: 0)
  --passes    number of times to loop through the corpus (default: 1)
"""

import argparse
import os
import socket
import struct
import sys
import time

HOST = "127.0.0.1"
TCP_PORT = 7999
CORPUS_DIR = "corpus"
MANIFEST = os.path.join(CORPUS_DIR, "manifest.txt")

DEFAULT_PURPOSE = (
    "specialist in turtles, their anatomy, biology, ecology, and conservation"
)


class KnodSession:
    """Persistent framed TCP session to knod."""

    def __init__(self, host: str = HOST, port: int = TCP_PORT, timeout: float = 600.0):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(timeout)
        self._sock.connect((host, port))

    def send(self, message: str) -> str:
        data = message.encode("utf-8")
        # Frame: 4-byte big-endian length + payload
        self._sock.sendall(struct.pack(">I", len(data)) + data)
        # Read 4-byte reply length
        reply_len_bytes = self._recv_exact(4)
        reply_len = struct.unpack(">I", reply_len_bytes)[0]
        if reply_len == 0:
            return ""
        return self._recv_exact(reply_len).decode("utf-8", errors="replace")

    def close(self):
        try:
            self._sock.close()
        except Exception:
            pass

    def _recv_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self._sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("connection closed by server")
            buf.extend(chunk)
        return bytes(buf)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def tcp_alive() -> bool:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect((HOST, TCP_PORT))
        s.close()
        return True
    except Exception:
        return False


def get_status(session: KnodSession) -> dict:
    try:
        resp = session.send("STATUS")
        result = {}
        for part in resp.strip().split():
            if ":" in part:
                k, v = part.split(":", 1)
                try:
                    result[k] = int(v)
                except ValueError:
                    result[k] = v  # keep string fields like graph: path
        return result
    except Exception:
        return {}


def wait_for_queue_drain(
    session: KnodSession, poll_interval: float = 5.0, stable_for: float = 10.0
):
    """Block until the ingest queue is empty, no goroutines are in-flight,
    and the GNN step count has stabilized."""
    print("  Waiting for queue to drain...", flush=True)
    last_step = -1
    stable_since = None

    while True:
        status = get_status(session)

        if not status:
            print("    (knod busy or unreachable, retrying...)", flush=True)
            time.sleep(poll_interval)
            stable_since = None
            continue

        queued = status.get("queued", 0)
        in_flight = status.get("in_flight", 0)
        step = status.get("strand_step", 0)
        thoughts = status.get("thoughts", 0)

        busy = queued > 0 or in_flight > 0

        if busy:
            print(
                f"    queued={queued} in_flight={in_flight} strand_step={step} thoughts={thoughts}",
                flush=True,
            )
            last_step = step
            stable_since = None
        elif step != last_step:
            print(
                f"    queued=0 in_flight=0 strand_step={step} thoughts={thoughts} (training...)",
                flush=True,
            )
            last_step = step
            stable_since = time.time()
        elif stable_since is not None and time.time() - stable_since >= stable_for:
            print(f"  Queue drained. thoughts={thoughts} strand_step={step}")
            return status

        time.sleep(poll_interval)


def load_manifest() -> list[tuple[str, str]]:
    entries = []
    if not os.path.exists(MANIFEST):
        for f in sorted(os.listdir(CORPUS_DIR)):
            if f.endswith(".txt") and f != "manifest.txt":
                entries.append((f[:-4], "on"))
        return entries
    with open(MANIFEST, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                entries.append((parts[0], parts[1]))
    return entries


def ingest_pass(
    session: KnodSession,
    entries: list[tuple[str, str]],
    delay: float,
) -> tuple[int, int]:
    ingested = 0
    errors = 0
    for name, category in entries:
        path = os.path.join(CORPUS_DIR, f"{name}.txt")
        if not os.path.exists(path):
            print(f"  skip  {name} (file missing)")
            continue
        with open(path, encoding="utf-8") as f:
            text = f.read()
        size = len(text)
        print(f"  [{category:3s}] {name} ({size:,} chars)...", end="", flush=True)
        try:
            resp = session.send(text)
            print(f"  {resp.strip() or 'sent'}")
            ingested += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            errors += 1
        if delay > 0:
            time.sleep(delay)
    return ingested, errors


def main():
    parser = argparse.ArgumentParser(description="Ingest corpus into running knod")
    parser.add_argument("--purpose", default=DEFAULT_PURPOSE)
    parser.add_argument("--only", choices=["on", "off"])
    parser.add_argument("--delay", type=float, default=0)
    parser.add_argument("--passes", type=int, default=1)
    args = parser.parse_args()

    if not tcp_alive():
        print(f"ERROR: knod not reachable at {HOST}:{TCP_PORT}")
        sys.exit(1)

    print(f"knod reachable at {HOST}:{TCP_PORT}")

    entries = load_manifest()
    if not entries:
        print("ERROR: no corpus files found. Run fetch_corpus.py first.")
        sys.exit(1)
    if args.only:
        entries = [(n, c) for n, c in entries if c == args.only]

    total_passes = args.passes
    total_ingested = 0
    total_errors = 0

    with KnodSession() as session:
        # Show where data is being ingested to.
        status = get_status(session)
        graph_path = status.get("graph", "(unknown)")
        print(f"  ingesting to: {graph_path}")

        print(f'\nSetting purpose: "{args.purpose}"')
        resp = session.send(f"PURPOSE:{args.purpose}")
        print(f"  -> {resp.strip()}")

        print(f"\nIngesting {len(entries)} articles x {total_passes} pass(es)...\n")

        for pass_num in range(1, total_passes + 1):
            if total_passes > 1:
                status = get_status(session)
                print(
                    f"{'=' * 50}\n"
                    f"  Pass {pass_num}/{total_passes}  "
                    f"[thoughts={status.get('thoughts', '?')} edges={status.get('edges', '?')} "
                    f"gnn_step={status.get('gnn_step', '?')} strand_step={status.get('strand_step', '?')} "
                    f"in_flight={status.get('in_flight', '?')}]\n"
                    f"{'=' * 50}\n"
                )

            ingested, errors = ingest_pass(session, entries, args.delay)
            total_ingested += ingested
            total_errors += errors

            if pass_num < total_passes:
                wait_for_queue_drain(session)

        if total_passes > 1:
            status = get_status(session)
            print(
                f"\nFinal graph state: thoughts={status.get('thoughts', '?')} "
                f"edges={status.get('edges', '?')} gnn_step={status.get('gnn_step', '?')} "
                f"strand_step={status.get('strand_step', '?')}"
            )

    print(
        f"\nDone: {total_ingested} queued, {total_errors} errors ({total_passes} pass(es))"
    )
    if total_errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
