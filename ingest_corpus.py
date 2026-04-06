"""
Feed cached corpus files into a running knod instance via TCP.

Expects:
  - knod running on TCP port 7999
  - corpus/ directory populated by fetch_corpus.py

Usage: python ingest_corpus.py [--purpose PURPOSE] [--only on|off] [--delay SECONDS]
  --purpose   set the node purpose before ingesting (default: turtle specialist)
  --only      only ingest "on"-topic or "off"-topic files
  --delay     seconds to wait between articles (default: 5)
"""

import argparse
import os
import socket
import sys
import time

HOST = "127.0.0.1"
TCP_PORT = 7999
CORPUS_DIR = "corpus"
MANIFEST = os.path.join(CORPUS_DIR, "manifest.txt")

DEFAULT_PURPOSE = (
    "specialist in turtles, their anatomy, biology, ecology, and conservation"
)


def tcp_send(text: str, expect_response: bool = False, timeout: float = 600.0) -> str:
    """Send text to knod over TCP.

    Always uses half-close (SHUT_WR) so the server can read all data before
    we tear down the connection.  Then waits for the server to close its end,
    which happens after handle_ingest / handle_ask completes.  This makes
    ingestion synchronous — we block until the server finishes processing.
    """
    data = text.encode("utf-8")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect((HOST, TCP_PORT))
    sock.sendall(data)

    # Signal EOF so the server's read_all_tcp returns.
    sock.shutdown(socket.SHUT_WR)

    # Wait for server to close the connection (it does so after processing).
    # Also collect any response bytes the server sends back.
    chunks = []
    while True:
        try:
            chunk = sock.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
        except socket.timeout:
            break
        except ConnectionError:
            break
    sock.close()
    return b"".join(chunks).decode("utf-8", errors="replace")


def tcp_alive() -> bool:
    """Check if knod TCP port is reachable."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect((HOST, TCP_PORT))
        s.close()
        return True
    except Exception:
        return False


def load_manifest() -> list[tuple[str, str]]:
    """Returns [(filename_stem, category), ...] in order."""
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


def main():
    parser = argparse.ArgumentParser(description="Ingest corpus into running knod")
    parser.add_argument("--purpose", default=DEFAULT_PURPOSE, help="set node purpose")
    parser.add_argument(
        "--only", choices=["on", "off"], help="only ingest on/off-topic"
    )
    parser.add_argument(
        "--delay", type=float, default=5.0, help="seconds between articles"
    )
    args = parser.parse_args()

    # Check knod is running.
    if not tcp_alive():
        print(f"ERROR: knod not reachable at {HOST}:{TCP_PORT}")
        print("Start knod first: just run  (or: just ingest)")
        sys.exit(1)

    print(f"knod reachable at {HOST}:{TCP_PORT}")

    # Set purpose.
    print(f'\nSetting purpose: "{args.purpose}"')
    resp = tcp_send(f"PURPOSE:{args.purpose}", expect_response=True, timeout=5)
    print(f"  -> {resp.strip()}")

    # Load manifest.
    entries = load_manifest()
    if not entries:
        print(f"\nERROR: no corpus files found. Run fetch_corpus.py first.")
        sys.exit(1)

    # Filter by category.
    if args.only:
        entries = [(name, cat) for name, cat in entries if cat == args.only]

    print(f"\nIngesting {len(entries)} articles (delay={args.delay}s)...\n")

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
            tcp_send(text)
            print(f"  sent")
            ingested += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            errors += 1

        # Wait for processing (LLM calls, embedding, GNN training).
        if args.delay > 0:
            time.sleep(args.delay)

    print(f"\nDone: {ingested} ingested, {errors} errors")
    if errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
