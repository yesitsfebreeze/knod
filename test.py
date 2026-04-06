"""
Full integration test: fetch corpus, start knod, ingest, test retrieval.

Orchestrates the three scripts:
  1. fetch_corpus.py  — download Wikipedia articles to corpus/
  2. ingest_corpus.py — feed corpus into running knod (multiple passes)
  3. test_retrieval.py — ask questions, check persistence

Usage: python test.py [--keep] [--passes N]
  --keep    don't wipe graph data before running
  --passes  number of corpus passes before running retrieval (default: 3)
"""

import os
import socket
import struct
import subprocess
import sys
import time

KNOD_EXE = os.path.join("bin", "knod.exe")
BASE_URL = "http://127.0.0.1:8080"
TCP_HOST = "127.0.0.1"
TCP_PORT = 7999


def _tcp_framed_send(text: str, timeout: float = 10.0) -> str:
    """Open a fresh framed TCP session, send one message, return reply."""
    try:
        data = text.encode("utf-8")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((TCP_HOST, TCP_PORT))
        s.sendall(struct.pack(">I", len(data)) + data)
        # read 4-byte reply length
        hdr = bytearray()
        while len(hdr) < 4:
            chunk = s.recv(4 - len(hdr))
            if not chunk:
                break
            hdr.extend(chunk)
        if len(hdr) < 4:
            s.close()
            return ""
        reply_len = struct.unpack(">I", bytes(hdr))[0]
        reply = bytearray()
        while len(reply) < reply_len:
            chunk = s.recv(reply_len - len(reply))
            if not chunk:
                break
            reply.extend(chunk)
        s.close()
        return reply.decode("utf-8", errors="replace")
    except Exception:
        return ""


# A "good" learning state: GNN has trained for at least this many steps.
# With 3 passes of 17 articles at ~10 thoughts each ≈ 510 thoughts,
# the model will have run strand+base training after every article batch.
GOOD_GNN_STEPS = 20


def run(cmd: list[str], description: str) -> int:
    print(f"\n{'=' * 60}")
    print(f"  {description}")
    print(f"{'=' * 60}\n")
    result = subprocess.run(cmd)
    return result.returncode


def kill_knod():
    subprocess.run(
        [
            "pwsh",
            "-NoProfile",
            "-Command",
            "Get-Process knod -ErrorAction SilentlyContinue | Stop-Process -Force",
        ],
        capture_output=True,
    )
    time.sleep(1)


def tcp_send(text: str, timeout: float = 10.0) -> str:
    """Send a framed TCP message to knod and return the response."""
    return _tcp_framed_send(text, timeout)


def get_status() -> dict:
    """Query knod STATUS. Returns dict with thoughts, edges, gnn_step, strand_step."""
    resp = tcp_send("STATUS")
    result = {}
    for part in resp.strip().split():
        if ":" in part:
            k, v = part.split(":", 1)
            try:
                result[k] = int(v)
            except ValueError:
                pass
    return result


def wait_for_queue_drain(poll_interval: float = 5.0, stable_for: float = 15.0):
    """Wait until the ingest queue is empty, no goroutines are in-flight,
    and GNN step count has stabilized."""
    print("\n  Waiting for ingest queue to drain and GNN to stabilize...")
    last_step = -1
    stable_since = None

    while True:
        status = get_status()

        if not status:
            print("    (knod busy or unreachable, retrying...)")
            time.sleep(poll_interval)
            stable_since = None
            continue

        step = status.get("strand_step", 0)
        thoughts = status.get("thoughts", 0)
        queued = status.get("queued", 0)
        in_flight = status.get("in_flight", 0)

        busy = queued > 0 or in_flight > 0

        if busy:
            print(
                f"    queued={queued} in_flight={in_flight} strand_step={step} thoughts={thoughts} edges={status.get('edges', 0)}"
            )
            last_step = step
            stable_since = None
        elif step != last_step:
            print(
                f"    queued=0 in_flight=0 strand_step={step} thoughts={thoughts} edges={status.get('edges', 0)} (training...)"
            )
            last_step = step
            stable_since = time.time()
        elif stable_since is not None and time.time() - stable_since >= stable_for:
            print(f"  Queue drained. Final: thoughts={thoughts} strand_step={step}")
            return status

        time.sleep(poll_interval)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Full integration test for knod")
    parser.add_argument(
        "--keep", action="store_true", help="don't wipe graph data before running"
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=3,
        help="number of corpus passes for continuous learning (default: 3)",
    )
    args = parser.parse_args()

    keep = args.keep
    passes = args.passes

    if not os.path.exists(KNOD_EXE):
        print(f"ERROR: {KNOD_EXE} not found. Run 'just build' first.")
        sys.exit(1)

    # Step 1: Fetch corpus (idempotent — skips cached files).
    rc = run([sys.executable, "fetch_corpus.py"], "FETCH CORPUS")
    if rc != 0:
        print("FATAL: fetch_corpus.py failed")
        sys.exit(1)

    # Step 2: Clean slate.
    kill_knod()
    if not keep:
        for name in ("knod.strand", "knod.graph", "knod.gnn"):
            path = os.path.join("bin", "data", name)
            if os.path.exists(path):
                os.remove(path)
                print(f"  Removed {path}")

    # Step 3: Start knod.
    print("\nStarting knod...")
    proc = subprocess.Popen(
        [KNOD_EXE],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=open(os.path.join("bin", "test.err.log"), "w"),
        cwd=os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin"),
    )
    print(f"  PID: {proc.pid}")

    # Wait for TCP to be ready (up to 15s).
    for _ in range(30):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1.0)
            s.connect((TCP_HOST, TCP_PORT))
            s.close()
            break
        except Exception:
            time.sleep(0.5)
    else:
        print("ERROR: knod did not become ready in time")
        kill_knod()
        sys.exit(1)
    print("  knod ready")

    try:
        # Step 4: Continuously ingest until learned.
        rc = run(
            [sys.executable, "ingest_corpus.py", f"--passes={passes}", "--delay=0"],
            f"INGEST CORPUS ({passes} passes — continuous learning)",
        )
        if rc != 0:
            print("FATAL: ingest_corpus.py failed")
            sys.exit(1)

        # Step 5: Wait for the async queue to drain and GNN to converge.
        print(f"\n{'=' * 60}")
        print("  WAITING FOR LEARNING TO STABILIZE")
        print(f"{'=' * 60}")
        final_status = wait_for_queue_drain(poll_interval=5.0, stable_for=15.0)

        gnn_step = final_status.get("gnn_step", 0)
        strand_step = final_status.get("strand_step", 0)
        thoughts = final_status.get("thoughts", 0)

        print(f"\n  Learning state:")
        print(f"    thoughts   = {thoughts}")
        print(f"    gnn_step   = {gnn_step}")
        print(f"    strand_step= {strand_step}")

        if strand_step < GOOD_GNN_STEPS:
            print(
                f"\n  WARNING: strand_step={strand_step} < {GOOD_GNN_STEPS} "
                f"(try --passes={passes + 2} for more training)"
            )
        else:
            print(
                f"\n  Learning state: GOOD (strand_step={strand_step} >= {GOOD_GNN_STEPS})"
            )

        # Step 6: Test retrieval (includes persistence check).
        rc = run([sys.executable, "test_retrieval.py"], "TEST RETRIEVAL")
        sys.exit(rc)

    finally:
        kill_knod()


if __name__ == "__main__":
    main()
