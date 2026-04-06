"""
Full integration test: fetch corpus, start knod, ingest, test retrieval.

Orchestrates the three scripts:
  1. fetch_corpus.py  — download Wikipedia articles to corpus/
  2. ingest_corpus.py — feed corpus into running knod
  3. test_retrieval.py — ask questions, check persistence

Usage: python test.py [--keep]
  --keep  don't wipe graph data before running
"""

import os
import subprocess
import sys
import time

KNOD_EXE = os.path.join("bin", "knod.exe")


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


def main():
    keep = "--keep" in sys.argv

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
        stderr=open(os.path.join("bin", "test_stderr.log"), "w"),
        cwd=os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin"),
    )
    print(f"  PID: {proc.pid}")
    time.sleep(3)

    try:
        # Step 4: Ingest.
        rc = run([sys.executable, "ingest_corpus.py"], "INGEST CORPUS")
        if rc != 0:
            print("FATAL: ingest_corpus.py failed")
            sys.exit(1)

        # Step 5: Test retrieval (includes persistence check).
        rc = run([sys.executable, "test_retrieval.py"], "TEST RETRIEVAL")
        sys.exit(rc)

    finally:
        kill_knod()


if __name__ == "__main__":
    main()
