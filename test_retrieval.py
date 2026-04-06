"""
Test retrieval against a running knod that has already ingested corpus data.

Tests:
  1. TCP ask — all questions get non-empty answers
  2. Persistence — stop knod, restart, verify answers survive

Usage: python test_retrieval.py [--no-persist]
  --no-persist  skip the stop/restart persistence check
"""

import argparse
import os
import socket
import subprocess
import sys
import time

KNOD_EXE = os.path.join("bin", "knod.exe")
HOST = "127.0.0.1"
TCP_PORT = 7999
STARTUP_TIMEOUT = 15

QUESTIONS = [
    "What is a turtle shell made of?",
    "How do sea turtles navigate across oceans?",
    "What are the differences between turtles and tortoises?",
    "Tell me about leatherback sea turtle anatomy",
    "How do turtles breathe?",
    "What threats do sea turtles face from human activity?",
    "What do green sea turtles eat?",
    "How do hawksbill turtles differ from other species?",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

passed = 0
failed = 0


def check(description: str, condition: bool):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {description}")
    else:
        failed += 1
        print(f"  FAIL  {description}")


def tcp_send(text: str, expect_response: bool = False, timeout: float = 120.0) -> str:
    data = text.encode("utf-8")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect((HOST, TCP_PORT))
    sock.sendall(data)

    if not expect_response:
        sock.shutdown(socket.SHUT_RDWR)
        sock.close()
        return ""

    sock.shutdown(socket.SHUT_WR)
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
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect((HOST, TCP_PORT))
        s.close()
        return True
    except Exception:
        return False


def wait_for_tcp(timeout: int = STARTUP_TIMEOUT) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        if tcp_alive():
            return True
        time.sleep(0.5)
    return False


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


def start_knod() -> subprocess.Popen:
    return subprocess.Popen(
        [KNOD_EXE],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=open(os.path.join("bin", "test_stderr.log"), "w"),
        cwd=os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin"),
    )


# ---------------------------------------------------------------------------
# Test phases
# ---------------------------------------------------------------------------


def test_ask():
    """Ask all questions via TCP, verify non-empty responses."""
    print("\n=== Ask (TCP) ===")
    for q in QUESTIONS:
        print(f"\n  Q: {q}")
        resp = tcp_send(f"ASK:{q}", expect_response=True, timeout=120)
        has_answer = len(resp.strip()) > 0
        check(f"answer received ({len(resp)} chars)", has_answer)
        if has_answer:
            preview = resp[:300].replace("\n", " ")
            print(f"  A: {preview}{'...' if len(resp) > 300 else ''}")


def test_persistence():
    """Stop knod, restart, ask again."""
    print("\n=== Persistence ===")

    baseline_q = QUESTIONS[0]

    print(f"\n  Pre-shutdown ask: {baseline_q}")
    pre = tcp_send(f"ASK:{baseline_q}", expect_response=True, timeout=120)
    pre_len = len(pre.strip())
    check(f"pre-shutdown answer ({pre_len} chars)", pre_len > 0)

    print("\n  Stopping knod...")
    kill_knod()
    time.sleep(2)

    print("  Restarting knod...")
    proc = start_knod()
    time.sleep(3)

    if not wait_for_tcp():
        check("knod restarted", False)
        proc.kill()
        return

    check("knod restarted", True)

    print(f"\n  Post-restart ask: {baseline_q}")
    post = tcp_send(f"ASK:{baseline_q}", expect_response=True, timeout=120)
    post_len = len(post.strip())
    check(f"post-restart answer ({post_len} chars)", post_len > 0)
    check("data persisted across restart", pre_len > 0 and post_len > 0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Test knod retrieval")
    parser.add_argument(
        "--no-persist", action="store_true", help="skip persistence test"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("KNOD RETRIEVAL TEST")
    print("=" * 60)

    if not tcp_alive():
        print(f"\nERROR: knod not reachable at {HOST}:{TCP_PORT}")
        print("Start knod and ingest corpus first.")
        sys.exit(1)

    print(f"knod reachable at {HOST}:{TCP_PORT}")

    test_ask()

    if not args.no_persist:
        test_persistence()

    total = passed + failed
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} passed, {failed} failed")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
