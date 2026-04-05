"""
Integration test for knod: set purpose, ingest mixed Wikipedia articles, ask questions.
The node specializes in turtles — it should prioritize turtle knowledge even when
fed unrelated content about surfing, volcanoes, and jungles.

Usage: python test.py
"""

import subprocess
import socket
import time
import sys
import os
import urllib.request
import re

KNOD_EXE = os.path.join("bin", "knod.exe")
HOST = "127.0.0.1"
PORT = 7999
STARTUP_TIMEOUT = 15

PURPOSE = "specialist in turtles, their anatomy, biology, blood vessels, and ecology"

WIKI_URLS = {
    "turtles": "https://en.wikipedia.org/wiki/Turtle",
    "surfing": "https://en.wikipedia.org/wiki/Surfing",
    "volcanoes": "https://en.wikipedia.org/wiki/Volcano",
    "jungle": "https://en.wikipedia.org/wiki/Jungle",
}

QUESTIONS = [
    "What is a turtle shell made of?",
    "How do turtles breathe underwater?",
    "What do sea turtles eat?",
    "Tell me about turtle anatomy and blood vessels",
]


def strip_html(html: str) -> str:
    """Extract readable text from HTML paragraph content."""
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL)
    paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", html, flags=re.DOTALL)
    text = "\n\n".join(paragraphs)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def fetch_wiki(topic: str, url: str) -> str:
    """Fetch and extract text from a Wikipedia article."""
    print(f"    Fetching {topic}: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "knod-test/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        text = strip_html(html)
        # Truncate to keep API costs reasonable.
        if len(text) > 6000:
            text = text[:6000]
        print(f"    Got {len(text)} chars")
        return text
    except Exception as e:
        print(f"    ERROR fetching {topic}: {e}")
        return ""


def send_tcp(text: str, expect_response: bool = False, timeout: float = 120.0) -> str:
    """Send text to knod over TCP. Optionally read response."""
    data = text.encode("utf-8")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect((HOST, PORT))
    sock.sendall(data)

    if not expect_response:
        sock.shutdown(socket.SHUT_RDWR)
        sock.close()
        return ""

    # Half-close write so server sees EOF, then read response.
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


def wait_for_port(timeout=STARTUP_TIMEOUT):
    """Wait until knod is listening."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            sock.connect((HOST, PORT))
            sock.close()
            return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    print("=" * 60)
    print("KNOD INTEGRATION TEST")
    print(f'Purpose: "{PURPOSE}"')
    print("=" * 60)

    if not os.path.exists(KNOD_EXE):
        print(f"ERROR: {KNOD_EXE} not found. Run 'just build' first.")
        sys.exit(1)

    # --- Kill any existing knod ---
    print("\n[1] Stopping any existing knod...")
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

    # --- Clean old graph/strand data ---
    for ext in ("knod.strand", "knod.graph", "knod.gnn"):
        old_file = os.path.join("bin", "data", ext)
        if os.path.exists(old_file):
            os.remove(old_file)
            print(f"    Removed old {ext}.")

    # --- Start knod ---
    print("\n[2] Starting knod...")
    proc = subprocess.Popen(
        [KNOD_EXE],
        stdout=sys.stdout,
        stderr=sys.stderr,
        cwd=os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin"),
    )
    print(f"    PID: {proc.pid}")

    print("    Waiting for TCP listener...")
    if not wait_for_port():
        print("    FAILED: knod did not start listening.")
        proc.kill()
        sys.exit(1)
    print("    Ready!")

    try:
        # --- Set purpose ---
        print(f'\n[3] Setting purpose: "{PURPOSE}"')
        resp = send_tcp(f"PURPOSE:{PURPOSE}", expect_response=True, timeout=5)
        print(f"    Response: {resp}")

        # --- Fetch Wikipedia articles ---
        print("\n[4] Fetching Wikipedia articles...")
        articles = {}
        for topic, url in WIKI_URLS.items():
            text = fetch_wiki(topic, url)
            if text:
                articles[topic] = text

        if not articles:
            print("    FAILED: could not fetch any articles.")
            return

        # --- Ingest articles ---
        print(f"\n[5] Ingesting {len(articles)} articles...")
        for topic, text in articles.items():
            print(f"\n    Ingesting '{topic}' ({len(text)} chars)...")
            send_tcp(text)
            # Wait for ingestion + OpenAI calls to complete.
            print("    Waiting for processing...")
            time.sleep(15)

            if proc.poll() is not None:
                print(f"    ERROR: knod exited with code {proc.returncode}")
                return

        # --- Ask questions ---
        print("\n" + "=" * 60)
        print("[6] ASKING QUESTIONS")
        print("=" * 60)

        for q in QUESTIONS:
            print(f"\n    Q: {q}")
            print("    (waiting for response...)")
            response = send_tcp(f"ASK:{q}", expect_response=True, timeout=120)
            if response:
                print(f"    A: {response[:500]}")
                if len(response) > 500:
                    print(f"       ... ({len(response)} total chars)")
            else:
                print("    A: [no response]")

        # --- Summary ---
        print("\n" + "=" * 60)
        print("TEST COMPLETE")
        print("=" * 60)
        print(f'\nPurpose: "{PURPOSE}"')
        print(f"Ingested: {', '.join(articles.keys())}")
        print(f"Asked {len(QUESTIONS)} questions")

    finally:
        print("\n[7] Stopping knod...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("    Done.")


if __name__ == "__main__":
    main()
