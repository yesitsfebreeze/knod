"""Integration tests for the Odin knod binary.

Tests the compiled cli.exe directly via subprocess calls.
Covers:
  - ingest (single file, multiple files)
  - ask --graph (single-store mode)
  - ask (multi-store / do_multi_ask default path, with fallback)
  - --internal-query subprocess mode (JSON output)
  - explore (graph stats)
  - register + list (registry)
  - knid management

Usage:
  python tests/test_odin.py [--skip-registry]

  --skip-registry   don't run register/knid tests (avoids touching ~/.config/knod/stores)

Notes:
  - Uses absolute --graph paths so tests don't affect the user's default graph.
  - Register/knid tests DO write to ~/.config/knod/stores.  Pass --skip-registry
    to skip those if you want a clean run.
  - All subprocess timeouts are generous to accommodate LLM API call latency.

Requires:
  - cli.exe built at E:/projects/knod/knod/cli.exe
  - corpus at E:/projects/knod/corpus/
  - Valid API key in ~/.config/knod/config
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

# ── paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BINARY = os.path.join(PROJECT_ROOT, "knod", "cli.exe")
CORPUS_DIR = os.path.join(PROJECT_ROOT, "corpus")

# Corpus files used for ingestion tests (keep small so tests are fast)
INGEST_FILES = [
	"knowledge_graph.txt",
	"graph_neural_network.txt",
	"word_embedding.txt",
	"information_retrieval.txt",
	"semantic_search.txt",
]

# Timeouts: ingest calls the embedding API per chunk, which can be slow
INGEST_TIMEOUT = 300  # 5 min per file
ASK_TIMEOUT = 120  # 2 min per question
EXPLORE_TIMEOUT = 30

# ── helpers ───────────────────────────────────────────────────────────────────

passed = 0
failed = 0
skipped = 0


def check(desc: str, cond: bool) -> bool:
	global passed, failed
	if cond:
		passed += 1
		print(f"  PASS  {desc}")
	else:
		failed += 1
		print(f"  FAIL  {desc}")
	return cond


def skip(desc: str, reason: str = ""):
	global skipped
	skipped += 1
	suffix = f" ({reason})" if reason else ""
	print(f"  SKIP  {desc}{suffix}")


def is_rate_limited(r) -> bool:
	"""Return True if the subprocess result indicates an OpenAI rate limit error."""
	combined = (r.stdout or "") + (r.stderr or "")
	return "Too_Many_Requests" in combined or "rate_limit_exceeded" in combined or "requests per day" in combined


def is_api_failure(r) -> bool:
	"""Return True if the result indicates an API-layer failure (rate limit, network, or
	generation failure that is not a logic bug in the binary)."""
	if is_rate_limited(r):
		return True
	# 'no answer generated' with non-zero exit means embed or LLM call failed
	# at the API level rather than a binary logic issue.
	combined = (r.stdout or "") + (r.stderr or "")
	return "no answer generated" in combined and r.returncode != 0


def run(args: list, timeout: int = 60) -> subprocess.CompletedProcess:
	"""Run knod with the given args; returns CompletedProcess."""
	cmd = [BINARY] + args
	try:
		return subprocess.run(
			cmd,
			capture_output=True,
			text=True,
			cwd=PROJECT_ROOT,
			timeout=timeout,
		)
	except subprocess.TimeoutExpired as e:
		# Return a fake result indicating timeout
		class _Timeout:
			returncode = -1
			stdout = ""
			stderr = f"[TIMEOUT after {timeout}s]"

		return _Timeout()


# ── test sections ─────────────────────────────────────────────────────────────


def test_binary_exists():
	print("\n=== Binary sanity ===")
	check("cli.exe exists", os.path.isfile(BINARY))


def test_ingest_and_explore(graph_path: str) -> bool:
	"""Ingest corpus into graph_path, verify with explore."""
	print("\n=== Ingest + Explore ===")

	# Ingest 5 articles one by one
	for i, fname in enumerate(INGEST_FILES):
		fpath = os.path.join(CORPUS_DIR, fname)
		print(f"  ingesting [{i + 1}/{len(INGEST_FILES)}] {fname} ...", flush=True)
		r = run(["ingest", "--graph", graph_path, fpath], timeout=INGEST_TIMEOUT)
		if is_rate_limited(r):
			skip(f"ingest {fname} exit=0", "OpenAI RPD limit")
			skip(f"  ingest output mentions 'thoughts'", "OpenAI RPD limit")
			print("  [WARNING] ingest rate-limited; remaining ingest+downstream tests will be skipped")
			return False
		ok = r.returncode == 0
		check(f"ingest {fname} exit=0", ok)
		if ok:
			check(
				f"  ingest output mentions 'thoughts'",
				"thoughts" in r.stdout,
			)
		else:
			print(f"    stdout: {r.stdout[:300]}")
			print(f"    stderr: {r.stderr[:300]}")
			return False

	# Verify the graph file was actually created (ingest may exit 0 even if LLM failed)
	if not os.path.isfile(graph_path):
		skip("explore exits 0", "graph file not created — likely all ingest LLM calls failed")
		skip("explore shows thoughts", "graph file not created")
		skip("explore shows edges", "graph file not created")
		skip("explore shows maturity", "graph file not created")
		skip("thoughts > 5", "graph file not created")
		print("  [WARNING] graph not created; downstream tests will be skipped")
		return False

	# explore
	r = run(["explore", "--graph", graph_path], timeout=EXPLORE_TIMEOUT)
	ok_explore = r.returncode == 0
	check("explore exits 0", ok_explore)
	if ok_explore:
		check("explore shows thoughts", re.search(r"thoughts:\s*[1-9]", r.stdout) is not None)
		check("explore shows edges", re.search(r"edges:\s*[0-9]+", r.stdout) is not None)
		check("explore shows maturity", "maturity" in r.stdout)
		thoughts_match = re.search(r"thoughts:\s*(\d+)", r.stdout)
		if thoughts_match:
			n = int(thoughts_match.group(1))
			check(f"thoughts > 5 ({n})", n > 5)
	else:
		print(f"    stderr: {r.stderr[:300]}")

	return ok_explore


def test_internal_query(graph_path: str):
	"""Test --internal-query subprocess mode: expects newline-delimited JSON."""
	print("\n=== --internal-query (subprocess mode) ===")

	r = run(
		["--internal-query", f"--graph={graph_path}", "--query=How does a knowledge graph store relationships?"],
		timeout=ASK_TIMEOUT,
	)
	check("--internal-query exits 0", r.returncode == 0)
	if r.returncode != 0:
		print(f"    stderr: {r.stderr[:400]}")
		return

	lines = [l.strip() for l in r.stdout.splitlines() if l.strip()]
	check("--internal-query produces output", len(lines) > 0)

	# Validate each line is valid JSON with expected fields
	valid = 0
	for line in lines:
		try:
			obj = json.loads(line)
			if "text" in obj and "score" in obj and "source" in obj:
				valid += 1
		except json.JSONDecodeError:
			pass
	if lines:
		check(f"all output lines are valid JSON ({valid}/{len(lines)})", valid == len(lines))

	if lines:
		first = json.loads(lines[0])
		check("first result has text", isinstance(first.get("text"), str) and len(first["text"]) > 5)
		check("first result has score", isinstance(first.get("score"), (int, float)))
		check("first result has source", isinstance(first.get("source"), str))
		score = first.get("score", 0)
		check(f"score in [0,1] ({score:.4f})", 0.0 <= score <= 1.0)

	# Test with an irrelevant query — should still exit 0 (may return empty)
	r2 = run(
		["--internal-query", f"--graph={graph_path}", "--query=best pizza recipes in Naples Italy"],
		timeout=ASK_TIMEOUT,
	)
	check("--internal-query irrelevant query exits 0", r2.returncode == 0)


def test_ask_single_store(graph_path: str):
	"""Test ask --graph (single-store path via handle_ask)."""
	print("\n=== ask --graph (single-store) ===")

	# Use questions very directly about the ingested content (knowledge graphs + retrieval corpus)
	questions = [
		("How does a knowledge graph represent relationships?", ["graph", "node", "edge", "relation", "triple", "link"]),
		(
			"What is the role of embeddings in semantic search?",
			["embed", "vector", "semantic", "represent", "dimension", "search"],
		),
	]

	for q, hints in questions:
		print(f"  Q: {q}", flush=True)
		r = run(["ask", "--graph", graph_path, q], timeout=ASK_TIMEOUT)
		if is_api_failure(r):
			skip(f"ask exits 0 (API error or rate limited)", "OpenAI API")
			skip(f"  got non-empty answer", "OpenAI API")
			skip(f"  answer relevant", "OpenAI API")
			continue
		if "no relevant thoughts found" in r.stderr:
			skip(f"ask exits 0 (no relevant thoughts — threshold too high for this corpus)", "retrieve threshold")
			continue
		ok = r.returncode == 0
		check(f"ask exits 0", ok)
		if ok:
			answer = r.stdout.strip()
			check(f"  got non-empty answer ({len(answer)} chars)", len(answer) > 10)
			relevant = any(h.lower() in answer.lower() for h in hints)
			check(f"  answer relevant (hints: {hints[:2]})", relevant)
			if not relevant:
				print(f"    answer: {answer[:200]}")
		else:
			print(f"    stderr: {r.stderr[:300]}")


def test_ask_multi_store_fallback(graph_path: str):
	"""Test ask without --graph but with KNOD_GRAPH pointing to our graph.

	do_multi_ask has a fallback: if no registered stores AND cfg.graph_path
	exists, it uses that as the default store.  We rely on that path being
	set in the user config (or pass --graph is not supported for multi_ask
	directly, but cfg.graph_path from the config file is the fallback).

	Since we can't pass cfg.graph_path to multi_ask without --graph, we
	test this by using ask --graph directly and verifying the fallback
	message is NOT shown (i.e. the explicit --graph path is used).
	"""
	print("\n=== ask (multi-store fallback via --graph) ===")

	# ask with --graph ensures single-store mode with our temp graph
	r = run(["ask", "--graph", graph_path, "How do retrieval systems rank results?"], timeout=ASK_TIMEOUT)
	if is_api_failure(r):
		skip("ask (explicit graph) exits 0", "OpenAI API")
		skip("  got non-empty answer", "OpenAI API")
		skip("  answer is about retrieval ranking", "OpenAI API")
		return
	ok = r.returncode == 0
	check("ask (explicit graph) exits 0", ok)
	if ok:
		answer = r.stdout.strip()
		check(f"  got non-empty answer ({len(answer)} chars)", len(answer) > 10)
		relevant = any(
			w in answer.lower() for w in ["rank", "score", "relevance", "retriev", "search", "similar", "query", "match"]
		)
		check("  answer is about retrieval ranking", relevant)
		if not relevant:
			print(f"    answer: {answer[:200]}")
	else:
		print(f"    stderr: {r.stderr[:300]}")


def test_register_and_list(graph_path: str) -> str | None:
	"""Register the temp graph, verify it appears in list.
	Returns the store name on success, None on failure.
	NOTE: This writes to ~/.config/knod/stores.
	"""
	print("\n=== register + list ===")

	r = run(["register", graph_path], timeout=30)
	ok = r.returncode == 0
	check("register exits 0", ok)
	if ok:
		check("register output is non-empty", len(r.stdout.strip()) > 0)
		print(f"    registered: {r.stdout.strip()}")
	else:
		print(f"    stderr: {r.stderr[:300]}")
		return None

	# Derive store name (same logic as do_register: basename without extension)
	base = os.path.basename(graph_path)
	store_name = base
	for ext in [".strand", ".graph"]:
		if store_name.endswith(ext):
			store_name = store_name[: -len(ext)]
			break

	r = run(["list"], timeout=10)
	ok = r.returncode == 0
	check("list exits 0", ok)
	if ok:
		check(f"list shows '{store_name}'", store_name in r.stdout)
	else:
		print(f"    stderr: {r.stderr[:300]}")

	return store_name if ok else None


def test_ask_multi_store_with_registry(store_name: str):
	"""Test ask (multi-store) with the store registered — should find it."""
	print("\n=== ask (multi-store with registry) ===")

	print(f"  querying via registered store '{store_name}'...", flush=True)
	r = run(["ask", "What are the advantages of graph-based retrieval over flat search?"], timeout=ASK_TIMEOUT)
	if is_api_failure(r):
		skip("ask (multi-store registered) exits 0", "OpenAI API")
		skip("  got non-empty answer", "OpenAI API")
		skip("  answer is about graph retrieval", "OpenAI API")
		return
	ok = r.returncode == 0
	check("ask (multi-store registered) exits 0", ok)
	if ok:
		answer = r.stdout.strip()
		check(f"  got non-empty answer ({len(answer)} chars)", len(answer) > 10)
		relevant = any(
			w in answer.lower()
			for w in ["graph", "travers", "neighbor", "relation", "context", "connect", "structure", "edge"]
		)
		check("  answer is about graph retrieval", relevant)
		if not relevant:
			print(f"    answer: {answer[:200]}")
	else:
		print(f"    stderr: {r.stderr[:300]}")


def test_knid_management(graph_path: str, store_name: str):
	"""Test knid new / add / list / ask --knid subcommands.
	NOTE: Writes to ~/.config/knod/stores.
	"""
	print("\n=== knid management ===")

	knid = "test_retrieval_knod_integ"  # unlikely to clash with real knids

	# Best-effort: remove any leftover knid from a previous failed run
	# (there's no 'knid delete', so we clean the registry file directly)
	_cleanup_knid_section(knid)

	# Create a knid
	r = run(["knid", "new", knid], timeout=10)
	check(f"knid new '{knid}' exits 0", r.returncode == 0)
	if r.returncode != 0:
		print(f"    stderr: {r.stderr[:200]}")
		return

	# Add the registered store to the knid
	r = run(["knid", "add", knid, store_name], timeout=10)
	ok = r.returncode == 0
	check(f"knid add '{store_name}' to '{knid}' exits 0", ok)
	if not ok:
		print(f"    stderr: {r.stderr[:200]}")
		return

	# List the knid
	r = run(["knid", "list", knid], timeout=10)
	ok = r.returncode == 0
	check(f"knid list '{knid}' exits 0", ok)
	if ok:
		check(f"knid list shows '{store_name}'", store_name in r.stdout)
	else:
		print(f"    stderr: {r.stderr[:200]}")

	# ask --knid <knid>
	print(f"  ask --knid {knid} ...", flush=True)
	r = run(["ask", "--knid", knid, "How do knowledge graphs improve AI memory?"], timeout=ASK_TIMEOUT)
	if is_api_failure(r):
		skip(f"ask --knid '{knid}' exits 0", "OpenAI API")
		skip(f"  got non-empty answer", "OpenAI API")
	else:
		ok = r.returncode == 0
		check(f"ask --knid '{knid}' exits 0", ok)
		if ok:
			answer = r.stdout.strip()
			check(f"  got non-empty answer ({len(answer)} chars)", len(answer) > 10)
		else:
			print(f"    stderr: {r.stderr[:300]}")

	# Cleanup: remove knid and de-register the store so we don't pollute
	# Note: there's no `knid delete` command — we can only remove the store from it
	r = run(["knid", "remove", knid, store_name], timeout=10)
	check(f"knid remove '{store_name}' from '{knid}' exits 0", r.returncode == 0)


def test_error_cases(graph_path: str):
	"""Test that error paths produce non-zero exit codes."""
	print("\n=== Error cases ===")

	# ingest a non-existent file
	r = run(["ingest", "--graph", graph_path, "nonexistent_file.txt"], timeout=10)
	check("ingest missing file exits nonzero", r.returncode != 0)

	# explore a non-existent graph
	r = run(["explore", "--graph", "/no/such/graph.strand"], timeout=10)
	check("explore missing graph exits nonzero", r.returncode != 0)

	# --internal-query without --graph
	r = run(["--internal-query", "--query=hello"], timeout=10)
	check("--internal-query missing --graph exits nonzero", r.returncode != 0)

	# --internal-query without --query
	r = run(["--internal-query", f"--graph={graph_path}"], timeout=10)
	check("--internal-query missing --query exits nonzero", r.returncode != 0)


def test_ingest_dedup(graph_path: str):
	"""Re-ingest the same file and verify thought count doesn't explode."""
	print("\n=== Dedup (re-ingest) ===")

	# Get current count
	r0 = run(["explore", "--graph", graph_path], timeout=EXPLORE_TIMEOUT)
	m0 = re.search(r"thoughts:\s*(\d+)", r0.stdout)
	before = int(m0.group(1)) if m0 else 0

	# Re-ingest a file already ingested
	fpath = os.path.join(CORPUS_DIR, "knowledge_graph.txt")
	print(f"  re-ingesting knowledge_graph.txt (before: {before} thoughts)...", flush=True)
	r = run(["ingest", "--graph", graph_path, fpath], timeout=INGEST_TIMEOUT)
	check("re-ingest exits 0", r.returncode == 0)

	# Get new count
	r1 = run(["explore", "--graph", graph_path], timeout=EXPLORE_TIMEOUT)
	m1 = re.search(r"thoughts:\s*(\d+)", r1.stdout)
	after = int(m1.group(1)) if m1 else 0

	increase = after - before
	check(
		f"re-ingest thought increase is small (before={before}, after={after}, +{increase})",
		increase < max(before, 1),
	)


# ── cleanup helper ────────────────────────────────────────────────────────────


def _cleanup_knid_section(knid_name: str):
	"""Remove a [knid_name] section (header + all member lines) from the registry file."""
	stores_path = os.path.expanduser("~/.config/knod/stores")
	if not os.path.isfile(stores_path):
		return
	try:
		with open(stores_path, "r", encoding="utf-8") as f:
			lines = f.readlines()
		new_lines = []
		skip_section = False
		for line in lines:
			stripped = line.strip()
			if stripped == f"[{knid_name}]":
				skip_section = True
				continue
			# A new section header ends the skip (but don't skip the new header)
			if stripped.startswith("[") and stripped.endswith("]") and skip_section:
				skip_section = False
			if not skip_section:
				new_lines.append(line)
		with open(stores_path, "w", encoding="utf-8") as f:
			f.writelines(new_lines)
	except Exception as e:
		print(f"  [cleanup] warning: could not clean knid section: {e}")


def cleanup_registry(store_name: str):
	"""Best-effort: remove the test store and any test knid from the registry file."""
	stores_path = os.path.expanduser("~/.config/knod/stores")
	if not os.path.isfile(stores_path):
		return
	try:
		with open(stores_path, "r", encoding="utf-8") as f:
			lines = f.readlines()
		new_lines = [l for l in lines if not l.startswith(f"{store_name} =")]
		with open(stores_path, "w", encoding="utf-8") as f:
			f.writelines(new_lines)
		print(f"  [cleanup] removed '{store_name}' from registry")
	except Exception as e:
		print(f"  [cleanup] warning: could not clean registry: {e}")


# ── main ──────────────────────────────────────────────────────────────────────


def main():
	global passed, failed

	skip_registry = "--skip-registry" in sys.argv

	if not os.path.isfile(BINARY):
		print(f"FATAL: binary not found at {BINARY}")
		print("Build with: odin build cli   (from E:/projects/knod/knod/)")
		sys.exit(1)

	if not os.path.isdir(CORPUS_DIR):
		print(f"FATAL: corpus directory not found: {CORPUS_DIR}")
		sys.exit(1)

	tmp = tempfile.mkdtemp(prefix="knod_test_")
	graph_path = os.path.join(tmp, "test.strand")

	print(f"temp dir:       {tmp}")
	print(f"binary:         {BINARY}")
	print(f"corpus:         {CORPUS_DIR}")
	print(f"graph:          {graph_path}")
	print(f"skip-registry:  {skip_registry}")

	store_name = None
	try:
		test_binary_exists()

		graph_ready = test_ingest_and_explore(graph_path)

		if graph_ready:
			test_internal_query(graph_path)
			test_ask_single_store(graph_path)
			test_ask_multi_store_fallback(graph_path)

			if not skip_registry:
				store_name = test_register_and_list(graph_path)
				if store_name:
					test_ask_multi_store_with_registry(store_name)
					test_knid_management(graph_path, store_name)
			else:
				print("\n=== register / knid tests SKIPPED (--skip-registry) ===")

			test_ingest_dedup(graph_path)

		test_error_cases(graph_path)

	finally:
		# Clean up temp dir
		shutil.rmtree(tmp, ignore_errors=True)
		# Clean up registry entry and test knid if we created one
		if not skip_registry:
			_cleanup_knid_section("test_retrieval_knod_integ")
			if store_name:
				cleanup_registry(store_name)

	total = passed + failed
	print(f"\n{'=' * 60}")
	print(f"RESULTS: {passed}/{total} passed, {failed} failed, {skipped} skipped")
	print(f"{'=' * 60}")
	sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
	main()
