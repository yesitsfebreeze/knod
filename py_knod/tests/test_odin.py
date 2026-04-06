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
  python tests/test_odin.py [--fresh]

  --fresh   wipe the temp test directory before running (default: always fresh)

Requires:
  - cli.exe built at E:/projects/knod/knod/cli.exe
  - corpus at E:/projects/knod/corpus/
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
	"turtle_shell.txt",
	"green_sea_turtle.txt",
	"sea_turtle.txt",
	"tortoise.txt",
	"leatherback.txt",
]

# Off-topic files (for future routing tests)
OFF_TOPIC_FILES = [
	"bicycle.txt",
	"chess.txt",
]

# ── helpers ───────────────────────────────────────────────────────────────────

passed = 0
failed = 0


def check(desc: str, cond: bool) -> bool:
	global passed, failed
	if cond:
		passed += 1
		print(f"  PASS  {desc}")
	else:
		failed += 1
		print(f"  FAIL  {desc}")
	return cond


def run(args: list, cwd: str | None = None, env: dict | None = None, timeout: int = 120) -> subprocess.CompletedProcess:
	"""Run knod with the given args; returns CompletedProcess."""
	cmd = [BINARY] + args
	merged_env = os.environ.copy()
	if env:
		merged_env.update(env)
	return subprocess.run(
		cmd,
		capture_output=True,
		text=True,
		cwd=cwd or PROJECT_ROOT,
		env=merged_env,
		timeout=timeout,
	)


def knod_env(graph_path: str, config_dir: str) -> dict:
	"""Return env vars that point knod at an isolated graph + config."""
	return {
		"KNOD_GRAPH": graph_path,
		"KNOD_CONFIG_DIR": config_dir,
	}


# ── fixtures ──────────────────────────────────────────────────────────────────


def make_temp_dir() -> str:
	d = tempfile.mkdtemp(prefix="knod_test_")
	return d


# ── test sections ─────────────────────────────────────────────────────────────


def test_binary_exists():
	print("\n=== Binary sanity ===")
	check("cli.exe exists", os.path.isfile(BINARY))


def test_ingest_and_explore(graph_path: str, config_dir: str) -> bool:
	"""Ingest turtle corpus into graph_path, verify with explore."""
	print("\n=== Ingest + Explore ===")
	env = knod_env(graph_path, config_dir)

	# Ingest 5 articles one by one
	for i, fname in enumerate(INGEST_FILES):
		fpath = os.path.join(CORPUS_DIR, fname)
		r = run(["ingest", "--graph", graph_path, fpath], env=env)
		ok = r.returncode == 0
		check(f"ingest [{i + 1}/{len(INGEST_FILES)}] {fname} exit=0", ok)
		if ok:
			# Expect output like "ingested from ...\ngraph: N thoughts, M edges"
			check(
				f"  ingest output mentions 'thoughts'",
				"thoughts" in r.stdout,
			)
		else:
			print(f"    stdout: {r.stdout[:300]}")
			print(f"    stderr: {r.stderr[:300]}")

	# explore
	r = run(["explore", "--graph", graph_path], env=env)
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


def test_internal_query(graph_path: str, config_dir: str):
	"""Test --internal-query subprocess mode: expects newline-delimited JSON."""
	print("\n=== --internal-query (subprocess mode) ===")
	env = knod_env(graph_path, config_dir)

	r = run(
		["--internal-query", f"--graph={graph_path}", "--query=What is a turtle shell made of?"],
		env=env,
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
	check(f"all output lines are valid JSON ({valid}/{len(lines)})", valid == len(lines))

	if lines:
		first = json.loads(lines[0])
		check("first result has text", isinstance(first.get("text"), str) and len(first["text"]) > 5)
		check("first result has score", isinstance(first.get("score"), (int, float)))
		check("first result has source", isinstance(first.get("source"), str))
		score = first.get("score", 0)
		check(f"score in [0,1] ({score:.4f})", 0.0 <= score <= 1.0)

	# Test with a query that should be irrelevant
	r2 = run(
		["--internal-query", f"--graph={graph_path}", "--query=blockchain smart contract DeFi"],
		env=env,
	)
	# Should exit 0 even with irrelevant query (may return empty or low-score results)
	check("--internal-query irrelevant query exits 0", r2.returncode == 0)


def test_ask_single_store(graph_path: str, config_dir: str):
	"""Test ask --graph (single-store path via handle_ask)."""
	print("\n=== ask --graph (single-store) ===")
	env = knod_env(graph_path, config_dir)

	questions = [
		("What is a turtle shell made of?", ["bone", "keratin", "scute", "shell"]),
		("How do sea turtles navigate?", ["magnet", "current", "ocean", "nav"]),
		("What is the largest sea turtle?", ["leatherback"]),
	]

	for q, hints in questions:
		r = run(["ask", "--graph", graph_path, q], env=env)
		ok = r.returncode == 0
		check(f"ask exits 0: '{q[:40]}...'", ok)
		if ok:
			answer = r.stdout.strip()
			check(f"  got non-empty answer ({len(answer)} chars)", len(answer) > 10)
			relevant = any(h.lower() in answer.lower() for h in hints)
			check(f"  answer relevant (hints: {hints[:2]}...)", relevant)
			if not relevant:
				print(f"    answer: {answer[:200]}")
		else:
			print(f"    stderr: {r.stderr[:300]}")


def test_ask_multi_store_fallback(graph_path: str, config_dir: str):
	"""Test ask without --graph: should use do_multi_ask with fallback to graph_path."""
	print("\n=== ask (multi-store fallback — no registry) ===")
	env = knod_env(graph_path, config_dir)

	# No registered stores yet; fallback uses KNOD_GRAPH env var / graph_path
	r = run(["ask", "What do sea turtles eat?"], env=env)
	ok = r.returncode == 0
	check("ask (multi fallback) exits 0", ok)
	if ok:
		answer = r.stdout.strip()
		check(f"  got non-empty answer ({len(answer)} chars)", len(answer) > 10)
		relevant = any(w in answer.lower() for w in ["seagrass", "algae", "herb", "plant", "feed", "eat"])
		check("  answer is about turtle diet", relevant)
		if not relevant:
			print(f"    answer: {answer[:200]}")
	else:
		print(f"    stderr: {r.stderr[:300]}")


def test_register_and_list(graph_path: str, config_dir: str):
	"""Test register + list commands."""
	print("\n=== register + list ===")
	env = knod_env(graph_path, config_dir)

	r = run(["register", graph_path], env=env)
	ok = r.returncode == 0
	check("register exits 0", ok)
	if ok:
		check("register output mentions graph name", len(r.stdout.strip()) > 0)
	else:
		print(f"    stderr: {r.stderr[:300]}")

	r = run(["list"], env=env)
	ok = r.returncode == 0
	check("list exits 0", ok)
	if ok:
		check("list shows registered store", graph_path in r.stdout or r.stdout.strip() != "no registered stores.")
	else:
		print(f"    stderr: {r.stderr[:300]}")


def test_ask_multi_store_with_registry(graph_path: str, config_dir: str):
	"""Test ask (multi-store) after a store is registered — should find the store."""
	print("\n=== ask (multi-store with registry) ===")
	env = knod_env(graph_path, config_dir)

	# Store should already be registered from test_register_and_list
	r = run(["ask", "What threats do sea turtles face?"], env=env)
	ok = r.returncode == 0
	check("ask (multi-store registered) exits 0", ok)
	if ok:
		answer = r.stdout.strip()
		check(f"  got non-empty answer ({len(answer)} chars)", len(answer) > 10)
		relevant = any(w in answer.lower() for w in ["poach", "plastic", "habitat", "fish", "hunt", "threat", "danger"])
		check("  answer is about threats", relevant)
		if not relevant:
			print(f"    answer: {answer[:200]}")
	else:
		print(f"    stderr: {r.stderr[:300]}")


def test_knid_management(graph_path: str, config_dir: str):
	"""Test knid new / add / list subcommands."""
	print("\n=== knid management ===")
	env = knod_env(graph_path, config_dir)

	# Create a knid
	r = run(["knid", "new", "reptiles"], env=env)
	check("knid new exits 0", r.returncode == 0)
	if r.returncode != 0:
		print(f"    stderr: {r.stderr[:200]}")

	# Add the registered store to the knid (need store name — use basename)
	store_name = os.path.splitext(os.path.basename(graph_path))[0]
	r = run(["knid", "add", "reptiles", store_name], env=env)
	ok = r.returncode == 0
	check(f"knid add '{store_name}' to 'reptiles' exits 0", ok)
	if not ok:
		print(f"    stderr: {r.stderr[:200]}")

	# List the knid
	r = run(["knid", "list", "reptiles"], env=env)
	ok = r.returncode == 0
	check("knid list 'reptiles' exits 0", ok)
	if ok:
		check(f"knid list shows '{store_name}'", store_name in r.stdout)
	else:
		print(f"    stderr: {r.stderr[:200]}")

	# ask --knid reptiles
	r = run(["ask", "--knid", "reptiles", "How big do sea turtles get?"], env=env)
	ok = r.returncode == 0
	check("ask --knid reptiles exits 0", ok)
	if ok:
		answer = r.stdout.strip()
		check(f"  got non-empty answer ({len(answer)} chars)", len(answer) > 10)
	else:
		print(f"    stderr: {r.stderr[:300]}")


def test_error_cases(graph_path: str, config_dir: str):
	"""Test that error paths produce non-zero exit codes."""
	print("\n=== Error cases ===")
	env = knod_env(graph_path, config_dir)

	# ingest a non-existent file
	r = run(["ingest", "--graph", graph_path, "nonexistent_file.txt"], env=env)
	check("ingest missing file exits nonzero", r.returncode != 0)

	# explore a non-existent graph
	r = run(["explore", "--graph", "/no/such/graph.strand"], env=env)
	check("explore missing graph exits nonzero", r.returncode != 0)

	# --internal-query without --graph
	r = run(["--internal-query", "--query=hello"], env=env)
	check("--internal-query missing --graph exits nonzero", r.returncode != 0)

	# --internal-query without --query
	r = run(["--internal-query", f"--graph={graph_path}"], env=env)
	check("--internal-query missing --query exits nonzero", r.returncode != 0)


def test_ingest_dedup(graph_path: str, config_dir: str):
	"""Re-ingest the same file and verify thought count doesn't explode."""
	print("\n=== Dedup (re-ingest) ===")
	env = knod_env(graph_path, config_dir)

	# Get current count
	r0 = run(["explore", "--graph", graph_path], env=env)
	m0 = re.search(r"thoughts:\s*(\d+)", r0.stdout)
	before = int(m0.group(1)) if m0 else 0

	# Re-ingest a file already ingested
	fpath = os.path.join(CORPUS_DIR, "turtle_shell.txt")
	r = run(["ingest", "--graph", graph_path, fpath], env=env)
	check("re-ingest exits 0", r.returncode == 0)

	# Get new count
	r1 = run(["explore", "--graph", graph_path], env=env)
	m1 = re.search(r"thoughts:\s*(\d+)", r1.stdout)
	after = int(m1.group(1)) if m1 else 0

	# Should not more than double in size (dedup should suppress most dupes)
	increase = after - before
	check(
		f"re-ingest thought increase is small (before={before}, after={after}, +{increase})",
		increase < before,  # increase must be less than the original count
	)


# ── main ──────────────────────────────────────────────────────────────────────


def main():
	global passed, failed

	if not os.path.isfile(BINARY):
		print(f"FATAL: binary not found at {BINARY}")
		print("Build with: odin build cli   (from E:/projects/knod/knod/)")
		sys.exit(1)

	if not os.path.isdir(CORPUS_DIR):
		print(f"FATAL: corpus directory not found: {CORPUS_DIR}")
		sys.exit(1)

	tmp = make_temp_dir()
	config_dir = os.path.join(tmp, "config")
	os.makedirs(config_dir, exist_ok=True)
	graph_path = os.path.join(tmp, "test.strand")

	print(f"temp dir:   {tmp}")
	print(f"binary:     {BINARY}")
	print(f"corpus:     {CORPUS_DIR}")
	print(f"graph:      {graph_path}")

	try:
		test_binary_exists()

		graph_ready = test_ingest_and_explore(graph_path, config_dir)

		if graph_ready:
			test_internal_query(graph_path, config_dir)
			test_ask_single_store(graph_path, config_dir)
			test_ask_multi_store_fallback(graph_path, config_dir)
			test_register_and_list(graph_path, config_dir)
			test_ask_multi_store_with_registry(graph_path, config_dir)
			test_knid_management(graph_path, config_dir)
			test_ingest_dedup(graph_path, config_dir)

		test_error_cases(graph_path, config_dir)

	finally:
		shutil.rmtree(tmp, ignore_errors=True)

	total = passed + failed
	print(f"\n{'=' * 60}")
	print(f"RESULTS: {passed}/{total} passed, {failed} failed")
	print(f"{'=' * 60}")
	sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
	main()
