"""End-to-end test: ingest multiple corpus articles, verify edges, ask questions, verify hybrid scoring + feedback.

Uses a persistent graph at bin/data/test_edges.graph so re-runs skip ingestion.
Pass --fresh to force re-ingestion.
"""

import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("test")

from py_knod.config import Config
from py_knod.specialist.graph import Graph
from py_knod.specialist.store import save_graph, load_graph, save_model, load_model
from py_knod.specialist import KnodMPNN, StrandLayer, GNNTrainer
from py_knod.ingest import Ingester
from py_knod.handler import Handler

passed = 0
failed = 0


def check(desc, cond):
	global passed, failed
	if cond:
		passed += 1
		print(f"  PASS  {desc}")
	else:
		failed += 1
		print(f"  FAIL  {desc}")


# === Paths ===
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
GRAPH_PATH = os.path.join(PROJECT_ROOT, "bin", "data", "test_edges.graph")
MODEL_PATH = os.path.join(PROJECT_ROOT, "bin", "data", "test_edges.pt")
CORPUS_DIR = os.path.join(PROJECT_ROOT, "corpus")

ARTICLES = [
	"turtle_shell.txt",
	"green_sea_turtle.txt",
	"sea_turtle.txt",
	"tortoise.txt",
	"leatherback.txt",
	"hawksbill.txt",
	"reptile.txt",
]

# Additional articles to ingest on top of base set
EXTRA_ARTICLES = [
	"komodo_dragon.txt",
	"king_cobra.txt",
	"crocodilian.txt",
	"chameleon.txt",
	"galapagos_tortoise.txt",
	# Re-ingest one to test dedup
	"turtle_shell.txt",
]

fresh = "--fresh" in sys.argv

# === Setup ===
cfg = Config.load()
handler = Handler(cfg)

if not fresh and os.path.exists(GRAPH_PATH):
	print("\n=== Loading Persisted Graph ===")
	handler.graph = load_graph(GRAPH_PATH)
	handler.model = KnodMPNN(cfg)
	handler.strand = StrandLayer(cfg.hidden_dim)
	if os.path.exists(MODEL_PATH):
		load_model(handler.model, handler.strand, MODEL_PATH)
		print("  Loaded model checkpoint")
	handler.trainer = GNNTrainer(handler.model, handler.strand, cfg)
	handler.ingester = Ingester(handler.graph, handler.provider, cfg)
	g = handler.graph
	print(f"  Loaded: {g.num_thoughts} thoughts, {g.num_edges} edges, maturity={g.maturity:.3f}")
else:
	print("\n=== Ingesting Corpus (first run — will persist) ===")
	handler.graph = Graph()
	handler.model = KnodMPNN(cfg)
	handler.strand = StrandLayer(cfg.hidden_dim)
	handler.trainer = GNNTrainer(handler.model, handler.strand, cfg)
	handler.ingester = Ingester(handler.graph, handler.provider, cfg)

	for i, fname in enumerate(ARTICLES):
		path = os.path.join(CORPUS_DIR, fname)
		text = open(path, encoding="utf-8").read()
		t0 = time.time()
		r = handler.handle_ingest(text, source=fname)
		dt = time.time() - t0
		print(f"  [{i + 1}/{len(ARTICLES)}] {fname}: {r['thoughts']} thoughts, {r['edges']} edges ({dt:.1f}s)")

	g = handler.graph
	print(f"\n  Total: {g.num_thoughts} thoughts, {g.num_edges} edges, maturity={g.maturity:.3f}")

	# Persist
	os.makedirs(os.path.dirname(GRAPH_PATH), exist_ok=True)
	save_graph(g, GRAPH_PATH)
	save_model(handler.model, handler.strand, MODEL_PATH)
	print(f"  Saved to {GRAPH_PATH}")

# === 1b. Ingest extra articles (tests dedup + expansion) ===
existing_sources = set(t.source for t in handler.graph.thoughts.values())
extras_to_ingest = [f for f in EXTRA_ARTICLES if f not in existing_sources or f == "turtle_shell.txt"]
if extras_to_ingest:
	new_extras = [f for f in extras_to_ingest if f not in existing_sources]
	dupe_extras = [f for f in extras_to_ingest if f in existing_sources]
	if new_extras or dupe_extras:
		print(f"\n=== Ingesting Extra Articles ({len(new_extras)} new, {len(dupe_extras)} re-ingest for dedup test) ===")
		before_thoughts = handler.graph.num_thoughts
		for i, fname in enumerate(extras_to_ingest):
			fpath = os.path.join(CORPUS_DIR, fname)
			text = open(fpath, encoding="utf-8").read()
			t0 = time.time()
			r = handler.handle_ingest(text, source=fname)
			dt = time.time() - t0
			label = "(RE-INGEST)" if fname in existing_sources else "(NEW)"
			print(f"  [{i + 1}/{len(extras_to_ingest)}] {fname} {label}: {r['thoughts']} thoughts, {r['edges']} edges ({dt:.1f}s)")
		after_thoughts = handler.graph.num_thoughts
		print(f"\n  Before: {before_thoughts} thoughts -> After: {after_thoughts} thoughts")
		# Save updated graph
		save_graph(handler.graph, GRAPH_PATH)
		save_model(handler.model, handler.strand, MODEL_PATH)
		print(f"  Saved to {GRAPH_PATH}")

g = handler.graph

# === 2. Verify graph structure ===
print("\n=== Graph Structure ===")
check(f"thoughts > 20 ({g.num_thoughts})", g.num_thoughts > 20)
check(f"edges > 5 ({g.num_edges})", g.num_edges > 5)
check(f"maturity > 0 ({g.maturity:.3f})", g.maturity > 0)
check(f"maturity <= 1.0", g.maturity <= 1.0)

# Count sources
sources = set(t.source for t in g.thoughts.values())
check(f"multiple sources ({len(sources)})", len(sources) >= 3)

# === 3. Ask questions ===
QUESTIONS = [
	("What is a turtle shell made of?", ["bone", "keratin", "scute"]),
	("How do sea turtles navigate?", ["magnet", "current", "ocean", "navigate"]),
	("What do green sea turtles eat?", ["seagrass", "algae", "herbivor"]),
	("What is the largest sea turtle species?", ["leatherback"]),
	("What threats do sea turtles face?", ["poach", "plastic", "habitat", "fish", "trade", "hunt"]),
	("How big can a Komodo dragon get?", ["meter", "feet", "length", "weigh", "large", "kg", "pound"]),
	("Is the king cobra venomous?", ["venom", "neurotox", "bite", "poison", "deadly"]),
	("How do chameleons change color?", ["color", "pigment", "chromatophore", "skin", "cell", "light"]),
	("Where do Galapagos tortoises live?", ["galapagos", "island", "ecuador", "archipelago"]),
]

print("\n=== Ask Questions (Hybrid Scoring) ===")
for q, hint_words in QUESTIONS:
	answer, srcs = handler.handle_ask(q)
	has_answer = len(answer) > 10
	# Check if answer touches on expected topic (any hint word appears)
	answer_lower = answer.lower()
	relevant = any(w in answer_lower for w in hint_words)
	print(f"\n  Q: {q}")
	print(f"  A: {answer}")
	print(f"  Sources: {', '.join(s.get('source', s.get('text', '')[:30]) if isinstance(s, dict) else s.source for s in srcs[:3])}")
	check(f"got answer ({len(answer)} chars)", has_answer)
	check(f"answer relevant (contains hint word)", relevant)
	if not relevant:
		print(f"    Expected one of: {hint_words}")
	check(f"sources returned ({len(srcs)})", len(srcs) > 0)

# === 4. Feedback loop ===
print("\n=== Feedback Loop ===")
accessed = [t for t in g.thoughts.values() if t.access_count > 0]
check(f"accessed thoughts tracked ({len(accessed)})", len(accessed) > 0)
if accessed:
	max_access = max(t.access_count for t in accessed)
	check(f"max access_count = {max_access}", max_access >= 1)
	check("last_accessed is recent", accessed[0].last_accessed > time.time() - 120)

# === 5. GNN was trained ===
print("\n=== GNN Training ===")
check("model exists", handler.model is not None)
check("strand exists", handler.strand is not None)

# === 6. Edge quality sample ===
print("\n=== Edge Samples ===")
import random

sample = random.sample(handler.graph.edges, min(5, len(handler.graph.edges)))
for e in sample:
	s = g.thoughts.get(e.source_id)
	t = g.thoughts.get(e.target_id)
	if s and t:
		print(f"  [{s.text[:45]}] -> [{t.text[:45]}] w={e.weight:.2f}")

total = passed + failed
print(f"\n{'=' * 60}")
print(f"RESULTS: {passed}/{total} passed, {failed} failed")
print(f"{'=' * 60}")
