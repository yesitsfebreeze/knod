"""Ask the store questions only it would know — specific facts from ingested corpus."""

import logging, os, time

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

from py_knod.config import Config
from py_knod.handler import Handler
from py_knod.storage.graph import Graph
from py_knod.storage import KnodMPNN, StrandLayer, GNNTrainer, Ingester

cfg = Config.load()
handler = Handler(cfg)
handler.graph = Graph()
handler.model = KnodMPNN(cfg)
handler.strand = StrandLayer(cfg.hidden_dim)
handler.trainer = GNNTrainer(handler.model, handler.strand, cfg)
handler.ingester = Ingester(handler.graph, handler.provider, cfg)

# Ingest 4 articles with rich specific facts
ARTICLES = ["hawksbill.txt", "leatherback.txt", "turtle_excluder.txt", "green_sea_turtle.txt"]
corpus_dir = os.path.join(os.path.dirname(__file__), "corpus")

for fname in ARTICLES:
	text = open(os.path.join(corpus_dir, fname), encoding="utf-8").read()
	t0 = time.time()
	r = handler.handle_ingest(text, source=fname)
	print(f"  Ingested {fname}: {r['thoughts']}t / {r['edges']}e  ({time.time() - t0:.0f}s)")

g = handler.graph
print(f"\n  Store: {g.num_thoughts} thoughts, {g.num_edges} edges, maturity={g.maturity:.2f}\n")

# Questions that require specific knowledge from the corpus
QUESTIONS = [
	"What is the heaviest hawksbill sea turtle ever recorded?",
	"How deep can leatherback turtles dive?",
	"What year did the United States require shrimp trawlers to use turtle excluder devices?",
	"What is the genus name Eretmochelys derived from?",
	"How do leatherback turtles survive in cold water?",
]

print("=" * 70)
for q in QUESTIONS:
	answer, sources = handler.handle_ask(q)
	print(f"Q: {q}")
	print(f"A: {answer}\n")
	for s in sources[:3]:
		print(f"   [{s['similarity']:.3f}] {s['text'][:90]}...")
	print("-" * 70)
