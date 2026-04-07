"""Ask the store questions only it would know — specific facts from ingested corpus."""

import logging, os, time

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

from py_knod.config import Config
from py_knod.handler import Handler
from py_knod.specialist.graph import Graph
from py_knod.specialist import KnodMPNN, StrandLayer, GNNTrainer
from py_knod.ingest import Ingester

cfg = Config.load()
handler = Handler(cfg)
handler.graph = Graph()
handler.model = KnodMPNN(cfg)
handler.strand = StrandLayer(cfg.hidden_dim)
handler.trainer = GNNTrainer(handler.model, handler.strand, cfg)
handler.ingester = Ingester(handler.graph, handler.provider, cfg)

# Ingest 4 articles with rich specific facts
ARTICLES = ["knowledge_graph.txt", "graph_neural_network.txt", "information_retrieval.txt", "word_embedding.txt"]
corpus_dir = os.path.join(os.path.dirname(__file__), "corpus")

for fname in ARTICLES:
	text = open(os.path.join(corpus_dir, fname), encoding="utf-8").read()
	t0 = time.time()
	r = handler.ingest_sync(text, source=fname)
	print(f"  Ingested {fname}: {r['thoughts']}t / {r['edges']}e  ({time.time() - t0:.0f}s)")

g = handler.graph
print(f"\n  Store: {g.num_thoughts} thoughts, {g.num_edges} edges, maturity={g.maturity:.2f}\n")

# Questions that require specific knowledge from the corpus
QUESTIONS = [
	"How does a knowledge graph differ from a relational database?",
	"What is message passing in graph neural networks?",
	"How does TF-IDF weight terms in information retrieval?",
	"What is the relationship between Word2Vec and word embeddings?",
	"How do vector databases enable nearest-neighbor search?",
]

print("=" * 70)
for q in QUESTIONS:
	answer, sources = handler.ask(q)
	print(f"Q: {q}")
	print(f"A: {answer}\n")
	for s in sources[:3]:
		print(f"   [{s['similarity']:.3f}] {s['text'][:90]}...")
	print("-" * 70)
