"""Python-native integration tests for py_knod.

Tests the full pipeline without requiring bin/knod.exe.
Covers: graph operations, MCMC gate, limbo, registry knids,
graph limits, edge decay, binary log, status fields, shared base model.
"""

import os
import tempfile
import time
from pathlib import Path

import numpy as np

from py_knod.config import Config
from py_knod.specialist.graph import Graph, Thought, Edge, LimboThought
from py_knod.specialist.store import (
	save_graph,
	load_graph,
	save_model,
	load_model,
	save_base_model,
	load_base_model,
	save_strand,
	load_strand,
	GraphLog,
)
from py_knod.specialist.gnn import KnodMPNN, StrandLayer
from py_knod.registry import Registry

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


def rand_emb(dim=1536):
	v = np.random.randn(dim).astype(np.float32)
	return v / (np.linalg.norm(v) + 1e-10)


# ============================================================
# 1. Graph limits
# ============================================================
def test_graph_limits():
	print("\n=== Graph Limits ===")
	g = Graph(purpose="test", max_thoughts=5, max_edges=3)
	for i in range(10):
		g.add_thought(f"thought {i}", rand_emb())
	check(f"max_thoughts enforced ({g.num_thoughts} <= 5)", g.num_thoughts <= 5)

	# Add edges
	ids = list(g.thoughts.keys())
	for i in range(6):
		if i + 1 < len(ids):
			g.add_edge(ids[i], ids[(i + 1) % len(ids)], 0.5, "test", rand_emb())
	check(f"max_edges enforced ({g.num_edges} <= 3)", g.num_edges <= 3)


# ============================================================
# 2. Edge decay
# ============================================================
def test_edge_decay():
	print("\n=== Edge Decay ===")
	g = Graph(purpose="test")
	t1 = g.add_thought("a", rand_emb())
	t2 = g.add_thought("b", rand_emb())

	# Create edge with old timestamp
	e = g.add_edge(t1.id, t2.id, 0.5, "link", rand_emb())
	e.created_at = time.time() - 3600 * 24  # 24 hours ago

	initial_weight = e.weight
	g.apply_edge_decay(0.01)  # 1% per hour
	check(f"edge weight decayed ({e.weight:.4f} < {initial_weight})", e.weight < initial_weight)

	# Very old edge should be removed
	e2 = g.add_edge(t1.id, t2.id, 0.02, "weak", rand_emb())
	e2.created_at = time.time() - 3600 * 1000  # very old
	before = g.num_edges
	g.apply_edge_decay(0.01)
	check(f"dead edge removed ({g.num_edges} < {before})", g.num_edges < before)


# ============================================================
# 3. MCMC gate
# ============================================================
def test_mcmc_gate():
	print("\n=== MCMC Gate ===")
	from py_knod.ingest.commit import _accept

	# Test _accept at different maturity levels
	accepted_at_zero = sum(_accept(0.0) for _ in range(100))
	check(f"maturity=0 always accepts ({accepted_at_zero}/100)", accepted_at_zero == 100)

	accepted_at_one = sum(_accept(1.0) for _ in range(1000))
	# At maturity 1.0, p = 0.05, so ~50/1000 expected
	check(f"maturity=1 mostly rejects ({accepted_at_one}/1000 < 200)", accepted_at_one < 200)


# ============================================================
# 4. LimboThought + Graph.limbo
# ============================================================
def test_limbo():
	print("\n=== Limbo ===")
	g = Graph(purpose="test")
	lt = LimboThought(text="rejected thought", embedding=rand_emb(), source="test")
	g.limbo.append(lt)
	check(f"limbo stores thoughts ({len(g.limbo)})", len(g.limbo) == 1)
	check("limbo thought has text", g.limbo[0].text == "rejected thought")
	check("limbo thought has created_at", g.limbo[0].created_at > 0)


# ============================================================
# 5. Graph persistence (pickle + limbo)
# ============================================================
def test_persistence():
	print("\n=== Persistence ===")
	with tempfile.TemporaryDirectory() as tmpdir:
		path = os.path.join(tmpdir, "test.graph")

		g = Graph(purpose="persist test", max_thoughts=100, max_edges=200)
		t1 = g.add_thought("hello", rand_emb(), "src1")
		t2 = g.add_thought("world", rand_emb(), "src2")
		g.add_edge(t1.id, t2.id, 0.8, "related", rand_emb())
		g.limbo.append(LimboThought(text="limbo item", embedding=rand_emb()))

		save_graph(g, path)
		g2 = load_graph(path)

		check(f"purpose preserved", g2.purpose == "persist test")
		check(f"thoughts preserved ({g2.num_thoughts})", g2.num_thoughts == 2)
		check(f"edges preserved ({g2.num_edges})", g2.num_edges == 1)
		check(f"limbo preserved ({len(g2.limbo)})", len(g2.limbo) == 1)
		check(f"max_thoughts preserved", g2.max_thoughts == 100)
		check(f"max_edges preserved", g2.max_edges == 200)
		check(f"edge created_at preserved", g2.edges[0].created_at > 0)


# ============================================================
# 6. Append-only binary log
# ============================================================
def test_binary_log():
	print("\n=== Binary Log ===")
	with tempfile.TemporaryDirectory() as tmpdir:
		path = os.path.join(tmpdir, "test.graph")

		# Save a base graph
		g = Graph(purpose="log test")
		t1 = g.add_thought("base thought", rand_emb())
		save_graph(g, path)

		# Append entries to the log
		log = GraphLog(path)
		check("log cleared after save", not log.exists)

		t2 = Thought(id=99, text="appended thought", embedding=rand_emb(), source="log")
		log.append_thought(t2)

		e = Edge(source_id=1, target_id=99, weight=0.7, reasoning="log edge", embedding=rand_emb())
		log.append_edge(e)

		lt = LimboThought(text="log limbo", embedding=rand_emb())
		log.append_limbo(lt)

		check("log file created", log.exists)

		# Reload — should have base + log entries
		g2 = load_graph(path)
		check(f"replayed thoughts ({g2.num_thoughts})", g2.num_thoughts == 2)
		check(f"replayed edges ({g2.num_edges})", g2.num_edges == 1)
		check(f"replayed limbo ({len(g2.limbo)})", len(g2.limbo) == 1)
		check("thought 99 exists", 99 in g2.thoughts)

		# Compact — save full state, log should be cleared
		save_graph(g2, path)
		check("log cleared after compact", not GraphLog(path).exists)


# ============================================================
# 7. Registry + knid sections
# ============================================================
def test_registry_knids():
	print("\n=== Registry Knids ===")
	with tempfile.TemporaryDirectory() as tmpdir:
		reg = Registry.__new__(Registry)
		reg._path = Path(os.path.join(tmpdir, "stores"))
		reg.stores = {}
		reg.knids = {}

		# Create actual .knod-compatible graph files so registry can read metadata
		cfg = Config()
		cfg.embedding_dim = 16
		cfg.hidden_dim = 8
		cfg.num_layers = 1
		from py_knod.specialist.store import save_all as _save_all

		for name, purpose in [
			("embeddings", "embedding systems"),
			("retrieval", "retrieval architecture"),
			("graph_nav", "graph navigation"),
		]:
			g = Graph(name=name, purpose=purpose)
			m = KnodMPNN(cfg)
			s = StrandLayer(cfg.hidden_dim)
			fpath = os.path.join(tmpdir, f"{name}.knod")
			_save_all(g, m, s, Path(fpath).with_suffix(""))
			# Directly populate stores (bypass file metadata read for unit test speed)
			reg.stores[name] = {"path": fpath, "purpose": purpose}
			# Also append to the file so persistence test works
			reg._append(fpath)

		# Create knids
		reg.add_to_knid("core", "embeddings")
		reg.add_to_knid("core", "retrieval")
		reg.add_to_knid("systems", "embeddings")
		reg.add_to_knid("systems", "graph_nav")

		check("knid 'core' has 2 members", len(reg.stores_in_knid("core")) == 2)
		check("knid 'systems' has 2 members", len(reg.stores_in_knid("systems")) == 2)
		check("list_knids returns 2", len(reg.list_knids()) == 2)

		# Remove
		reg.remove_from_knid("core", "embeddings")
		check("removed from knid", len(reg.stores_in_knid("core")) == 1)

		# Persistence — reload reads .knod metadata for names
		reg2 = Registry.__new__(Registry)
		reg2._path = Path(reg._path)
		reg2.stores = {}
		reg2.knids = {}
		reg2._load()
		check(f"stores persisted ({len(reg2.stores)})", len(reg2.stores) == 3)
		check(f"knids persisted ({len(reg2.knids)})", len(reg2.knids) == 2)
		check("core knid persisted", "retrieval" in reg2.stores_in_knid("core"))

		# Unregister removes from knids
		reg.unregister("graph_nav")
		check("unregister removes from knid", "graph_nav" not in reg.stores_in_knid("systems"))


# ============================================================
# 8. Shared base GNN checkpoint
# ============================================================
def test_shared_base_model():
	print("\n=== Shared Base Model ===")
	cfg = Config()
	cfg.embedding_dim = 16
	cfg.hidden_dim = 8
	cfg.num_layers = 1

	model = KnodMPNN(cfg)
	# Save base
	with tempfile.TemporaryDirectory() as tmpdir:
		import py_knod.specialist.store as store_mod

		orig_path = store_mod._BASE_GNN_PATH
		store_mod._BASE_GNN_PATH = Path(os.path.join(tmpdir, "base.gnn"))
		try:
			save_base_model(model)
			check("base model saved", os.path.exists(store_mod._BASE_GNN_PATH))

			# Load into a fresh model
			model2 = KnodMPNN(cfg)
			loaded = load_base_model(model2)
			check("base model loaded", loaded)

			# Verify weights match
			for p1, p2 in zip(model.parameters(), model2.parameters()):
				if not (p1.data == p2.data).all():
					check("weights match", False)
					break
			else:
				check("weights match", True)
		finally:
			store_mod._BASE_GNN_PATH = orig_path


# ============================================================
# 9. Config fields
# ============================================================
def test_config_fields():
	print("\n=== Config Fields ===")
	cfg = Config()
	check("max_thoughts default is 0", cfg.max_thoughts == 0)
	check("max_edges default is 0", cfg.max_edges == 0)
	check("decay_coefficient default is 0", cfg.decay_coefficient == 0.0)
	check("limbo_scan_interval default is 60", cfg.limbo_scan_interval == 60.0)
	check("limbo_cluster_min default is 3", cfg.limbo_cluster_min == 3)
	check("limbo_cluster_threshold default is 0.75", cfg.limbo_cluster_threshold == 0.75)
	check("specialist_match_threshold default is 0.8", cfg.specialist_match_threshold == 0.8)


# ============================================================
# 10. Handler status fields
# ============================================================
def test_status_fields():
	print("\n=== Status Fields ===")
	from py_knod.handler import Handler

	cfg = Config()
	cfg.api_key = "test"
	cfg.embedding_dim = 16
	cfg.hidden_dim = 8
	cfg.num_layers = 1

	with tempfile.TemporaryDirectory() as tmpdir:
		cfg.graph_path = os.path.join(tmpdir, "test.graph")
		handler = Handler(cfg)
		handler.graph = Graph()
		handler.model = KnodMPNN(cfg)
		handler.strand = StrandLayer(cfg.hidden_dim)
		handler._queue = __import__("queue").Queue(maxsize=128)
		handler._in_flight = __import__("threading").Event()

		status = handler.status()
		check("status has queued", "queued=" in status)
		check("status has in_flight", "in_flight=" in status)
		check("status has limbo", "limbo=" in status)
		check("status has thoughts", "thoughts=" in status)
		check("status has edges", "edges=" in status)


# ============================================================
# Run all tests
# ============================================================
if __name__ == "__main__":
	test_graph_limits()
	test_edge_decay()
	test_mcmc_gate()
	test_limbo()
	test_persistence()
	test_binary_log()
	test_registry_knids()
	test_shared_base_model()
	test_config_fields()
	test_status_fields()

	print(f"\n{'=' * 40}")
	print(f"  {passed} passed, {failed} failed")
	print(f"{'=' * 40}")

	if failed > 0:
		exit(1)
