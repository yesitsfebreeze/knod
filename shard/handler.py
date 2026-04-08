"""Handler — owns all shards; orchestrates ingest, limbo, and retrieval."""

import logging
import threading
from pathlib import Path
from queue import Queue, Full

import numpy as np

from .config import Config
from .provider import Provider
from .registry import Registry, store_path
from .shard.graph import Graph, Thought
from .shard.gnn import ShardMPNN, ShardLayer
from .shard.trainer import GNNTrainer
from .shard.types import GraphEvent, EventListener, ShardIndexEntry, Shard, IngestResult
from .shard.store import save_all, load_all, load_base_model, read_shard_metadata, save_base_model
from .util.math import cosine
from .ingest import Ingester
from .limbo import find_clusters, promote_cluster
from .retrieval import (
	cosine_scores,
	edge_scores,
	gnn_scores,
	merge,
	deduplicate,
	best_chains_from,
	rate_thoughts,
	expand,
	PathChain,
	answer,
	synthesize_direct,
)

log = logging.getLogger(__name__)

QUEUE_CAPACITY = 128


class Handler:
	def __init__(self, cfg: Config):
		self.cfg = cfg
		self.provider = Provider(cfg)
		self.registry = Registry()
		self.graph: Graph | None = None
		self.model: ShardMPNN | None = None
		self.shard: ShardLayer | None = None
		self.trainer: GNNTrainer | None = None
		self.ingester: Ingester | None = None
		self._shards: dict[str, Shard] = {}
		self._index: dict[str, ShardIndexEntry] = {}  # shard name → index entry
		self.mu = threading.Lock()
		self._queue: Queue | None = None
		self._queue_worker: threading.Thread | None = None
		self._limbo_thread: threading.Thread | None = None
		self._poll_thread: threading.Thread | None = None
		self._shutdown = threading.Event()
		self._in_flight = threading.Event()
		self._listeners: dict[str, list[EventListener]] = {}
		self._listeners_lock = threading.Lock()
		self._retrieval_count: int = 0
		self._last_poll_state: dict = {}
		self._last_poll_state_lock = threading.Lock()
		self._last_shard_state: dict[str, tuple[int, int]] = {}  # name -> (thoughts, edges)

	def init(self):
		base = Path(self.cfg.graph_path).with_suffix("")
		shard_file = base.with_suffix(".shard")

		if shard_file.exists():
			log.info("Loading existing graph from %s", shard_file)
			self.graph, self.model, self.shard = load_all(self.cfg, base)
		else:
			log.info("Creating new graph")
			self.graph = Graph(
				max_thoughts=self.cfg.max_thoughts,
				max_edges=self.cfg.max_edges,
				maturity_divisor=self.cfg.maturity_divisor,
			)
			self.model = ShardMPNN(self.cfg)
			self.shard = ShardLayer(self.cfg.hidden_dim)

		# Bootstrap base model on first run if missing
		if not load_base_model(self.model, self.cfg):
			save_base_model(self.model, self.cfg)
			log.info("Bootstrapped base model: %s", self.cfg.base_gnn_path or "~/.config/shard/base.gnn")

		self.trainer = GNNTrainer(self.model, self.shard, self.cfg)
		self.ingester = Ingester(self.graph, self.provider, self.cfg)

		# Load registered shards
		self._load_shards()

		# Start async ingest queue
		self._queue = Queue(maxsize=QUEUE_CAPACITY)
		self._queue_worker = threading.Thread(target=self._queue_loop, daemon=True)
		self._queue_worker.start()

		# Start limbo background scan
		self._limbo_thread = threading.Thread(target=self._limbo_scan_loop, daemon=True)
		self._limbo_thread.start()

		# Start store polling (every 5 seconds)
		self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
		self._poll_thread.start()

	def save(self):
		base = Path(self.cfg.graph_path).with_suffix("")
		save_all(self.graph, self.model, self.shard, base)

	def shutdown(self):
		self._shutdown.set()
		if self._queue is not None:
			self._queue.put(None)  # sentinel
		self.save()

	def enqueue(self, text: str, source: str = "", descriptor: str = "") -> tuple[bool, int]:
		"""Try to enqueue ingest. Returns (queued, pending_count)."""
		if self._queue is None:
			return False, 0
		try:
			self._queue.put_nowait((text, source, descriptor))
			return True, self._queue.qsize()
		except Full:
			return False, self._queue.qsize()

	def _queue_loop(self):
		while True:
			item = self._queue.get()
			if item is None:
				break
			text, source, descriptor = item
			self._in_flight.set()
			try:
				self._ingest_sync(text, source, descriptor)
			except Exception:
				log.exception("Queue ingest failed")
			finally:
				self._in_flight.clear()

	def _ingest_sync(self, text: str, source: str = "", descriptor: str = "") -> IngestResult:
		result = self.ingester.ingest(text, source, descriptor)

		thought_ids = [t.id for t in result.committed]
		relinked, pairs_scanned = self._relink_thoughts(self.graph, thought_ids)
		result.relinked = relinked
		result.relink_pairs_scanned = pairs_scanned

		with self.mu:
			# Apply edge decay if configured
			if self.cfg.decay_coefficient > 0:
				self.graph.apply_edge_decay(self.cfg.decay_coefficient)
			if result.committed and self.graph.num_edges > 0:
				loss, _ = self.trainer.train_on_graph_with_routing(self.graph)
				log.info("GNN training loss: %.4f", loss)
			self.save()
		self._fire_event(
			GraphEvent(
				kind="ingest_complete",
				thoughts=self.graph.num_thoughts,
				edges=self.graph.num_edges,
				committed=len(result.committed),
			)
		)
		return result

	def ingest_sync(self, text: str, source: str = "", descriptor: str = "") -> dict:
		"""Ingest synchronously. Returns rich stats dict."""
		result = self._ingest_sync(text, source, descriptor)
		return {
			"committed": len(result.committed),
			"committed_thoughts": [{"id": t.id, "text": t.text[:200]} for t in result.committed],
			"rejected_to_limbo": result.rejected,
			"deduplicated": result.deduplicated,
			"relinked": result.relinked,
			"relink_pairs_scanned": result.relink_pairs_scanned,
			"thoughts": self.graph.num_thoughts,
			"edges": self.graph.num_edges,
		}

	def ingest(self, text: str, source: str = "", descriptor: str = "") -> str:
		"""Try async queue, fallback to sync. Returns status string."""
		queued, pending = self.enqueue(text, source, descriptor)
		if queued:
			return f"queued ({pending} pending)"
		# fallback to sync
		self.ingest_sync(text, source, descriptor)
		return "ok"

	def ask(self, query: str, cluster: str | None = None) -> tuple[str, list[dict]]:
		"""Retrieval pipeline: score + expand → deduplicate → rate → answer.

		When `cluster` is provided, only shards in that cluster group are queried.
		Otherwise, the default graph + all registered shards are queried.
		"""
		query_emb = self.provider.embed_text(query)

		all_scored: list[list[tuple]] = []
		all_chains: list[PathChain] = []

		if cluster:
			# Scoped query: only shards in this cluster
			store_names = self.registry.stores_in_cluster(cluster)
			for sname in store_names:
				shard = self._shards.get(sname)
				if shard is None:
					continue
				try:
					shard_scored, shard_chains = self._score_shard(query_emb, shard.graph, shard.model, shard.shard)
					all_scored.append(shard_scored)
					all_chains.extend(shard_chains)
				except Exception:
					log.warning("Shard '%s' query failed", sname, exc_info=True)
		else:
			# Full query: default graph + all shards
			local_scored, local_chains = self._score_shard(query_emb, self.graph, self.model, self.shard)
			all_scored.append(local_scored)
			all_chains.extend(local_chains)

			for name, shard in self._shards.items():
				# Profile-based routing: skip shards below similarity threshold
				if shard.graph.profile is not None:
					sim = cosine(shard.graph.profile, query_emb)
					if sim < self.cfg.query_routing_threshold:
						log.debug("Shard '%s' profile sim=%.3f < %.3f, skipping", name, sim, self.cfg.query_routing_threshold)
						continue
					log.debug("Shard '%s' profile sim=%.3f >= %.3f, including", name, sim, self.cfg.query_routing_threshold)
				try:
					shard_scored, shard_chains = self._score_shard(query_emb, shard.graph, shard.model, shard.shard)
					all_scored.append(shard_scored)
					all_chains.extend(shard_chains)
				except Exception:
					log.warning("Shard '%s' query failed", name, exc_info=True)

		# Deduplicate across shards
		scored = deduplicate(all_scored, self.cfg.top_k)
		if not scored:
			return "No relevant knowledge found.", []

		# Rate — re-rank thoughts by direct query relevance
		scored = rate_thoughts(query_emb, scored)

		# Filter chains to only those whose terminal thought survived scoring
		relevant_chains = best_chains_from(all_chains, scored)

		# Confidence gate: skip LLM when the graph already has a strong answer
		top_score = scored[0][1]
		if top_score >= self.cfg.confidence_threshold:
			log.info("Confidence gate: top_score=%.3f >= %.3f, skipping LLM", top_score, self.cfg.confidence_threshold)
			return synthesize_direct(scored)

		text, sources = answer(query, scored, self.provider, chains=relevant_chains or None)

		# Ingest the LLM answer back into the graph so future similar queries
		# can be answered directly — the learning flywheel.
		self.enqueue(text, source="query_response")

		# Periodic edge refinement based on retrieval feedback
		self._retrieval_count += 1
		if self.cfg.refinement_interval > 0 and self._retrieval_count % self.cfg.refinement_interval == 0:
			self.graph.refine_edges(
				boost=self.cfg.refinement_boost,
				dampen=self.cfg.refinement_dampen,
				min_traversals=self.cfg.refinement_min_traversals,
			)
			for shard in self._shards.values():
				shard.graph.refine_edges(
					boost=self.cfg.refinement_boost,
					dampen=self.cfg.refinement_dampen,
					min_traversals=self.cfg.refinement_min_traversals,
				)
			log.info("Edge refinement pass completed (retrieval #%d)", self._retrieval_count)

		return text, sources

	def find_thoughts_by_query(self, query: str, k: int = 5) -> list[dict]:
		"""Embed query and search for semantically similar thoughts across all shards."""
		emb = self.provider.embed_text(query)
		# Search global graph
		all_neighbors: list[tuple] = list(self.graph.find_thoughts(emb, k=k, threshold=self.cfg.similarity_threshold))
		# Search every shard
		for shard in self._shards.values():
			all_neighbors.extend(shard.graph.find_thoughts(emb, k=k, threshold=self.cfg.similarity_threshold))
		# Sort by similarity descending, deduplicate by text, return top-k
		seen: set[str] = set()
		results = []
		for t, sim in sorted(all_neighbors, key=lambda x: x[1], reverse=True):
			if t.text not in seen:
				seen.add(t.text)
				results.append({"id": t.id, "text": t.text, "similarity": round(sim, 3), "source": t.source})
			if len(results) >= k:
				break
		return results

	def status(self) -> str:
		queued = self._queue.qsize() if self._queue else 0
		in_flight = 1 if self._in_flight.is_set() else 0
		return (
			f"thoughts={self.graph.num_thoughts} "
			f"edges={self.graph.num_edges} "
			f"queued={queued} "
			f"in_flight={in_flight} "
			f"limbo={len(self.graph.limbo)} "
			f"purpose={self.graph.purpose or '(none)'}"
		)

	def get_diff(self) -> dict:
		"""Get diff from last poll. Returns added/changed items since last call."""
		current_stores = set(self.registry.list_stores().keys())

		current = {
			"thought_count": self.graph.num_thoughts,
			"edge_count": self.graph.num_edges,
			"thought_ids": set(self.graph.thoughts.keys()),
			"limbo_count": len(self.graph.limbo),
			"queue_size": self._queue.qsize() if self._queue else 0,
			"in_flight": 1 if self._in_flight.is_set() else 0,
			"stores": current_stores,
		}

		with self._last_poll_state_lock:
			if not self._last_poll_state:
				self._last_poll_state = current.copy()
				self._last_poll_state["thought_ids"] = current["thought_ids"].copy()
				return {
					"initial": True,
					"thought_count": current["thought_count"],
					"edge_count": current["edge_count"],
					"limbo_count": current["limbo_count"],
					"queue_size": current["queue_size"],
					"in_flight": current["in_flight"],
					"stores": list(current_stores),
					"new_stores": [],
				}

			prev = self._last_poll_state
			prev_stores = prev.get("stores", set())
			added_stores = current_stores - prev_stores

			added_thought_ids = current["thought_ids"] - prev["thought_ids"]

			new_thoughts = []
			if added_thought_ids:
				for tid in sorted(added_thought_ids):
					t = self.graph.thoughts.get(tid)
					if t:
						new_thoughts.append(
							{
								"id": t.id,
								"text": t.text,
								"source": t.source,
								"store": "global",
							}
						)

			diff = {
				"initial": False,
				"thought_count": current["thought_count"],
				"edge_count": current["edge_count"],
				"added_thoughts": new_thoughts,
				"added_count": len(new_thoughts),
				"limbo_count": current["limbo_count"],
				"queue_size": current["queue_size"],
				"in_flight": current["in_flight"],
				"stores": list(current_stores),
				"new_stores": list(added_stores),
			}

			self._last_poll_state = current.copy()
			self._last_poll_state["thought_ids"] = current["thought_ids"].copy()

			return diff

	def set_purpose(self, purpose: str):
		self.graph.purpose = purpose
		self.save()

	def add_descriptor(self, name: str, text: str):
		self.graph.descriptors[name] = text
		self.save()

	def remove_descriptor(self, name: str) -> bool:
		removed = self.graph.descriptors.pop(name, None) is not None
		if removed:
			self.save()
		return removed

	def resolve_descriptor(self, name: str) -> str:
		"""Resolve a descriptor name to its text. Returns '' if not found."""
		return self.graph.descriptors.get(name, "")

	def create_shard(self, name: str, purpose: str, location: str, cluster: str | None = None) -> str:
		"""Create a new Shard graph, save it, and register it.

		Returns the graph file path.
		"""
		hashed = store_path(location, name)
		base = hashed.with_suffix("")

		graph = Graph(
			name=name,
			purpose=purpose,
			max_thoughts=self.cfg.max_thoughts,
			max_edges=self.cfg.max_edges,
		)
		model = ShardMPNN(self.cfg)
		shard = ShardLayer(self.cfg.hidden_dim)
		save_all(graph, model, shard, base)

		graph_path = str(hashed)
		self.registry.register(graph_path)

		if cluster:
			self.registry.add_to_cluster(cluster, name)

		return graph_path

	def ingest_into_shard(
		self,
		shard_name: str,
		text: str,
		source: str = "",
		descriptor: str = "",
	) -> int:
		"""Ingest text directly into a named shard graph. Returns number committed."""
		s = self._shards.get(shard_name)
		if s is None:
			raise KeyError(f"Shard '{shard_name}' not loaded")
		ingester = Ingester(s.graph, self.provider, self.cfg)
		result = ingester.ingest(text, source=source, descriptor=descriptor)
		return len(result.committed)

	def ingested_sources(self) -> set[str]:
		"""Return the set of all source strings already in the graph and all shards."""
		sources: set[str] = set()
		for t in self.graph.thoughts.values():
			sources.add(t.source)
		for shard in self._shards.values():
			for t in shard.graph.thoughts.values():
				sources.add(t.source)
		return sources

	@property
	def graph_info(self) -> dict:
		"""Read-only snapshot of graph metadata. No internal objects exposed."""
		return {
			"purpose": self.graph.purpose,
			"thought_count": self.graph.num_thoughts,
			"edge_count": self.graph.num_edges,
			"maturity": self.graph.maturity,
			"descriptors": dict(self.graph.descriptors),
		}

	@property
	def all_thoughts(self) -> list[dict]:
		"""Snapshot of all thoughts as lightweight dicts."""
		return [{"id": t.id, "text": t.text[:200], "source": t.source} for t in self.graph.thoughts.values()]

	def graph_full(self) -> dict:
		"""Return complete graph as nodes + edges for visualization."""
		nodes = []
		edges = []
		seen_edge_keys: set[tuple[str, str]] = set()
		thought_vectors: list[tuple[dict, np.ndarray]] = []

		def _collect(graph, store: str):
			prefix = "" if store == "global" else f"{store}:"
			for t in sorted(graph.thoughts.values(), key=lambda thought: thought.id):
				node = {
					"key": f"{prefix}{t.id}",
					"label": t.text[:80],
					"source": t.source,
					"store": store,
					"type": "thought",
					"access_count": t.access_count,
					"created_at": t.created_at,
					"last_accessed": t.last_accessed,
				}
				nodes.append(node)
				if t.embedding is not None and len(t.embedding) > 0:
					thought_vectors.append((node, np.asarray(t.embedding, dtype=float).ravel()))
			for e in sorted(graph.edges, key=lambda edge: (edge.source_id, edge.target_id, edge.created_at)):
				key = (f"{prefix}{e.source_id}", f"{prefix}{e.target_id}")
				if key in seen_edge_keys:
					continue
				seen_edge_keys.add(key)
				edges.append(
					{
						"source": f"{prefix}{e.source_id}",
						"target": f"{prefix}{e.target_id}",
						"weight": round(e.weight, 3),
						"reasoning": e.reasoning[:120],
						"success_rate": round(e.success_rate, 3),
						"traversal_count": e.traversal_count,
						"created_at": e.created_at,
						"source_name": e.source,
					}
				)

		_collect(self.graph, "global")
		for name, shard in sorted(self._shards.items()):
			_collect(shard.graph, name)

		if thought_vectors:
			min_dim = min(vec.shape[0] for _, vec in thought_vectors)
			matrix = np.stack([vec[:min_dim] for _, vec in thought_vectors])
			centered = matrix - matrix.mean(axis=0, keepdims=True)
			coords = centered[:, :3] if min_dim >= 3 else centered
			if centered.shape[0] >= 2 and min_dim > 0:
				try:
					_, _, vt = np.linalg.svd(centered, full_matrices=False)
					basis = vt[: min(3, vt.shape[0])]
					coords = centered @ basis.T
				except np.linalg.LinAlgError:
					coords = centered[:, : min(3, centered.shape[1])]
			if coords.shape[1] < 3:
				coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])))
			scale = np.percentile(np.abs(coords), 95, axis=0)
			scale[scale < 1e-6] = 1.0
			normalized = np.clip(coords / scale, -1.5, 1.5)
			for (node, _), pos in zip(thought_vectors, normalized, strict=False):
				node["embed_pos"] = [round(float(v), 4) for v in pos[:3]]

		# Add shard hub nodes and link each shard's thoughts to it
		for name, shard in self._shards.items():
			hub_key = f"_shard:{name}"
			nodes.append(
				{
					"key": hub_key,
					"label": name,
					"source": "",
					"store": name,
					"type": "shard",
					"access_count": 0,
					"created_at": 0,
				}
			)
			for t in shard.graph.thoughts.values():
				edge_key = (hub_key, f"{name}:{t.id}")
				if edge_key not in seen_edge_keys:
					seen_edge_keys.add(edge_key)
					edges.append(
						{
							"source": hub_key,
							"target": f"{name}:{t.id}",
							"weight": 0.3,
							"reasoning": f"belongs to {name}",
							"success_rate": 0.0,
						}
					)

		# Compute k-nearest-neighbor edges based on embedding cosine similarity
		knn_edges = []
		# Collect all thoughts with their embeddings and prefixed keys
		all_thoughts = [(node["key"], vec) for node, vec in thought_vectors]

		if len(all_thoughts) > 1:
			from .util.math import cosine

			k = min(3, len(all_thoughts) - 1)
			for i, (key_i, emb_i) in enumerate(all_thoughts):
				sims = []
				for j, (key_j, emb_j) in enumerate(all_thoughts):
					if i == j:
						continue
					sims.append((cosine(emb_i, emb_j), key_j))
				sims.sort(key=lambda x: -x[0])
				for sim, key_j in sims[:k]:
					edge_key = (key_i, key_j)
					rev_key = (key_j, key_i)
					if edge_key not in seen_edge_keys and rev_key not in seen_edge_keys:
						seen_edge_keys.add(edge_key)
						knn_edges.append(
							{
								"source": key_i,
								"target": key_j,
								"weight": round(max(0.0, sim), 3),
								"reasoning": "embedding similarity",
								"success_rate": 0.0,
								"traversal_count": 0,
								"created_at": 0,
								"type": "knn",
							}
						)

		return {"nodes": nodes, "edges": edges, "knn_edges": knn_edges}

	def graph_meta(self) -> dict:
		"""Lightweight snapshot: shard hub nodes + total thought count for fast initial render."""
		nodes = []
		edges = []
		seen: set[tuple[str, str]] = set()
		total = self.graph.num_thoughts
		for name, shard in self._shards.items():
			total += shard.graph.num_thoughts
			hub_key = f"_shard:{name}"
			nodes.append({
				"key": hub_key,
				"label": name,
				"source": "",
				"store": name,
				"type": "shard",
				"access_count": 0,
				"created_at": 0,
			})
			for t in shard.graph.thoughts.values():
				ek = (hub_key, f"{name}:{t.id}")
				if ek not in seen:
					seen.add(ek)
					edges.append({
						"source": hub_key,
						"target": f"{name}:{t.id}",
						"weight": 0.3,
						"reasoning": f"belongs to {name}",
						"success_rate": 0.0,
					})
		return {"nodes": nodes, "edges": edges, "knn_edges": [], "total_thoughts": total}

	def graph_thoughts(self, offset: int = 0, limit: int = 200) -> dict:
		"""Paginated thought nodes from global graph and all shards."""
		all_nodes = []
		for t in sorted(self.graph.thoughts.values(), key=lambda t: t.id):
			all_nodes.append({
				"key": str(t.id),
				"label": t.text[:80],
				"source": t.source,
				"store": "global",
				"type": "thought",
				"access_count": t.access_count,
				"created_at": t.created_at,
				"last_accessed": t.last_accessed,
			})
		for name, shard in sorted(self._shards.items()):
			for t in sorted(shard.graph.thoughts.values(), key=lambda t: t.id):
				all_nodes.append({
					"key": f"{name}:{t.id}",
					"label": t.text[:80],
					"source": t.source,
					"store": name,
					"type": "thought",
					"access_count": t.access_count,
					"created_at": t.created_at,
					"last_accessed": t.last_accessed,
				})
		return {"nodes": all_nodes[offset : offset + limit]}

	def _relink_thoughts(self, graph: Graph, thought_ids: list[int]) -> tuple[int, int]:
		"""Relink only the specified thought IDs against all existing thoughts.

		Returns (edges_created, pairs_scanned).
		"""
		if not thought_ids:
			return 0, 0

		created = 0
		scanned = 0

		for tid in thought_ids:
			if tid not in graph.thoughts:
				continue
			thought = graph.thoughts[tid]

			neighbors = graph.find_thoughts(thought.embedding, k=50, threshold=self.cfg.similarity_threshold)
			candidates = [(t, sim) for t, sim in neighbors if t.id != tid]

			existing_neighbors = {nid for nid, _ in graph.get_neighbors(tid)}

			to_link = [(t, sim) for t, sim in candidates if t.id not in existing_neighbors]
			scanned += len(to_link)

			if not to_link:
				continue

			cand_texts = [t.text for t, _ in to_link]
			results = self.provider.batch_link_reason(thought.text, cand_texts)

			valid = [r for r in results if r["weight"] >= self.cfg.min_link_weight and 0 <= r["index"] < len(to_link)]

			if not valid:
				continue

			reasoning_texts = [r["reasoning"] for r in valid]
			embeddings = self.provider.embed_texts(reasoning_texts)

			for r, emb in zip(valid, embeddings):
				target = to_link[r["index"]][0]
				edge = graph.add_edge(
					source_id=thought.id,
					target_id=target.id,
					weight=r["weight"],
					reasoning=r["reasoning"],
					embedding=emb,
				)
				if edge is not None:
					created += 1

		return created, scanned

	def relink(self) -> dict:
		"""Scan all thoughts and create missing edges between similar pairs.

		Works on the global graph and all shards. Uses the LLM to
		generate link reasoning for newly discovered connections.
		"""
		total_created = 0
		total_scanned = 0

		def _relink_graph(graph: Graph) -> int:
			nonlocal total_scanned
			created = 0
			thoughts = list(graph.thoughts.values())
			if len(thoughts) < 2:
				return 0

			# Build set of existing edge pairs for fast lookup
			existing: set[tuple[int, int]] = set()
			for e in graph.edges:
				existing.add((e.source_id, e.target_id))
				existing.add((e.target_id, e.source_id))

			# Find similar pairs that aren't connected
			pairs_to_link: list[tuple[Thought, Thought, float]] = []
			for i, a in enumerate(thoughts):
				for b in thoughts[i + 1 :]:
					total_scanned += 1
					if (a.id, b.id) in existing:
						continue
					sim = cosine(a.embedding, b.embedding)
					if sim >= self.cfg.similarity_threshold:
						pairs_to_link.append((a, b, sim))

			if not pairs_to_link:
				return 0

			# Batch LLM link reasoning — group by source thought
			from collections import defaultdict

			groups: dict[int, list[tuple[Thought, float]]] = defaultdict(list)
			src_map: dict[int, Thought] = {}
			for a, b, sim in pairs_to_link:
				groups[a.id].append((b, sim))
				src_map[a.id] = a

			for src_id, candidates in groups.items():
				src = src_map[src_id]
				cand_texts = [c.text for c, _ in candidates]
				cand_thoughts = [c for c, _ in candidates]

				results = self.provider.batch_link_reason(src.text, cand_texts)
				valid = [r for r in results if r["weight"] >= self.cfg.min_link_weight and 0 <= r["index"] < len(cand_thoughts)]
				if not valid:
					continue

				reasoning_texts = [r["reasoning"] for r in valid]
				embeddings = self.provider.embed_texts(reasoning_texts)

				for r, emb in zip(valid, embeddings):
					target = cand_thoughts[r["index"]]
					edge = graph.add_edge(
						source_id=src.id,
						target_id=target.id,
						weight=r["weight"],
						reasoning=r["reasoning"],
						embedding=emb,
					)
					if edge is not None:
						created += 1

			return created

		with self.mu:
			total_created += _relink_graph(self.graph)
			for name, shard in self._shards.items():
				total_created += _relink_graph(shard.graph)

			if total_created > 0:
				self.save()

		log.info("Relink: created %d edges (scanned %d pairs)", total_created, total_scanned)
		return {
			"edges_created": total_created,
			"pairs_scanned": total_scanned,
			"total_edges": self.graph.num_edges,
			"total_thoughts": self.graph.num_thoughts,
		}

	def explore_thought(self, thought_id: int) -> dict | None:
		"""Return a thought with its edges and neighbors, or None if not found.

		Searches the global graph and all shards.
		"""
		# Search global graph first
		graph, store = self.graph, "global"
		thought = graph.thoughts.get(thought_id)

		# If not in global, search shards
		if thought is None:
			for name, shard in self._shards.items():
				thought = shard.graph.thoughts.get(thought_id)
				if thought is not None:
					graph, store = shard.graph, name
					break

		if thought is None:
			return None

		neighbors = graph.get_neighbors(thought_id)
		edges = []
		for neighbor_id, edge in neighbors:
			neighbor = graph.thoughts.get(neighbor_id)
			edges.append(
				{
					"target_id": neighbor_id,
					"target_text": neighbor.text[:200] if neighbor else "",
					"weight": round(edge.weight, 3),
					"reasoning": edge.reasoning,
					"success_rate": round(edge.success_rate, 3),
					"traversal_count": edge.traversal_count,
				}
			)

		return {
			"id": thought.id,
			"text": thought.text,
			"source": thought.source,
			"created_at": thought.created_at,
			"access_count": thought.access_count,
			"last_accessed": thought.last_accessed,
			"store": store,
			"edges": edges,
			"neighbor_count": len(edges),
		}

	def traverse(self, start_id: int, depth: int = 2, max_nodes: int = 50) -> dict | None:
		"""BFS from a thought, returning the local subgraph.

		Searches across the global graph and all shards.
		"""
		# Find the starting thought and its graph
		graph, store = self.graph, "global"
		if start_id not in graph.thoughts:
			for name, shard in self._shards.items():
				if start_id in shard.graph.thoughts:
					graph, store = shard.graph, name
					break
			else:
				return None

		nodes = []
		edge_list = []
		visited: set[int] = set()
		frontier = [(start_id, 0)]
		truncated = False

		while frontier:
			tid, d = frontier.pop(0)
			if tid in visited:
				continue
			if len(visited) >= max_nodes:
				truncated = True
				break
			visited.add(tid)

			thought = graph.thoughts.get(tid)
			if thought is None:
				continue

			nodes.append(
				{
					"id": thought.id,
					"text": thought.text[:200],
					"source": thought.source,
					"store": store,
					"depth": d,
				}
			)

			if d < depth:
				for neighbor_id, edge in graph.get_neighbors(tid):
					if neighbor_id not in visited:
						frontier.append((neighbor_id, d + 1))
					# Record edge (avoid duplicates by canonical ordering)
					edge_key = (min(edge.source_id, edge.target_id), max(edge.source_id, edge.target_id))
					edge_entry = {
						"source_id": edge.source_id,
						"target_id": edge.target_id,
						"weight": round(edge.weight, 3),
						"reasoning": edge.reasoning,
					}
					if edge_entry not in edge_list:
						edge_list.append(edge_entry)

		return {
			"root": start_id,
			"store": store,
			"nodes": nodes,
			"edges": edge_list,
			"truncated": truncated,
		}

	def graph_stats(self) -> dict:
		"""Aggregate statistics across the global graph and all shards."""

		def _edge_stats(edges):
			if not edges:
				return 0.0, 0.0
			weights = [e.weight for e in edges]
			rates = [e.success_rate for e in edges if e.traversal_count > 0]
			avg_w = sum(weights) / len(weights)
			avg_sr = sum(rates) / len(rates) if rates else 0.0
			return round(avg_w, 3), round(avg_sr, 3)

		avg_w, avg_sr = _edge_stats(self.graph.edges)
		stats = {
			"global": {
				"thoughts": self.graph.num_thoughts,
				"edges": self.graph.num_edges,
				"maturity": round(self.graph.maturity, 3),
				"limbo": len(self.graph.limbo),
				"purpose": self.graph.purpose or "",
				"descriptors": len(self.graph.descriptors),
				"avg_edge_weight": avg_w,
				"avg_edge_success_rate": avg_sr,
			},
			"shards": [],
		}
		for name, entry in self._index.items():
			shard = self._shards.get(name)
			entry = {
				"name": entry.name,
				"purpose": entry.purpose,
				"thoughts": entry.num_thoughts,
				"edges": entry.num_edges,
			}
			if shard:
				sw, ssr = _edge_stats(shard.graph.edges)
				entry["maturity"] = round(shard.graph.maturity, 3)
				entry["avg_edge_weight"] = sw
				entry["avg_edge_success_rate"] = ssr
			stats["shards"].append(entry)

		stats["total_shards"] = len(self._index)
		return stats

	def list_shards(self) -> list[dict]:
		"""Return metadata for all loaded shards, including cluster membership."""
		# Build reverse index: store_name → list of clusters it belongs to
		cluster_membership: dict[str, list[str]] = {}
		for cluster_name, members in self.registry.clusters.items():
			for member in members:
				cluster_membership.setdefault(member, []).append(cluster_name)

		result = []
		for name, entry in self._index.items():
			result.append(
				{
					"name": entry.name,
					"purpose": entry.purpose,
					"num_thoughts": entry.num_thoughts,
					"num_edges": entry.num_edges,
					"descriptors": entry.descriptors,
					"clusters": cluster_membership.get(name, []),
				}
			)
		return result

	def on(self, event: str, listener: EventListener) -> "Handler":
		"""Register an event listener. event: 'ingest_complete' | 'limbo_promoted' | '*'"""
		with self._listeners_lock:
			self._listeners.setdefault(event, []).append(listener)
		return self

	def off(self, event: str, listener: EventListener) -> None:
		"""Deregister an event listener."""
		with self._listeners_lock:
			bucket = self._listeners.get(event, [])
			self._listeners[event] = [l for l in bucket if l is not listener]

	def _fire_event(self, event: GraphEvent) -> None:
		with self._listeners_lock:
			specific = list(self._listeners.get(event.kind, []))
			wildcard = list(self._listeners.get("*", []))
		for listener in specific + wildcard:
			try:
				listener(event)
			except Exception:
				log.exception("Event listener raised for kind=%s", event.kind)

	def _upsert_shard_node(self, name: str, shard: "shard"):
		"""Upsert a registry node for this shard in the global graph.

		Embeds the shard's aggregate profile as a thought in the global graph
		so the GNN learns where each shard lives in the full knowledge network.
		"""
		if shard.graph.profile is None:
			return

		profile = shard.graph.profile.copy()
		text = f"[shard:{name}] {shard.purpose}"

		existing_tid = self.graph._registry_nodes.get(name)
		if existing_tid and existing_tid in self.graph.thoughts:
			# Update embedding + text on existing registry node
			self.graph.thoughts[existing_tid].embedding = profile
			self.graph.thoughts[existing_tid].text = text
			self.graph._update_profile(profile)
		else:
			# Create new registry node + edges to nearby global thoughts
			t = self.graph.add_thought(text, profile, source=f"Shard:{name}")
			if t is None:
				return
			self.graph._registry_nodes[name] = t.id
			neighbors = self.graph.find_thoughts(profile, k=self.cfg.top_k, threshold=self.cfg.similarity_threshold)
			for neighbor, sim in neighbors:
				if neighbor.id != t.id:
					self.graph.add_edge(
						source_id=t.id,
						target_id=neighbor.id,
						weight=sim,
						reasoning=f"Shard '{name}' covers this topic",
						embedding=profile,
					)

	def _load_shards(self):
		"""Load all registered shards into cache, build index, and upsert registry nodes."""
		# Auto-discover .shard files in the shard directory that aren't yet registered
		shard_dir = Path(self.cfg.graph_path).parent
		if shard_dir.is_dir():
			from .shard.store import read_shard_metadata as _read_meta
			registered_paths = {Path(e["path"]).resolve() for e in self.registry.stores.values()}
			skip = {Path(self.cfg.graph_path).resolve()}
			if self.cfg.base_gnn_path:
				skip.add(Path(self.cfg.base_gnn_path).resolve())
			# Also skip base.shard — it's the PyTorch base model, not a graph shard
			skip.add((shard_dir / "base.shard").resolve())
			for shard_file in shard_dir.glob("*.shard"):
				resolved = shard_file.resolve()
				if resolved in skip:
					continue
				if resolved in registered_paths:
					continue
				try:
					_read_meta(str(resolved))  # validates it's a real graph shard
				except Exception:
					log.debug("Skipping non-graph file: %s", shard_file.name)
					continue
				try:
					self.registry.register(str(resolved))
					log.info("Auto-registered shard: %s", shard_file.name)
				except Exception:
					log.warning("Failed to auto-register %s", shard_file, exc_info=True)

		stale: list[str] = []
		for name, entry in self.registry.stores.items():
			try:
				graph_path = entry["path"]
				if not Path(graph_path).exists():
					log.warning("Shard '%s' graph not found, removing: %s", name, graph_path)
					stale.append(name)
					continue
				base = Path(graph_path).with_suffix("")
				graph, model, shard_layer = load_all(self.cfg, base)
				self._shards[name] = Shard(
					name=name,
					purpose=entry.get("purpose", "") or graph.purpose,
					graph=graph,
					model=model,
					shard=shard_layer,
				)
				# Build index entry from loaded graph metadata
				self._index[name] = ShardIndexEntry(
					name=name,
					purpose=graph.purpose,
					descriptors=dict(graph.descriptors),
					profile=graph.profile.copy() if graph.profile is not None else None,
					num_thoughts=graph.num_thoughts,
					num_edges=graph.num_edges,
				)
				log.info("Loaded Shard '%s' (%d thoughts, %d edges)", name, graph.num_thoughts, graph.num_edges)
			except Exception:
				log.warning("Failed to load Shard '%s'", name, exc_info=True)

		for name in stale:
			self.registry.unregister(name)
		if stale:
			log.info("Removed %d stale Shard(s) from registry", len(stale))

		log.info("Shard index: %d entries loaded", len(self._index))

		# Upsert registry nodes so the global graph knows about all shards
		for name, shard in self._shards.items():
			self._upsert_shard_node(name, shard)

	def _score_shard(
		self,
		query_emb,
		graph: Graph,
		model: ShardMPNN,
		shard: ShardLayer,
	) -> tuple[list[tuple], list[PathChain]]:
		"""Run all three scoring signals + merge + expand for one Shard.

		Returns (scored_thoughts, path_chains).

		Pass 1: standard merge → seed thoughts (direct high-scoring matches).
		Pass 2: scan all thoughts at a relaxed cosine floor to find distant
		         targets that didn't survive merge threshold.
		Then expand() does Dijkstra path traversal from seeds towards targets.
		"""
		if not graph.thoughts:
			return [], []

		cos = cosine_scores(query_emb, graph)
		try:
			gnn = gnn_scores(query_emb, graph, model, shard)
		except Exception:
			log.debug("GNN scoring failed", exc_info=True)
			gnn = {}
		edg = edge_scores(query_emb, graph, self.cfg)

		seeds = merge(cos, gnn, edg, graph, self.cfg)
		if not seeds:
			return [], []

		# Pass 2: find distant targets — thoughts with decent cosine that
		# didn't make it through merge.  These become Dijkstra endpoints.
		seed_ids = {t.id for t, _ in seeds}
		distant_floor = max(self.cfg.similarity_threshold * 0.6, 0.15)
		targets: set[int] = set()
		for tid, sim in cos.items():
			if tid not in seed_ids and sim >= distant_floor:
				targets.add(tid)

		scored, chains = expand(seeds, query_emb, graph, self.cfg, targets=targets or None)
		return scored, chains

	def _limbo_scan_loop(self):
		"""Background: scan limbo every cfg.limbo_scan_interval seconds."""
		while not self._shutdown.is_set():
			self._shutdown.wait(self.cfg.limbo_scan_interval)
			if self._shutdown.is_set():
				break
			try:
				with self.mu:
					self._scan_limbo()
					self.save()
			except Exception:
				log.exception("Limbo scan failed")

	def _scan_limbo(self):
		"""Find clusters in limbo and promote them."""
		clusters = find_clusters(self.graph, self.cfg)
		if not clusters:
			return

		promoted_indices: set[int] = set()
		modified_shards: set[str] = set()
		for cluster_indices in clusters:
			cluster_thoughts = [self.graph.limbo[i] for i in cluster_indices]
			try:
				shard_name = promote_cluster(
					cluster_thoughts,
					self._shards,
					self.provider,
					self.cfg,
					self.registry,
					self.cfg.graph_path,
				)
				promoted_indices.update(cluster_indices)
				if shard_name:
					modified_shards.add(shard_name)
			except Exception:
				log.exception("Cluster promotion failed")

		if promoted_indices:
			self.graph.limbo = [lt for i, lt in enumerate(self.graph.limbo) if i not in promoted_indices]

		# Upsert registry nodes for modified shards into global graph
		for shard_name in modified_shards:
			if shard_name in self._shards:
				self._upsert_shard_node(shard_name, self._shards[shard_name])
			self._fire_event(
				GraphEvent(
					kind="limbo_promoted",
					thoughts=self.graph.num_thoughts,
					edges=self.graph.num_edges,
					committed=0,
					detail={"shard": shard_name},
				)
			)

	def _poll_loop(self):
		"""Background: poll registry for new stores every 5 seconds."""
		while not self._shutdown.is_set():
			self._shutdown.wait(5)
			if self._shutdown.is_set():
				break
			try:
				self._poll_stores()
			except Exception:
				log.exception("Store poll failed")

	def _poll_stores(self):
		"""Check registry for new stores and load them."""
		current_stores = set(self.registry.stores.keys())
		loaded_stores = set(self._shards.keys())

		# Initialize on first poll
		if not self._last_shard_state:
			for name in loaded_stores:
				shard = self._shards.get(name)
				if shard:
					self._last_shard_state[name] = (shard.graph.num_thoughts, shard.graph.num_edges)
			return

		# Check for new stores
		new_stores = current_stores - loaded_stores
		loaded_before = set(self._shards.keys())

		for name in new_stores:
			entry = self.registry.stores.get(name)
			if not entry:
				continue
			graph_path = entry["path"]
			if not Path(graph_path).exists():
				log.warning("New store '%s' graph not found: %s", name, graph_path)
				continue
			try:
				base = Path(graph_path).with_suffix("")
				graph, model, shard_layer = load_all(self.cfg, base)
				self._shards[name] = shard(
					name=name,
					purpose=entry.get("purpose", "") or graph.purpose,
					graph=graph,
					model=model,
					shard=shard_layer,
				)
				self._index[name] = ShardIndexEntry(
					name=name,
					purpose=graph.purpose,
					descriptors=dict(graph.descriptors),
					profile=graph.profile.copy() if graph.profile is not None else None,
					num_thoughts=graph.num_thoughts,
					num_edges=graph.num_edges,
				)
				self._upsert_shard_node(name, self._shards[name])
				log.info("Loaded new store '%s' (%d thoughts, %d edges)", name, graph.num_thoughts, graph.num_edges)
				self._fire_event(
					GraphEvent(
						kind="store_added",
						thoughts=self.graph.num_thoughts,
						edges=self.graph.num_edges,
						committed=0,
						detail={"store": name},
					)
				)
			except Exception:
				log.warning("Failed to load new store '%s'", name, exc_info=True)

		# Check for removed stores
		removed_stores = loaded_stores - current_stores
		for name in removed_stores:
			del self._shards[name]
			self._index.pop(name, None)
			self.graph._registry_nodes.pop(name, None)
			self._last_shard_state.pop(name, None)
			log.info("Removed store '%s'", name)
			self._fire_event(
				GraphEvent(
					kind="store_removed",
					thoughts=self.graph.num_thoughts,
					edges=self.graph.num_edges,
					committed=0,
					detail={"store": name},
				)
			)

		# Track changes to existing stores
		for name, shard in self._shards.items():
			prev = self._last_shard_state.get(name, (0, 0))
			curr = (shard.graph.num_thoughts, shard.graph.num_edges)
			if curr != prev:
				self._last_shard_state[name] = curr
				self._fire_event(
					GraphEvent(
						kind="store_changed",
						thoughts=self.graph.num_thoughts,
						edges=self.graph.num_edges,
						committed=curr[0] - prev[0],
						detail={"store": name, "thoughts": curr[0], "edges": curr[1]},
					)
				)
