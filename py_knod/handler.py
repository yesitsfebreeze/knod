"""Handler — owns all specialists; orchestrates ingest, limbo, and retrieval."""

import logging
import threading
import time as _time
from pathlib import Path
from queue import Queue, Full

import numpy as np

from .config import Config
from .provider import Provider
from .registry import Registry
from .specialist import (
	Graph,
	KnodMPNN,
	StrandLayer,
	GNNTrainer,
	GraphEvent,
	EventListener,
	SpecialistIndexEntry,
	Specialist,
	IngestResult,
	save_all,
	load_all,
	load_base_model,
	read_knod_metadata,
)
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
		self.model: KnodMPNN | None = None
		self.strand: StrandLayer | None = None
		self.trainer: GNNTrainer | None = None
		self.ingester: Ingester | None = None
		self._specialists: dict[str, Specialist] = {}
		self._index: dict[str, SpecialistIndexEntry] = {}  # specialist name → index entry
		self.mu = threading.Lock()
		self._queue: Queue | None = None
		self._queue_worker: threading.Thread | None = None
		self._limbo_thread: threading.Thread | None = None
		self._shutdown = threading.Event()
		self._in_flight = threading.Event()  # set while an ingest is running
		# Event bus — replaces raw-socket pub/sub
		self._listeners: dict[str, list[EventListener]] = {}
		self._listeners_lock = threading.Lock()
		self._retrieval_count: int = 0

	def init(self):
		base = Path(self.cfg.graph_path).with_suffix("")
		graph_file = base.with_suffix(".graph")

		if graph_file.exists():
			log.info("Loading existing graph from %s", graph_file)
			self.graph, self.model, self.strand = load_all(self.cfg, base)
		else:
			log.info("Creating new graph")
			self.graph = Graph(
				max_thoughts=self.cfg.max_thoughts,
				max_edges=self.cfg.max_edges,
				maturity_divisor=self.cfg.maturity_divisor,
			)
			self.model = KnodMPNN(self.cfg)
			self.strand = StrandLayer(self.cfg.hidden_dim)

		self.trainer = GNNTrainer(self.model, self.strand, self.cfg)
		self.ingester = Ingester(self.graph, self.provider, self.cfg)

		# Load registered specialists
		self._load_specialists()

		# Start async ingest queue
		self._queue = Queue(maxsize=QUEUE_CAPACITY)
		self._queue_worker = threading.Thread(target=self._queue_loop, daemon=True)
		self._queue_worker.start()

		# Start limbo background scan
		self._limbo_thread = threading.Thread(target=self._limbo_scan_loop, daemon=True)
		self._limbo_thread.start()

	def save(self):
		base = Path(self.cfg.graph_path).with_suffix("")
		save_all(self.graph, self.model, self.strand, base)

	def shutdown(self):
		self._shutdown.set()
		if self._queue is not None:
			self._queue.put(None)  # sentinel
		self.save()

	# ---- async ingest queue ----

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

	# ---- core operations ----

	def _ingest_sync(self, text: str, source: str = "", descriptor: str = "") -> IngestResult:
		result = self.ingester.ingest(text, source, descriptor)
		with self.mu:
			# Apply edge decay if configured
			if self.cfg.decay_coefficient > 0:
				self.graph.apply_edge_decay(self.cfg.decay_coefficient)
			if result.committed and self.graph.num_edges > 0:
				# Reload base model (may have been updated by specialist training)
				load_base_model(self.model)
				loss = self.trainer.train_on_graph(self.graph)
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

	def ask(self, query: str, knid: str | None = None) -> tuple[str, list[dict]]:
		"""Retrieval pipeline: score + expand → deduplicate → rate → answer.

		When `knid` is provided, only specialists in that knid group are queried.
		Otherwise, the default graph + all registered specialists are queried.
		"""
		query_emb = self.provider.embed_text(query)

		all_scored: list[list[tuple]] = []
		all_chains: list[PathChain] = []

		if knid:
			# Scoped query: only specialists in this knid
			store_names = self.registry.stores_in_knid(knid)
			for sname in store_names:
				spec = self._specialists.get(sname)
				if spec is None:
					continue
				try:
					spec_scored, spec_chains = self._score_specialist(query_emb, spec.graph, spec.model, spec.strand)
					all_scored.append(spec_scored)
					all_chains.extend(spec_chains)
				except Exception:
					log.warning("Specialist '%s' query failed", sname, exc_info=True)
		else:
			# Full query: default graph + all specialists
			local_scored, local_chains = self._score_specialist(query_emb, self.graph, self.model, self.strand)
			all_scored.append(local_scored)
			all_chains.extend(local_chains)

			for name, spec in self._specialists.items():
				if spec.graph.profile is not None:
					sim = cosine(spec.graph.profile, query_emb)
					log.debug("Specialist '%s' profile sim=%.3f", name, sim)
				try:
					spec_scored, spec_chains = self._score_specialist(query_emb, spec.graph, spec.model, spec.strand)
					all_scored.append(spec_scored)
					all_chains.extend(spec_chains)
				except Exception:
					log.warning("Specialist '%s' query failed", name, exc_info=True)

		# Deduplicate across specialists
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
			for spec in self._specialists.values():
				spec.graph.refine_edges(
					boost=self.cfg.refinement_boost,
					dampen=self.cfg.refinement_dampen,
					min_traversals=self.cfg.refinement_min_traversals,
				)
			log.info("Edge refinement pass completed (retrieval #%d)", self._retrieval_count)

		return text, sources

	def find_thoughts_by_query(self, query: str, k: int = 5) -> list[dict]:
		"""Embed query and search for semantically similar thoughts across all specialists."""
		emb = self.provider.embed_text(query)
		# Search global graph
		all_neighbors: list[tuple] = list(self.graph.find_thoughts(emb, k=k, threshold=self.cfg.similarity_threshold))
		# Search every specialist
		for spec in self._specialists.values():
			all_neighbors.extend(spec.graph.find_thoughts(emb, k=k, threshold=self.cfg.similarity_threshold))
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

	def ingest_into_specialist(
		self,
		specialist_name: str,
		text: str,
		source: str = "",
		descriptor: str = "",
	) -> int:
		"""Ingest text directly into a named specialist graph. Returns number committed."""
		spec = self._specialists.get(specialist_name)
		if spec is None:
			raise KeyError(f"Specialist '{specialist_name}' not loaded")
		ingester = Ingester(spec.graph, self.provider, self.cfg)
		result = ingester.ingest(text, source=source, descriptor=descriptor)
		return len(result.committed)

	def ask_knid(self, query: str, knid: str) -> tuple[str, list[dict]]:
		"""Ask scoped to all specialists in a knid group."""
		store_names = self.registry.stores_in_knid(knid)
		query_emb = self.provider.embed_text(query)
		all_scored = []
		all_chains: list[PathChain] = []
		for sname in store_names:
			spec = self._specialists.get(sname)
			if spec is None:
				continue
			spec_scored, spec_chains = self._score_specialist(query_emb, spec.graph, spec.model, spec.strand)
			all_scored.append(spec_scored)
			all_chains.extend(spec_chains)
		scored = deduplicate(all_scored, self.cfg.top_k)
		if not scored:
			return "No relevant knowledge found in knid.", []

		# Rate — re-rank thoughts by direct query relevance (parity with ask())
		scored = rate_thoughts(query_emb, scored)

		# Filter chains to only those whose terminal thought survived scoring
		relevant_chains = best_chains_from(all_chains, scored)

		return answer(query, scored, self.provider, chains=relevant_chains or None)

	def ingested_sources(self) -> set[str]:
		"""Return the set of all source strings already in the graph and all specialists."""
		sources: set[str] = set()
		for t in self.graph.thoughts.values():
			sources.add(t.source)
		for spec in self._specialists.values():
			for t in spec.graph.thoughts.values():
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

	# ---- graph inspection tools ----

	def explore_thought(self, thought_id: int) -> dict | None:
		"""Return a thought with its edges and neighbors, or None if not found.

		Searches the global graph and all specialists.
		"""
		# Search global graph first
		graph, store = self.graph, "global"
		thought = graph.thoughts.get(thought_id)

		# If not in global, search specialists
		if thought is None:
			for name, spec in self._specialists.items():
				thought = spec.graph.thoughts.get(thought_id)
				if thought is not None:
					graph, store = spec.graph, name
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

		Searches across the global graph and all specialists.
		"""
		# Find the starting thought and its graph
		graph, store = self.graph, "global"
		if start_id not in graph.thoughts:
			for name, spec in self._specialists.items():
				if start_id in spec.graph.thoughts:
					graph, store = spec.graph, name
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
		"""Aggregate statistics across the global graph and all specialists."""

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
			"specialists": [],
		}
		for name, entry in self._index.items():
			spec = self._specialists.get(name)
			spec_entry = {
				"name": entry.name,
				"purpose": entry.purpose,
				"thoughts": entry.num_thoughts,
				"edges": entry.num_edges,
			}
			if spec:
				sw, ssr = _edge_stats(spec.graph.edges)
				spec_entry["maturity"] = round(spec.graph.maturity, 3)
				spec_entry["avg_edge_weight"] = sw
				spec_entry["avg_edge_success_rate"] = ssr
			stats["specialists"].append(spec_entry)

		stats["total_specialists"] = len(self._index)
		return stats

	def list_specialists(self) -> list[dict]:
		"""Return metadata for all loaded specialists, including knid membership."""
		# Build reverse index: store_name → list of knids it belongs to
		knid_membership: dict[str, list[str]] = {}
		for knid_name, members in self.registry.knids.items():
			for member in members:
				knid_membership.setdefault(member, []).append(knid_name)

		result = []
		for name, entry in self._index.items():
			result.append(
				{
					"name": entry.name,
					"purpose": entry.purpose,
					"num_thoughts": entry.num_thoughts,
					"num_edges": entry.num_edges,
					"descriptors": entry.descriptors,
					"knids": knid_membership.get(name, []),
				}
			)
		return result

	# ---- event bus (replaces raw-socket pub/sub) ----

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

	# ---- specialist management ----

	def _upsert_specialist_node(self, name: str, spec: "Specialist"):
		"""Upsert a registry node for this specialist in the global graph.

		Embeds the specialist's aggregate profile as a thought in the global graph
		so the GNN learns where each specialist lives in the full knowledge network.
		"""
		if spec.graph.profile is None:
			return

		profile = spec.graph.profile.copy()
		text = f"[specialist:{name}] {spec.purpose}"

		existing_tid = self.graph._registry_nodes.get(name)
		if existing_tid and existing_tid in self.graph.thoughts:
			# Update embedding + text on existing registry node
			self.graph.thoughts[existing_tid].embedding = profile
			self.graph.thoughts[existing_tid].text = text
			self.graph._update_profile(profile)
		else:
			# Create new registry node + edges to nearby global thoughts
			t = self.graph.add_thought(text, profile, source=f"specialist:{name}")
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
						reasoning=f"Specialist '{name}' covers this topic",
						embedding=profile,
					)

	def _load_specialists(self):
		"""Load all registered specialists into cache, build index, and upsert registry nodes."""
		for name, entry in self.registry.stores.items():
			try:
				graph_path = entry["path"]
				if not Path(graph_path).exists():
					log.warning("Specialist '%s' graph not found: %s", name, graph_path)
					continue
				base = Path(graph_path).with_suffix("")
				graph, model, strand = load_all(self.cfg, base)
				self._specialists[name] = Specialist(
					name=name,
					purpose=entry.get("purpose", "") or graph.purpose,
					graph=graph,
					model=model,
					strand=strand,
				)
				# Build index entry from loaded graph metadata
				self._index[name] = SpecialistIndexEntry(
					name=name,
					purpose=graph.purpose,
					descriptors=dict(graph.descriptors),
					profile=graph.profile.copy() if graph.profile is not None else None,
					num_thoughts=graph.num_thoughts,
					num_edges=graph.num_edges,
				)
				log.info("Loaded specialist '%s' (%d thoughts, %d edges)", name, graph.num_thoughts, graph.num_edges)
			except Exception:
				log.warning("Failed to load specialist '%s'", name, exc_info=True)

		log.info("Specialist index: %d entries loaded", len(self._index))

		# Upsert registry nodes so the global graph knows about all specialists
		for name, spec in self._specialists.items():
			self._upsert_specialist_node(name, spec)

	def _score_specialist(
		self,
		query_emb,
		graph: Graph,
		model: KnodMPNN,
		strand: StrandLayer,
	) -> tuple[list[tuple], list[PathChain]]:
		"""Run all three scoring signals + merge + expand for one specialist.

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
			gnn = gnn_scores(query_emb, graph, model, strand)
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

	# ---- limbo background scan ----

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
		modified_specialists: set[str] = set()
		for cluster_indices in clusters:
			cluster_thoughts = [self.graph.limbo[i] for i in cluster_indices]
			try:
				spec_name = promote_cluster(
					cluster_thoughts,
					self._specialists,
					self.provider,
					self.cfg,
					self.registry,
					self.cfg.graph_path,
				)
				promoted_indices.update(cluster_indices)
				if spec_name:
					modified_specialists.add(spec_name)
			except Exception:
				log.exception("Cluster promotion failed")

		if promoted_indices:
			self.graph.limbo = [lt for i, lt in enumerate(self.graph.limbo) if i not in promoted_indices]

		# Upsert registry nodes for modified specialists into global graph
		for spec_name in modified_specialists:
			if spec_name in self._specialists:
				self._upsert_specialist_node(spec_name, self._specialists[spec_name])
			self._fire_event(
				GraphEvent(
					kind="limbo_promoted",
					thoughts=self.graph.num_thoughts,
					edges=self.graph.num_edges,
					committed=0,
					detail={"specialist": spec_name},
				)
			)
