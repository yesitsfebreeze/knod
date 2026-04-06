"""Handler — owns all specialists; orchestrates ingest, limbo, and retrieval."""

import logging
import threading
import time as _time
from pathlib import Path
from queue import Queue, Full

from .config import Config
from .provider import Provider
from .registry import Registry
from .specialist import (
	Graph,
	KnodMPNN,
	StrandLayer,
	GNNTrainer,
	save_all,
	load_all,
)
from .ingest import Ingester
from .limbo import find_clusters, promote_cluster
from .retrieval import cosine_scores, edge_scores, gnn_scores, merge, deduplicate, answer

log = logging.getLogger(__name__)

QUEUE_CAPACITY = 128


class Specialist:
	"""One specialist = graph + model + strand."""

	__slots__ = ("name", "purpose", "graph", "model", "strand")

	def __init__(self, name, purpose, graph, model, strand):
		self.name = name
		self.purpose = purpose
		self.graph = graph
		self.model = model
		self.strand = strand


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
		self.mu = threading.Lock()
		self._queue: Queue | None = None
		self._queue_worker: threading.Thread | None = None
		self._limbo_thread: threading.Thread | None = None
		self._shutdown = threading.Event()
		self._in_flight = threading.Event()  # set while an ingest is running
		self._subs: list = []
		self._subs_lock = threading.Lock()

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

	def _ingest_sync(self, text: str, source: str = "", descriptor: str = "") -> int:
		committed = self.ingester.ingest(text, source, descriptor)
		n_committed = len(committed)
		with self.mu:
			# Apply edge decay if configured
			if self.cfg.decay_coefficient > 0:
				self.graph.apply_edge_decay(self.cfg.decay_coefficient)
			if committed and self.graph.num_edges > 0:
				loss = self.trainer.train_on_graph(self.graph)
				log.info("GNN training loss: %.4f", loss)
			self.save()
		self._push_event(self.handle_status())
		return n_committed

	def handle_ingest(self, text: str, source: str = "", descriptor: str = "") -> dict:
		"""Ingest synchronously. Returns stats dict."""
		n_committed = self._ingest_sync(text, source, descriptor)
		return {
			"committed": n_committed,
			"thoughts": self.graph.num_thoughts,
			"edges": self.graph.num_edges,
		}

	def handle_ingest_queued(self, text: str, source: str = "", descriptor: str = "") -> str:
		"""Try async queue, fallback to sync. Returns status string."""
		queued, pending = self.enqueue(text, source, descriptor)
		if queued:
			return f"queued ({pending} pending)"
		# fallback to sync
		self.handle_ingest(text, source, descriptor)
		return "ok"

	def handle_ask(self, query: str) -> tuple[str, list[dict]]:
		"""Fan-out query to all specialists, merge signals, generate answer."""
		query_emb = self.provider.embed_text(query)

		# Score against the default graph
		local_scored = self._score_specialist(query_emb, self.graph, self.model, self.strand)

		# Fan out to all registered specialists
		all_scored = [local_scored]
		for name, spec in self._specialists.items():
			try:
				all_scored.append(self._score_specialist(query_emb, spec.graph, spec.model, spec.strand))
			except Exception:
				log.warning("Specialist '%s' query failed", name, exc_info=True)

		# Q_DED: deduplicate across specialists
		scored = deduplicate(all_scored, self.cfg.top_k)
		if not scored:
			return "No relevant knowledge found.", []

		return answer(query, scored, self.provider)

	def _score_specialist(
		self,
		query_emb,
		graph: Graph,
		model: KnodMPNN,
		strand: StrandLayer,
	) -> list[tuple]:
		"""Run all three scoring signals + merge for one specialist."""
		if not graph.thoughts:
			return []

		cos = cosine_scores(query_emb, graph)
		try:
			gnn = gnn_scores(query_emb, graph, model, strand)
		except Exception:
			log.debug("GNN scoring failed", exc_info=True)
			gnn = {}
		edg = edge_scores(query_emb, graph, self.cfg)

		return merge(cos, gnn, edg, graph, self.cfg)

	def handle_status(self) -> str:
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

	def handle_set_purpose(self, purpose: str):
		self.graph.purpose = purpose
		self.save()

	def handle_descriptor_add(self, name: str, text: str):
		self.graph.descriptors[name] = text
		self.save()

	def handle_descriptor_remove(self, name: str) -> bool:
		removed = self.graph.descriptors.pop(name, None) is not None
		if removed:
			self.save()
		return removed

	# ---- pub/sub for TCP subscribers ----

	def subscribe(self, sock):
		with self._subs_lock:
			self._subs.append(sock)

	def unsubscribe(self, sock):
		with self._subs_lock:
			self._subs = [s for s in self._subs if s is not sock]

	def _push_event(self, msg: str):
		with self._subs_lock:
			dead = []
			for sock in self._subs:
				try:
					data = msg.encode()
					hdr = len(data).to_bytes(4, "big")
					sock.sendall(hdr + data)
				except Exception:
					dead.append(sock)
			for s in dead:
				self._subs.remove(s)

	# ---- specialist management ----

	def _load_specialists(self):
		"""Load all registered specialists into cache."""
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
					purpose=entry.get("purpose", ""),
					graph=graph,
					model=model,
					strand=strand,
				)
				log.info("Loaded specialist '%s'", name)
			except Exception:
				log.warning("Failed to load specialist '%s'", name, exc_info=True)

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
		for cluster_indices in clusters:
			cluster_thoughts = [self.graph.limbo[i] for i in cluster_indices]
			try:
				promote_cluster(
					cluster_thoughts,
					self._specialists,
					self.provider,
					self.cfg,
					self.registry,
					self.cfg.graph_path,
				)
				promoted_indices.update(cluster_indices)
			except Exception:
				log.exception("Cluster promotion failed")

		if promoted_indices:
			self.graph.limbo = [lt for i, lt in enumerate(self.graph.limbo) if i not in promoted_indices]
