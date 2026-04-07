"""Limbo promote — route a cluster to an existing specialist or spawn a new one.

Matches FLOW.md LIMBO subgraph:
  LB_NM  — LLM: name + describe the cluster
  LB_EP  — Embed cluster purpose
  LB_MAT — Existing specialist profile match ≥ specialist_match_threshold?
  LB_PRO — Promote to matching specialist
  LB_NEW — Spawn new specialist

After promotion or spawn, runs Phase 3 (link) + GNN training so the
specialist starts with proper edges and trained weights.
"""

import logging
from pathlib import Path

import numpy as np

from ..config import Config
from ..provider import Provider
from ..registry import store_path
from ..specialist.graph import Graph, LimboThought
from ..specialist.gnn import KnodMPNN, StrandLayer
from ..specialist.store import save_all, load_base_model
from ..specialist.types import Specialist
from ..util.math import cosine

log = logging.getLogger(__name__)


def bootstrap_thoughts(
	thought_ids: list[int],
	graph: Graph,
	model: KnodMPNN,
	strand: StrandLayer,
	provider: Provider,
	cfg: Config,
):
	"""Link a set of thoughts to their neighbors in the graph and train GNN.

	Called after spawning a new specialist or promoting thoughts to an existing one.
	Runs Phase 3 (link reasoning) on the given thoughts, adds edges, then trains.
	"""
	if len(thought_ids) < 2 or graph.num_thoughts < 2:
		return

	from ..ingest.prepare import PreparedThought, PreparedArticle
	from ..ingest.link import link

	prepared = []
	for tid in thought_ids:
		t = graph.thoughts.get(tid)
		if t is None:
			continue
		pt = PreparedThought(text=t.text, embedding=t.embedding, source=t.source)
		candidates = graph.find_thoughts(t.embedding, k=cfg.top_k, threshold=cfg.similarity_threshold)
		pt.candidate_ids = [c.id for c, _ in candidates if c.id != tid]
		pt.candidate_texts = [c.text for c, _ in candidates if c.id != tid]
		if pt.candidate_ids:
			prepared.append((tid, pt))

	if not prepared:
		return

	article = PreparedArticle(thoughts=[pt for _, pt in prepared], source="bootstrap")
	link(article, provider, cfg)

	edges_added = 0
	for tid, pt in prepared:
		for link_data, emb in zip(pt.links, pt.link_embeddings):
			idx = link_data["index"]
			if 0 <= idx < len(pt.candidate_ids):
				target_id = pt.candidate_ids[idx]
				if target_id in graph.thoughts:
					graph.add_edge(
						source_id=tid,
						target_id=target_id,
						weight=link_data["weight"],
						reasoning=link_data["reasoning"],
						embedding=emb,
					)
					edges_added += 1

	log.info("Bootstrap: added %d edges for %d thoughts", edges_added, len(prepared))

	if graph.num_edges > 0:
		from ..specialist.trainer import GNNTrainer

		# Load latest shared base weights before training (bidirectional sync)
		load_base_model(model)
		trainer = GNNTrainer(model, strand, cfg)
		loss = trainer.train_on_graph(graph)
		log.info("Bootstrap GNN training loss: %.4f", loss)


def promote_cluster(
	cluster: list[LimboThought],
	specialists: dict,  # dict[str, Specialist]
	provider: Provider,
	cfg: Config,
	registry,
	graph_base_path: str,
) -> str | None:
	"""Promote a limbo cluster: try existing specialist, else spawn new one.

	Returns the specialist name the cluster was promoted into (or None on error).
	After promotion, runs bootstrap (Phase 3 + GNN training) and saves.
	"""
	texts = [lt.text for lt in cluster]
	name, purpose = provider.suggest_store(texts)

	best_match = None
	best_sim = 0.0

	if specialists:
		purpose_emb = provider.embed_text(purpose)
		for sname, spec in specialists.items():
			if spec.graph.profile is not None:
				sim = cosine(spec.graph.profile, purpose_emb)
				if sim > best_sim:
					best_sim = sim
					best_match = sname

	if best_match and best_sim >= cfg.specialist_match_threshold:
		spec = specialists[best_match]
		new_ids = []
		for lt in cluster:
			t = spec.graph.add_thought(lt.text, lt.embedding, lt.source)
			if t:
				new_ids.append(t.id)
		# Bootstrap: link + train
		bootstrap_thoughts(new_ids, spec.graph, spec.model, spec.strand, provider, cfg)
		# Save specialist
		graph_path = registry.stores[best_match]["path"]
		base = Path(graph_path).with_suffix("")
		save_all(spec.graph, spec.model, spec.strand, base)
		log.info("Promoted %d limbo thoughts to specialist '%s'", len(cluster), best_match)
		return best_match
	else:
		return _spawn_specialist(name, purpose, cluster, specialists, provider, cfg, registry, graph_base_path)


def _spawn_specialist(
	name: str,
	purpose: str,
	cluster: list[LimboThought],
	specialists: dict,
	provider: Provider,
	cfg: Config,
	registry,
	graph_base_path: str,
) -> str:
	"""Create a new specialist graph from a limbo cluster, link thoughts, and train."""
	from ..specialist.graph import Graph as SpecGraph

	graph = SpecGraph(
		name=name,
		purpose=purpose,
		max_thoughts=cfg.max_thoughts,
		max_edges=cfg.max_edges,
		maturity_divisor=cfg.maturity_divisor,
	)
	model = KnodMPNN(cfg)
	strand = StrandLayer(cfg.hidden_dim)

	# Inherit global base weights so the specialist starts with global knowledge
	load_base_model(model)

	new_ids = []
	for lt in cluster:
		t = graph.add_thought(lt.text, lt.embedding, lt.source)
		if t:
			new_ids.append(t.id)

	# Bootstrap: link between cluster thoughts + train GNN
	bootstrap_thoughts(new_ids, graph, model, strand, provider, cfg)

	store_dir = Path(graph_base_path).with_suffix("").parent
	hashed_path = store_path(store_dir, name)
	base = hashed_path.with_suffix("")
	save_all(graph, model, strand, base)

	graph_path = str(hashed_path)
	registry.register(graph_path)

	specialists[name] = Specialist(
		name=name,
		purpose=purpose,
		graph=graph,
		model=model,
		strand=strand,
	)
	log.info("Spawned new specialist '%s' with %d thoughts, %d edges", name, len(cluster), graph.num_edges)
	return name
