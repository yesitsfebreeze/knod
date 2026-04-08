"""Limbo promote — route a cluster to an existing shard or spawn a new one.

Matches FLOW.md LIMBO subgraph:
  LB_NM  — LLM: name + describe the cluster
  LB_EP  — Embed cluster purpose
  LB_MAT — Existing shard profile match ≥ shard_match_threshold?
  LB_PRO — Promote to matching shard
  LB_NEW — Spawn new shard

After promotion or spawn, runs Phase 3 (link) + GNN training so the
shard starts with proper edges and trained weights.
"""

import logging
from pathlib import Path

import numpy as np

from ..config import Config
from ..provider import Provider
from ..registry import Registry, store_path
from ..shard.graph import Graph, LimboThought
from ..shard.gnn import ShardMPNN, ShardLayer
from ..shard.store import save_all, load_base_model
from ..shard.types import Shard
from ..util.math import cosine

log = logging.getLogger(__name__)


def bootstrap_thoughts(
	thought_ids: list[int],
	graph: Graph,
	model: ShardMPNN,
	shard: ShardLayer,
	provider: Provider,
	cfg: Config,
):
	"""Link a set of thoughts to their neighbors in the graph and train GNN.

	Called after spawning a new shard or promoting thoughts to an existing one.
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
		from ..shard.trainer import GNNTrainer

		# Load latest shared base weights before training (bidirectional sync)
		load_base_model(model)
		trainer = GNNTrainer(model, shard, cfg)
		loss = trainer.train_on_graph(graph)
		log.info("Bootstrap GNN training loss: %.4f", loss)


def promote_cluster(
	cluster: list[LimboThought],
	shards: dict[str, Shard],
	provider: Provider,
	cfg: Config,
	registry: Registry,
	graph_base_path: str,
) -> str | None:
	"""Promote a limbo cluster: try existing shard, else spawn new one.

	Returns the shard name the cluster was promoted into (or None on error).
	After promotion, runs bootstrap (Phase 3 + GNN training) and saves.
	"""
	texts = [lt.text for lt in cluster]
	name, purpose = provider.suggest_store(texts)

	best_match = None
	best_sim = 0.0

	if shards:
		purpose_emb = provider.embed_text(purpose)
		for sname, shard in shards.items():
			if shard.graph.profile is not None:
				sim = cosine(shard.graph.profile, purpose_emb)
				if sim > best_sim:
					best_sim = sim
					best_match = sname

	if best_match and best_sim >= cfg.shard_match_threshold:
		shard = shards[best_match]
		new_ids = []
		for lt in cluster:
			t = shard.graph.add_thought(lt.text, lt.embedding, lt.source)
			if t:
				new_ids.append(t.id)
		# Bootstrap: link + train
		bootstrap_thoughts(new_ids, shard.graph, shard.model, shard.shard, provider, cfg)
		# Refine existing edges in the target Shard (re-evaluate based on new content)
		shard.graph.refine_edges(
			boost=cfg.refinement_boost,
			dampen=cfg.refinement_dampen,
			min_traversals=1,
		)
		# Save Shard
		graph_path = registry.stores[best_match]["path"]
		base = Path(graph_path).with_suffix("")
		save_all(shard.graph, shard.model, shard.shard, base)
		log.info(
			"Promoted %d limbo thoughts to Shard '%s' (refined %d edges)", len(cluster), best_match, shard.graph.num_edges
		)
		return best_match
	else:
		return _spawn_shard(name, purpose, cluster, shards, provider, cfg, registry, graph_base_path)


def _spawn_shard(
	name: str,
	purpose: str,
	cluster: list[LimboThought],
	shards: dict[str, Shard],
	provider: Provider,
	cfg: Config,
	registry: Registry,
	graph_base_path: str,
) -> str:
	"""Create a new Shard graph from a limbo cluster, link thoughts, and train."""
	from ..shard.graph import Graph as ShardGraph

	graph = ShardGraph(
		name=name,
		purpose=purpose,
		max_thoughts=cfg.max_thoughts,
		max_edges=cfg.max_edges,
		maturity_divisor=cfg.maturity_divisor,
	)
	model = ShardMPNN(cfg)
	shard = ShardLayer(cfg.hidden_dim)

	# Inherit global base weights so the Shard starts with global knowledge
	load_base_model(model)

	new_ids = []
	for lt in cluster:
		t = graph.add_thought(lt.text, lt.embedding, lt.source)
		if t:
			new_ids.append(t.id)

	# Bootstrap: link between cluster thoughts + train GNN
	bootstrap_thoughts(new_ids, graph, model, shard, provider, cfg)

	store_dir = Path(graph_base_path).with_suffix("").parent
	hashed_path = store_path(store_dir, name)
	base = hashed_path.with_suffix("")
	save_all(graph, model, shard, base)

	graph_path = str(hashed_path)
	registry.register(graph_path)

	shards[name] = shard(
		name=name,
		purpose=purpose,
		graph=graph,
		model=model,
		shard=shard,
	)
	log.info("Spawned new Shard '%s' with %d thoughts, %d edges", name, len(cluster), graph.num_edges)
	return name
