"""Limbo promote — route a cluster to an existing specialist or spawn a new one.

Matches FLOW.md LIMBO subgraph:
  LB_NM  — LLM: name + describe the cluster
  LB_EP  — Embed cluster purpose
  LB_MAT — Existing specialist profile match ≥ specialist_match_threshold?
  LB_PRO — Promote to matching specialist
  LB_NEW — Spawn new specialist
"""

import logging
from pathlib import Path

import numpy as np

from ..config import Config
from ..provider import Provider
from ..specialist.graph import Graph, LimboThought
from ..specialist.gnn import KnodMPNN, StrandLayer
from ..specialist.store import save_all

log = logging.getLogger(__name__)


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
	"""
	texts = [lt.text for lt in cluster]
	name, purpose = provider.suggest_store(texts)

	best_match = None
	best_sim = 0.0

	if specialists:
		purpose_emb = provider.embed_text(purpose)
		for sname, spec in specialists.items():
			if spec.graph.profile is not None:
				p_norm = spec.graph.profile / (np.linalg.norm(spec.graph.profile) + 1e-10)
				q_norm = purpose_emb / (np.linalg.norm(purpose_emb) + 1e-10)
				sim = float(np.dot(p_norm, q_norm))
				if sim > best_sim:
					best_sim = sim
					best_match = sname

	if best_match and best_sim >= cfg.specialist_match_threshold:
		spec = specialists[best_match]
		for lt in cluster:
			spec.graph.add_thought(lt.text, lt.embedding, lt.source)
		log.info("Promoted %d limbo thoughts to specialist '%s'", len(cluster), best_match)
		return best_match
	else:
		return _spawn_specialist(name, purpose, cluster, specialists, cfg, registry, graph_base_path)


def _spawn_specialist(
	name: str,
	purpose: str,
	cluster: list[LimboThought],
	specialists: dict,
	cfg: Config,
	registry,
	graph_base_path: str,
) -> str:
	"""Create a new specialist graph from a limbo cluster."""
	from ..specialist.graph import Graph as SpecGraph

	graph = SpecGraph(
		purpose=purpose,
		max_thoughts=cfg.max_thoughts,
		max_edges=cfg.max_edges,
	)
	model = KnodMPNN(cfg)
	strand = StrandLayer(cfg.hidden_dim)

	for lt in cluster:
		graph.add_thought(lt.text, lt.embedding, lt.source)

	safe_name = name.replace(" ", "_").lower()
	base = Path(graph_base_path).with_suffix("").parent / safe_name
	save_all(graph, model, strand, base)

	graph_path = str(base.with_suffix(".graph"))
	registry.register(name, graph_path, purpose)

	from ..handler import Specialist

	specialists[name] = Specialist(
		name=name,
		purpose=purpose,
		graph=graph,
		model=model,
		strand=strand,
	)
	log.info("Spawned new specialist '%s' with %d thoughts", name, len(cluster))
	return name
