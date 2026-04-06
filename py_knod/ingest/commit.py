"""Phase 4 · Commit — MCMC gate: accept thought + edges, or route to limbo.

Gate logic (matches FLOW.md):
  - Has valid links  → Accept (always)
  - No links, store immature (maturity == 0)  → Accept
  - No links, store mature  → MCMC gate  p = 0.05^maturity
    - Passes   → Accept
    - Rejected → Limbo
"""

import logging
import random

from ..specialist.graph import Graph, Thought, LimboThought
from .prepare import PreparedArticle

log = logging.getLogger(__name__)


def _accept(maturity: float) -> bool:
	"""MCMC acceptance probability: p = 0.05^maturity."""
	if maturity <= 0:
		return True
	p = 0.05**maturity
	return random.random() < p


def commit(article: PreparedArticle, graph: Graph) -> list[Thought]:
	"""Phase 4: Insert accepted thoughts + edges; route rejected unlinked thoughts to limbo."""
	committed = []

	for pt in article.thoughts:
		has_links = len(pt.links) > 0

		if not has_links and not _accept(graph.maturity):
			graph.limbo.append(
				LimboThought(
					text=pt.text,
					embedding=pt.embedding,
					source=pt.source,
				)
			)
			log.debug("Sent to limbo: %s…", pt.text[:60])
			continue

		thought = graph.add_thought(pt.text, pt.embedding, pt.source)

		for link, emb in zip(pt.links, pt.link_embeddings):
			target_id = pt.candidate_ids[link["index"]]
			if target_id in graph.thoughts:
				graph.add_edge(
					source_id=thought.id,
					target_id=target_id,
					weight=link["weight"],
					reasoning=link["reasoning"],
					embedding=emb,
				)

		committed.append(thought)

	log.info("Committed %d/%d thoughts", len(committed), len(article.thoughts))
	return committed
