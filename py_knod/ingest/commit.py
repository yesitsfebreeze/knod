import logging
import random

from ..specialist.graph import Graph, Thought, LimboThought
from ..specialist.types import IngestResult
from .prepare import PreparedArticle

log = logging.getLogger(__name__)


def _accept(maturity: float, base: float = 0.05) -> bool:
	if maturity <= 0:
		return True
	p = base**maturity
	return random.random() < p


def commit(article: PreparedArticle, graph: Graph, deduplicated: int = 0) -> IngestResult:
	committed = []
	rejected = 0

	for pt in article.thoughts:
		has_links = len(pt.links) > 0

		if not has_links and not _accept(graph.maturity, base=0.05):
			graph.limbo.append(
				LimboThought(
					text=pt.text,
					embedding=pt.embedding,
					source=pt.source,
				)
			)
			rejected += 1
			log.debug("Sent to limbo: %s…", pt.text[:60])
			continue
		if has_links and not _accept(graph.maturity, base=0.5):
			graph.limbo.append(
				LimboThought(
					text=pt.text,
					embedding=pt.embedding,
					source=pt.source,
				)
			)
			rejected += 1
			log.debug("Sent to limbo (linked): %s…", pt.text[:60])
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
	return IngestResult(committed=committed, rejected=rejected, deduplicated=deduplicated)
