import logging
import random

from ..strand.graph import Graph, Thought, LimboThought
from ..strand.types import IngestResult
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
	# Map article index → real thought ID for intra-batch edge resolution
	index_to_real_id: dict[int, int] = {}
	# Deferred intra-batch edges (resolved after all thoughts committed)
	deferred_edges: list[tuple[int, dict, object]] = []  # (article_idx, link, embedding)

	for idx, pt in enumerate(article.thoughts):
		has_links = len(pt.links) > 0
		base = 0.5 if has_links else 0.05

		if not _accept(graph.maturity, base=base):
			graph.limbo.append(
				LimboThought(
					text=pt.text,
					embedding=pt.embedding,
					source=pt.source,
				)
			)
			rejected += 1
			log.debug("Sent to limbo%s: %s…", " (linked)" if has_links else "", pt.text[:60])
			continue

		thought = graph.add_thought(pt.text, pt.embedding, pt.source)
		index_to_real_id[idx] = thought.id

		for link, emb in zip(pt.links, pt.link_embeddings):
			target_id = pt.candidate_ids[link["index"]]
			if target_id < 0:
				# Intra-batch sibling — defer until all thoughts committed
				deferred_edges.append((idx, link, emb, pt.candidate_ids[link["index"]]))
			elif target_id in graph.thoughts:
				graph.add_edge(
					source_id=thought.id,
					target_id=target_id,
					weight=link["weight"],
					reasoning=link["reasoning"],
					embedding=emb,
				)

		committed.append(thought)

	# Resolve deferred intra-batch edges
	for src_idx, link, emb, neg_id in deferred_edges:
		src_real = index_to_real_id.get(src_idx)
		tgt_article_idx = -(neg_id + 1)
		tgt_real = index_to_real_id.get(tgt_article_idx)
		if src_real is not None and tgt_real is not None:
			graph.add_edge(
				source_id=src_real,
				target_id=tgt_real,
				weight=link["weight"],
				reasoning=link["reasoning"],
				embedding=emb,
			)

	log.info("Committed %d/%d thoughts", len(committed), len(article.thoughts))
	return IngestResult(committed=committed, rejected=rejected, deduplicated=deduplicated)
