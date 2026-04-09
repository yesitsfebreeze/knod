import logging
import random

from ..shard.graph import Graph, Thought, LimboThought
from ..shard.types import IngestResult
from .prepare import PreparedArticle

log = logging.getLogger(__name__)


def _accept(maturity: float, base: float, strictness: float = 1.0) -> bool:
	if strictness <= 0 or maturity <= 0:
		return True
	p = base ** (maturity * strictness)
	return random.random() < p


def commit(
	article: PreparedArticle,
	graph: Graph,
	deduplicated: int = 0,
	linked_base: float = 0.5,
	unlinked_base: float = 0.3,
	strictness: float = 1.0,
) -> IngestResult:
	committed = []
	rejected = 0
	# Map article index → real thought ID for intra-batch edge resolution
	index_to_real_id: dict[int, int] = {}
	# Deferred intra-batch edges (resolved after all thoughts committed)
	deferred_edges: list[tuple[int, dict, object]] = []  # (article_idx, link, embedding)

	for idx, pt in enumerate(article.thoughts):
		has_links = len(pt.links) > 0
		base = linked_base if has_links else unlinked_base

		if not _accept(graph.maturity, base=base, strictness=strictness):
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
		log.debug("  committed thought[%d] id=%d: %s", idx, thought.id, pt.text[:80])

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

	edges_added = sum(len(t.links) for t in article.thoughts)
	log.info(
		"Committed %d/%d thoughts, %d edges added, %d sent to limbo",
		len(committed), len(committed) + rejected, edges_added, rejected,
	)
	return IngestResult(committed=committed, rejected=rejected, deduplicated=deduplicated)
