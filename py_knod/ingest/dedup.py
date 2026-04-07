import logging

from ..config import Config
from ..specialist.graph import Graph
from ..util.math import normalize
from .prepare import PreparedArticle

log = logging.getLogger(__name__)


def dedup(article: PreparedArticle, graph: Graph, cfg: Config) -> int:
	if cfg.dedup_threshold <= 0 or not graph.thoughts:
		return 0

	merged = 0
	keep = []

	for pt in article.thoughts:
		results = graph.find_thoughts(pt.embedding, k=1, threshold=0.0)
		best_thought, best_sim = results[0] if results else (None, 0.0)

		if best_sim >= cfg.dedup_threshold and best_thought is not None:
			best_thought.embedding = normalize((best_thought.embedding + pt.embedding) / 2.0)
			merged += 1
			log.debug("Merged (%.3f): %s", best_sim, pt.text[:60])
		else:
			keep.append(pt)

	article.thoughts = keep

	if merged:
		log.info("Dedup: merged %d/%d thoughts (threshold=%.2f)", merged, merged + len(keep), cfg.dedup_threshold)
	return merged
