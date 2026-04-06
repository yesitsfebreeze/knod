"""Phase 2 · Snapshot — cosine search to find candidate neighbor thoughts per thought."""

from ..config import Config
from ..specialist.graph import Graph
from .prepare import PreparedArticle


def snapshot(article: PreparedArticle, graph: Graph, cfg: Config):
	"""Phase 2: For each thought find candidate neighbors via cosine search on live graph."""
	for pt in article.thoughts:
		neighbors = graph.find_thoughts(pt.embedding, k=cfg.top_k, threshold=cfg.similarity_threshold)
		pt.candidate_ids = [t.id for t, _ in neighbors]
		pt.candidate_texts = [t.text for t, _ in neighbors]
