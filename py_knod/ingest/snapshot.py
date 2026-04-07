from ..config import Config
from ..specialist.graph import Graph
from .prepare import PreparedArticle

def snapshot(article: PreparedArticle, graph: Graph, cfg: Config):
	for pt in article.thoughts:
		neighbors = graph.find_thoughts(pt.embedding, k=cfg.top_k, threshold=cfg.similarity_threshold)
		pt.candidate_ids = [t.id for t, _ in neighbors]
		pt.candidate_texts = [t.text for t, _ in neighbors]
