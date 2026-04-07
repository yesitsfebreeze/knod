from ..config import Config
from ..specialist.graph import Graph
from ..util.math import cosine
from .prepare import PreparedArticle

def snapshot(article: PreparedArticle, graph: Graph, cfg: Config):
	for i, pt in enumerate(article.thoughts):
		neighbors = graph.find_thoughts(pt.embedding, k=cfg.top_k, threshold=cfg.similarity_threshold)
		pt.candidate_ids = [t.id for t, _ in neighbors]
		pt.candidate_texts = [t.text for t, _ in neighbors]

		# Include sibling thoughts from same article as candidates.
		# Use negative temp IDs (-(j+1)) resolved to real IDs in commit.
		for j, sibling in enumerate(article.thoughts):
			if i == j:
				continue
			sim = cosine(pt.embedding, sibling.embedding)
			if sim >= cfg.similarity_threshold:
				pt.candidate_ids.append(-(j + 1))
				pt.candidate_texts.append(sibling.text)
