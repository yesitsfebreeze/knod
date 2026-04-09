import logging
from concurrent.futures import ThreadPoolExecutor

from ..config import Config
from ..provider import Provider
from .prepare import PreparedArticle, PreparedThought

log = logging.getLogger(__name__)


def _link_one(pt: PreparedThought, provider: Provider, cfg: Config, document_context: str = ""):
	if not pt.candidate_texts:
		return
	results = provider.batch_link_reason(pt.text, pt.candidate_texts, document_context)
	valid_links = [r for r in results if r["weight"] >= cfg.min_link_weight and 0 <= r["index"] < len(pt.candidate_ids)]
	if valid_links:
		reasoning_texts = [l["reasoning"] for l in valid_links]
		embeddings = provider.embed_texts(reasoning_texts)
		pt.links = valid_links
		pt.link_embeddings = embeddings
		for lk in valid_links:
			log.debug(
				"  edge: %r → candidate[%d] (weight=%.3f) — %s",
				pt.text[:60],
				lk["index"],
				lk["weight"],
				lk.get("reasoning", "")[:80],
			)


def link(article: PreparedArticle, provider: Provider, cfg: Config, document_context: str = ""):
	with ThreadPoolExecutor(max_workers=4) as pool:
		list(pool.map(lambda pt: _link_one(pt, provider, cfg, document_context), article.thoughts))

	total_links = sum(len(pt.links) for pt in article.thoughts)
	log.info("Linking: %d edges across %d thoughts", total_links, len(article.thoughts))
