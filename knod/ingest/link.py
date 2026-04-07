import logging
from concurrent.futures import ThreadPoolExecutor

from ..config import Config
from ..provider import Provider
from .prepare import PreparedArticle, PreparedThought

log = logging.getLogger(__name__)


def _link_one(pt: PreparedThought, provider: Provider, cfg: Config):
	if not pt.candidate_texts:
		return
	results = provider.batch_link_reason(pt.text, pt.candidate_texts)
	valid_links = [r for r in results if r["weight"] >= cfg.min_link_weight and 0 <= r["index"] < len(pt.candidate_ids)]
	if valid_links:
		reasoning_texts = [l["reasoning"] for l in valid_links]
		embeddings = provider.embed_texts(reasoning_texts)
		pt.links = valid_links
		pt.link_embeddings = embeddings


def link(article: PreparedArticle, provider: Provider, cfg: Config):
	with ThreadPoolExecutor(max_workers=4) as pool:
		list(pool.map(lambda pt: _link_one(pt, provider, cfg), article.thoughts))
