import logging

from ..config import Config
from ..provider import Provider
from ..Shard.graph import Graph
from ..Shard.types import IngestResult
from .prepare import prepare
from .snapshot import snapshot
from .dedup import dedup
from .link import link
from .commit import commit

log = logging.getLogger(__name__)


class Ingester:
	def __init__(self, graph: Graph, provider: Provider, cfg: Config):
		self.graph = graph
		self.provider = provider
		self.cfg = cfg

	def ingest(self, text: str, source: str = "", descriptor: str = "") -> IngestResult:
		article = prepare(text, source, descriptor, self.graph, self.provider, self.cfg)
		if not article.thoughts:
			log.warning("Decomposition produced no thoughts")
			return IngestResult()

		snapshot(article, self.graph, self.cfg)

		deduplicated = dedup(article, self.graph, self.cfg)
		if not article.thoughts:
			log.info("All thoughts merged into existing — nothing new to commit")
			return IngestResult(deduplicated=deduplicated)

		link(article, self.provider, self.cfg)
		return commit(article, self.graph, deduplicated=deduplicated)
