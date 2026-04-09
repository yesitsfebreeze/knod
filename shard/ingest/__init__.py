import logging

from ..config import Config
from ..provider import Provider
from ..shard.graph import Graph
from ..shard.types import IngestResult
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
		container = self.graph.name or "global"
		log.info("Ingest → container=%r  source=%r  input=%d chars", container, source or "", len(text))

		article = prepare(text, source, descriptor, self.graph, self.provider, self.cfg)
		if not article.thoughts:
			log.warning("Ingest → container=%r: decomposition produced no thoughts", container)
			return IngestResult()

		snapshot(article, self.graph, self.cfg)

		deduplicated = dedup(article, self.graph, self.cfg)
		if not article.thoughts:
			log.info("Ingest → container=%r: all %d thoughts merged into existing — nothing new to commit", container, deduplicated)
			return IngestResult(deduplicated=deduplicated)

		link(article, self.provider, self.cfg)
		result = commit(
			article,
			self.graph,
			deduplicated=deduplicated,
			linked_base=self.cfg.mcmc_linked_base,
			unlinked_base=self.cfg.mcmc_unlinked_base,
		)
		log.info(
			"Ingest done → container=%r  committed=%d  deduplicated=%d  rejected=%d  "
			"graph_total thoughts=%d edges=%d",
			container,
			len(result.committed),
			result.deduplicated,
			result.rejected,
			self.graph.num_thoughts,
			self.graph.num_edges,
		)
		return result
