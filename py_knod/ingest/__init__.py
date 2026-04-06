"""INGESTION pipeline — orchestrates the four phases:

Phase 1 · Prepare   — decompose + embed
Phase 2 · Snapshot  — cosine candidate search
Phase 3 · Link      — LLM weight + reasoning (parallel)
Phase 4 · Commit    — MCMC gate → accept / limbo
"""

import logging

from ..config import Config
from ..provider import Provider
from ..specialist.graph import Graph, Thought
from .prepare import prepare
from .snapshot import snapshot
from .dedup import dedup
from .link import link
from .commit import commit

log = logging.getLogger(__name__)


class Ingester:
	"""Runs the full 4-phase ingest pipeline for a single specialist graph."""

	def __init__(self, graph: Graph, provider: Provider, cfg: Config):
		self.graph = graph
		self.provider = provider
		self.cfg = cfg

	def ingest(self, text: str, source: str = "", descriptor: str = "") -> list[Thought]:
		"""Full ingest pipeline. Returns list of committed thoughts."""
		# Phase 1: Prepare
		article = prepare(text, source, descriptor, self.graph, self.provider, self.cfg)
		if not article.thoughts:
			log.warning("Decomposition produced no thoughts")
			return []

		# Phase 2: Snapshot
		snapshot(article, self.graph, self.cfg)

		# Phase 2.5: Dedup — merge near-duplicates into existing thoughts
		dedup(article, self.graph, self.cfg)
		if not article.thoughts:
			log.info("All thoughts merged into existing — nothing new to commit")
			return []

		# Phase 3: Link
		link(article, self.provider, self.cfg)

		# Phase 4: Commit
		return commit(article, self.graph)
