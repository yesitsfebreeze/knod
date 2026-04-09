import logging
from dataclasses import dataclass, field

import numpy as np

from ..config import Config
from ..provider import Provider
from ..shard.graph import Graph

log = logging.getLogger(__name__)


@dataclass
class PreparedThought:
	text: str
	embedding: np.ndarray
	source: str
	candidate_ids: list[int] = field(default_factory=list)
	candidate_texts: list[str] = field(default_factory=list)
	links: list[dict] = field(default_factory=list)
	link_embeddings: list[np.ndarray] = field(default_factory=list)


@dataclass
class PreparedArticle:
	thoughts: list[PreparedThought]
	source: str


def prepare(text: str, source: str, descriptor: str, graph: Graph, provider: Provider, cfg: Config) -> PreparedArticle:
	descriptors = graph.descriptors if not descriptor else {descriptor: graph.descriptors.get(descriptor, descriptor)}

	decomposed = provider.decompose_text(text, graph.purpose, descriptors or None)
	decomposed = [d for d in decomposed if d.get("text", "").strip()]
	log.info("Decomposed into %d thoughts", len(decomposed))
	for i, d in enumerate(decomposed):
		log.debug("  thought[%d]: %s", i, d["statement"])

	# Embed over the full reconstructed text (statement + context) for richer
	# semantic signal; store the same text as the thought body.
	full_texts = [d["text"] for d in decomposed]
	embeddings = provider.embed_texts(full_texts)

	thoughts = [
		PreparedThought(text=t, embedding=emb, source=source)
		for t, emb in zip(full_texts, embeddings)
	]
	return PreparedArticle(thoughts=thoughts, source=source)
