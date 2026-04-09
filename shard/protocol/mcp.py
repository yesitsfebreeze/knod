"""MCP server — exposes shard graph as Model Context Protocol tools + resources."""

from __future__ import annotations

import json
import logging

from mcp.server.fastmcp import FastMCP

from ..handler import Handler

log = logging.getLogger(__name__)


def _mcp(handler: Handler, host: str = "127.0.0.1", port: int = 8766) -> FastMCP:
	"""Build a FastMCP server wired to the shared Handler."""
	mcp = FastMCP("shard", json_response=True, host=host, port=port)

	# ---- Tools ----

	@mcp.tool()
	def ask(query: str) -> str:
		"""Ask a question against the knowledge graph. Returns an answer grounded in stored thoughts."""
		log.info("ask: %.80s", query)
		answer, sources = handler.ask(query)
		log.debug("ask result: %.200s", answer)
		return json.dumps({"answer": answer, "sources": sources})

	@mcp.tool()
	def ingest(text: str, source: str = "", descriptor: str = "", shard: str = "") -> str:
		"""Ingest text into the knowledge graph. Use shard= to target a specific specialist shard, or leave empty to auto-route to the best matching shard."""
		log.info("ingest: source=%s shard=%s len=%d", source or "(none)", shard or "auto", len(text))
		log.debug("ingest content: %.200s", text)
		if shard:
			try:
				count = handler.ingest_into_shard(shard, text, source=source, descriptor=descriptor)
				return json.dumps({"queued": False, "shard": shard, "committed": count})
			except KeyError:
				return json.dumps({"error": f"Shard '{shard}' not loaded. Use list_shards to see available shards or create_shard to create one."})
		result = handler.ingest(text, source=source, descriptor=descriptor)
		return result

	@mcp.tool()
	def set_purpose(purpose: str) -> str:
		"""Set the purpose / focus of the knowledge graph."""
		log.info("set_purpose: %.80s", purpose)
		handler.set_purpose(purpose)
		return f"Purpose set to: {purpose}"

	@mcp.tool()
	def add_descriptor(name: str, description: str) -> str:
		"""Add a descriptor (named context hint) to guide future ingestion."""
		log.info("descriptor added: %s", name)
		log.debug("descriptor content: %.200s", description)
		handler.add_descriptor(name, description)
		return f"Descriptor '{name}' added."

	@mcp.tool()
	def remove_descriptor(name: str) -> str:
		"""Remove a descriptor by name."""
		log.info("descriptor removed: %s", name)
		ok = handler.remove_descriptor(name)
		return f"Descriptor '{name}' removed." if ok else f"Descriptor '{name}' not found."

	@mcp.tool()
	def find_thoughts(query: str, k: int = 5) -> str:
		"""Search for thoughts semantically similar to the query. Returns top-k matches without generating an LLM answer."""
		log.info("find_thoughts: k=%d %.80s", k, query)
		results = handler.find_thoughts_by_query(query, k=k)
		return json.dumps(results)

	@mcp.tool()
	def explore_thought(thought_id: int) -> str:
		"""Explore a single thought: see its text, edges, neighbors, and reasoning. Returns the thought with all its connections in the graph."""
		log.info("explore_thought: id=%d", thought_id)
		result = handler.explore_thought(thought_id)
		if result is None:
			return json.dumps({"error": f"Thought {thought_id} not found"})
		return json.dumps(result)

	@mcp.tool()
	def traverse(start_id: int, depth: int = 2, max_nodes: int = 50) -> str:
		"""Walk the graph from a starting thought via BFS. Returns the local subgraph: nodes, edges, and reasoning chains up to the given depth."""
		log.info("traverse: start=%d depth=%d", start_id, depth)
		result = handler.traverse(start_id, depth=depth, max_nodes=max_nodes)
		if result is None:
			return json.dumps({"error": f"Thought {start_id} not found"})
		return json.dumps(result)

	@mcp.tool()
	def graph_stats() -> str:
		"""Get aggregate statistics for the knowledge graph: thought/edge counts, maturity, limbo size, Shard summaries, and edge quality metrics."""
		log.info("graph_stats")
		return json.dumps(handler.graph_stats())

	@mcp.tool()
	def list_shards() -> str:
		"""List all loaded shards with their purpose, thought/edge counts, descriptors, and cluster membership."""
		log.info("list_shards")
		return json.dumps(handler.list_shards())

	@mcp.tool()
	def ingest_sync(text: str, source: str = "", descriptor: str = "", shard: str = "") -> str:
		"""Ingest text synchronously and return committed thoughts, rejection count, and dedup count. Use shard= to target a specific specialist shard, or leave empty to auto-route to the best match."""
		target = shard or handler.route_shard(text)
		log.info("ingest_sync: source=%s shard=%s len=%d", source or "(none)", target or "limbo", len(text))
		log.debug("ingest_sync content: %.200s", text)
		if target:
			try:
				count = handler.ingest_into_shard(target, text, source=source, descriptor=descriptor)
				return json.dumps({"shard": target, "committed": count})
			except KeyError:
				return json.dumps({"error": f"Shard '{target}' not loaded. Use list_shards or create_shard first."})
		result = handler.ingest_sync(text, source=source, descriptor=descriptor)
		log.info("ingest_sync done: %d thoughts, %d edges", result.get("thoughts", 0), result.get("edges", 0))
		return json.dumps(result)

	@mcp.tool()
	def relink() -> str:
		"""Scan all existing thoughts and create missing edges between similar pairs. Use this after bulk ingestion to ensure the graph is fully connected."""
		log.info("relink: triggered")
		result = handler.relink()
		log.info("relink done: %s", result)
		return json.dumps(result)

	@mcp.tool()
	def link(source_id: int, target_id: int, reasoning: str = "", confidence: float = 0.0, shard: str = "") -> str:
		"""Link two thoughts. The LLM scores the relationship; confidence (0.0–1.0) is an inverse multiplier — at 0.0 the LLM weight is used as-is, at 1.0 the link is forced to maximum weight. Optionally supply your own reasoning text. Use shard= to target a specific shard."""
		log.info("link: %d → %d%s", source_id, target_id, f" shard={shard}" if shard else "")
		log.debug("link reasoning: %.200s", reasoning)
		result = handler.link_thoughts(source_id, target_id, reasoning=reasoning, confidence=confidence, shard_name=shard or None)
		return json.dumps(result)

	@mcp.tool()
	def forget(thought_id: int, shard: str = "") -> str:
		"""Remove a thought and all its edges from the graph. Use shard= to target a specific shard, or leave empty for the global graph."""
		log.info("forget: thought %d%s", thought_id, f" shard={shard}" if shard else "")
		result = handler.forget(thought_id, shard_name=shard or None)
		return json.dumps(result)

	@mcp.tool()
	def create_shard(name: str, purpose: str) -> str:
		"""Create a new specialist shard with a given name and purpose. The shard is immediately available for ingest without restart."""
		log.info("create_shard: name=%s", name)
		from pathlib import Path
		location = str(Path(handler.cfg.graph_path).parent)
		try:
			graph_path = handler.create_shard(name, purpose, location)
			return json.dumps({"ok": True, "name": name, "path": graph_path})
		except Exception as e:
			return json.dumps({"ok": False, "error": str(e)})

	@mcp.tool()
	def register_shard(path: str) -> str:
		"""Load and register an existing .shard file at runtime. Makes it immediately available for ingest and queries without restart."""
		log.info("register_shard: path=%s", path)
		result = handler.register_shard_runtime(path)
		return json.dumps(result)

	@mcp.tool()
	def rebootstrap_shards(only_empty: bool = True) -> str:
		"""Re-run link reasoning and GNN training on loaded shards. Set only_empty=false to reprocess all shards, not just those with zero edges."""
		log.info("rebootstrap_shards: only_empty=%s", only_empty)
		result = handler.rebootstrap_shards(only_empty=only_empty)
		return json.dumps(result)

	# ---- Resources ----

	@mcp.resource("shard://status")
	def status() -> str:
		"""Current graph status: thought count, edge count, purpose."""
		return handler.status()

	@mcp.resource("shard://graph")
	def graph_info() -> str:
		"""Graph metadata as JSON."""
		return json.dumps(handler.graph_info)

	@mcp.resource("shard://thoughts")
	def list_thoughts() -> str:
		"""List all thoughts as JSON array (id, text snippet, source)."""
		return json.dumps(handler.all_thoughts)

	@mcp.resource("shard://descriptors")
	def list_descriptors() -> str:
		"""List all descriptors as JSON."""
		return json.dumps(handler.graph_info["descriptors"])

	@mcp.resource("shard://shards")
	def shards_resource() -> str:
		"""List all loaded shards as JSON."""
		return json.dumps(handler.list_shards())

	@mcp.resource("shard://stats")
	def stats_resource() -> str:
		"""Aggregate graph statistics as JSON."""
		return json.dumps(handler.graph_stats())

	# ---- Prompts ----

	@mcp.prompt()
	def research(topic: str) -> str:
		"""Generate a prompt to research a topic using the knowledge graph."""
		return (
			f"Use the shard knowledge graph to answer questions about: {topic}\n\n"
			"1. First use find_thoughts to see what's already known\n"
			"2. Then use ask to get a synthesized answer\n"
			"3. If knowledge is lacking, use ingest to add relevant text"
		)

	return mcp
