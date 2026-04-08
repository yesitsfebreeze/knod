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
		answer, sources = handler.ask(query)
		return json.dumps({"answer": answer, "sources": sources})

	@mcp.tool()
	def ingest(text: str, source: str = "", descriptor: str = "") -> str:
		"""Ingest text into the knowledge graph. The text is decomposed into atomic thoughts, linked, and stored."""
		result = handler.ingest(text, source=source, descriptor=descriptor)
		return result

	@mcp.tool()
	def set_purpose(purpose: str) -> str:
		"""Set the purpose / focus of the knowledge graph."""
		handler.set_purpose(purpose)
		return f"Purpose set to: {purpose}"

	@mcp.tool()
	def add_descriptor(name: str, description: str) -> str:
		"""Add a descriptor (named context hint) to guide future ingestion."""
		handler.add_descriptor(name, description)
		return f"Descriptor '{name}' added."

	@mcp.tool()
	def remove_descriptor(name: str) -> str:
		"""Remove a descriptor by name."""
		ok = handler.remove_descriptor(name)
		return f"Descriptor '{name}' removed." if ok else f"Descriptor '{name}' not found."

	@mcp.tool()
	def find_thoughts(query: str, k: int = 5) -> str:
		"""Search for thoughts semantically similar to the query. Returns top-k matches without generating an LLM answer."""
		results = handler.find_thoughts_by_query(query, k=k)
		return json.dumps(results)

	@mcp.tool()
	def explore_thought(thought_id: int) -> str:
		"""Explore a single thought: see its text, edges, neighbors, and reasoning. Returns the thought with all its connections in the graph."""
		result = handler.explore_thought(thought_id)
		if result is None:
			return json.dumps({"error": f"Thought {thought_id} not found"})
		return json.dumps(result)

	@mcp.tool()
	def traverse(start_id: int, depth: int = 2, max_nodes: int = 50) -> str:
		"""Walk the graph from a starting thought via BFS. Returns the local subgraph: nodes, edges, and reasoning chains up to the given depth."""
		result = handler.traverse(start_id, depth=depth, max_nodes=max_nodes)
		if result is None:
			return json.dumps({"error": f"Thought {start_id} not found"})
		return json.dumps(result)

	@mcp.tool()
	def graph_stats() -> str:
		"""Get aggregate statistics for the knowledge graph: thought/edge counts, maturity, limbo size, Shard summaries, and edge quality metrics."""
		return json.dumps(handler.graph_stats())

	@mcp.tool()
	def list_shards() -> str:
		"""List all loaded shards with their purpose, thought/edge counts, descriptors, and cluster membership."""
		return json.dumps(handler.list_shards())

	@mcp.tool()
	def ingest_sync(text: str, source: str = "", descriptor: str = "") -> str:
		"""Ingest text synchronously and return what was created: committed thoughts with IDs, rejection count, and dedup count. Use this instead of ingest when you need to verify what was stored."""
		result = handler.ingest_sync(text, source=source, descriptor=descriptor)
		return json.dumps(result)

	@mcp.tool()
	def relink() -> str:
		"""Scan all existing thoughts and create missing edges between similar pairs. Use this after bulk ingestion to ensure the graph is fully connected."""
		result = handler.relink()
		return json.dumps(result)

	@mcp.tool()
	def link(source_id: int, target_id: int, reasoning: str = "", confidence: float = 0.0, shard: str = "") -> str:
		"""Link two thoughts. The LLM scores the relationship; confidence (0.0–1.0) is an inverse multiplier — at 0.0 the LLM weight is used as-is, at 1.0 the link is forced to maximum weight. Optionally supply your own reasoning text. Use shard= to target a specific shard."""
		result = handler.link_thoughts(source_id, target_id, reasoning=reasoning, confidence=confidence, shard_name=shard or None)
		return json.dumps(result)

	@mcp.tool()
	def forget(thought_id: int, shard: str = "") -> str:
		"""Remove a thought and all its edges from the graph. Use shard= to target a specific shard, or leave empty for the global graph."""
		result = handler.forget(thought_id, shard_name=shard or None)
		return json.dumps(result)

	@mcp.tool()
	def rebootstrap_shards(only_empty: bool = True) -> str:
		"""Re-run link reasoning and GNN training on loaded shards. Set only_empty=false to reprocess all shards, not just those with zero edges."""
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
