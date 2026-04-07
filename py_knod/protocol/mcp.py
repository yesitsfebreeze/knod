"""MCP server — exposes knod graph as Model Context Protocol tools + resources."""

from __future__ import annotations

import json
import logging

from mcp.server.fastmcp import FastMCP

from ..handler import Handler

log = logging.getLogger(__name__)


def _mcp(handler: Handler) -> FastMCP:
	"""Build a FastMCP server wired to the shared Handler."""
	mcp = FastMCP("knod", json_response=True)

	# ---- Tools ----

	@mcp.tool()
	def ask(query: str) -> str:
		"""Ask a question against the knowledge graph. Returns an answer grounded in stored thoughts."""
		answer, sources = handler.handle_ask(query)
		return json.dumps({"answer": answer, "sources": sources})

	@mcp.tool()
	def ingest(text: str, source: str = "", descriptor: str = "") -> str:
		"""Ingest text into the knowledge graph. The text is decomposed into atomic thoughts, linked, and stored."""
		result = handler.handle_ingest_queued(text, source=source, descriptor=descriptor)
		return result

	@mcp.tool()
	def set_purpose(purpose: str) -> str:
		"""Set the purpose / focus of the knowledge graph."""
		handler.handle_set_purpose(purpose)
		return f"Purpose set to: {purpose}"

	@mcp.tool()
	def add_descriptor(name: str, description: str) -> str:
		"""Add a descriptor (named context hint) to guide future ingestion."""
		handler.handle_descriptor_add(name, description)
		return f"Descriptor '{name}' added."

	@mcp.tool()
	def remove_descriptor(name: str) -> str:
		"""Remove a descriptor by name."""
		ok = handler.handle_descriptor_remove(name)
		return f"Descriptor '{name}' removed." if ok else f"Descriptor '{name}' not found."

	@mcp.tool()
	def find_thoughts(query: str, k: int = 5) -> str:
		"""Search for thoughts semantically similar to the query. Returns top-k matches without generating an LLM answer."""
		results = handler.find_thoughts_by_query(query, k=k)
		return json.dumps(results)

	# ---- Resources ----

	@mcp.resource("knod://status")
	def status() -> str:
		"""Current graph status: thought count, edge count, purpose."""
		return handler.handle_status()

	@mcp.resource("knod://graph")
	def graph_info() -> str:
		"""Graph metadata as JSON."""
		return json.dumps(handler.graph_info)

	@mcp.resource("knod://thoughts")
	def list_thoughts() -> str:
		"""List all thoughts as JSON array (id, text snippet, source)."""
		return json.dumps(handler.all_thoughts)

	@mcp.resource("knod://descriptors")
	def list_descriptors() -> str:
		"""List all descriptors as JSON."""
		return json.dumps(handler.graph_info["descriptors"])

	# ---- Prompts ----

	@mcp.prompt()
	def research(topic: str) -> str:
		"""Generate a prompt to research a topic using the knowledge graph."""
		return (
			f"Use the knod knowledge graph to answer questions about: {topic}\n\n"
			"1. First use find_thoughts to see what's already known\n"
			"2. Then use ask to get a synthesized answer\n"
			"3. If knowledge is lacking, use ingest to add relevant text"
		)

	return mcp
