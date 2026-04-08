"""Test MCP ingest."""

import sys

sys.path.insert(0, "E:\\projects\\shard")

from shard.config import Config
from shard.handler import Handler

cfg = Config.load()
print(f"Voyage: {cfg.voyage_model}")
print(f"Anthropic: {cfg.anthropic_model}")
print(f"Local: {cfg.local_base_url}")

handler = Handler(cfg)
handler.init()

print("Testing ingest...")
result = handler.ingest_sync(
	"The Model Context Protocol (MCP) is an open protocol that standardizes how applications provide context to large language models. MCP uses a client-server architecture.",
	source="mcp-test",
)
print(f"Result: {result}")

stats = handler.graph_stats()
thoughts = stats["global"]["thoughts"]
edges = stats["global"]["edges"]
print(f"Total thoughts: {thoughts}")
print(f"Total edges: {edges}")

handler.shutdown()
print("Ingest test complete!")
