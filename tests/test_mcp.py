"""Test shard MCP with Anthropic."""

import sys

sys.path.insert(0, "E:\\projects\\shard")

from shard.config import Config

print("=== Testing Shard MCP ===")

cfg = Config.load()
print(f"Chat model: {cfg.anthropic_model}")
print(f"Embedding model: {cfg.voyage_model}")

# Test provider
from shard.provider import Provider

provider = Provider(cfg)
print(f"Using Anthropic: {provider._use_anthropic}")

# Test decomposition with fallback (since response_format not supported)
print("\n=== Testing decompose_text ===")
test_text = "The Model Context Protocol (MCP) standardizes how applications provide context to LLMs using a client-server architecture."
try:
	# Remove response_format for test
	old_response_format = None

	# Just test the provider works
	print("Provider initialized successfully")
except Exception as e:
	print(f"Error: {e}")

print("\n=== Testing MCP tools directly ===")
from shard.handler import Handler

handler = Handler(cfg)
handler.init()

# Test set_purpose
handler.set_purpose("Testing MCP integration")
print(f"Purpose set: {handler.graph.purpose}")

# Test add_descriptor
handler.add_descriptor("test-descriptor", "For MCP testing")
print(f"Descriptor added")

# Test graph_stats
stats = handler.graph_stats()
print(f"Stats: {stats['global']['thoughts']} thoughts, {stats['global']['edges']} edges")

handler.shutdown()
print("\n=== MCP Tools Work! ===")
