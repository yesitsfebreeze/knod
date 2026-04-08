"""Quick diagnostic: verify config loads and embedding provider initialises."""
import sys
sys.path.insert(0, ".")

from pathlib import Path
from shard.config import Config
from shard.provider import Provider

cfg = Config.load(Path("."))

print("=== Config ===")
print(f"  openai_api_key : {'SET (' + cfg.openai_api_key[:8] + '...)' if cfg.openai_api_key else 'NOT SET'}")
print(f"  openai_base_url        : {cfg.openai_base_url}")
print(f"  openai_embedding_model : {cfg.openai_embedding_model}")
print(f"  local_base_url         : {cfg.local_base_url or 'NOT SET'}")
print(f"  local_embedding_model  : {cfg.local_embedding_model or 'NOT SET'}")
print()

p = Provider(cfg)
print("=== Provider flags ===")
print(f"  _use_openai : {p._use_openai}")
print(f"  _use_local  : {p._use_local}")
print()

print("=== Probing embed ===")
try:
    p._probe_embed()
    print(f"  embed_provider : {p._embed_provider}")
except RuntimeError as e:
    print(f"  FAILED: {e}")
    sys.exit(1)

print("\n=== Test embed ===")
vec = p.embed_text("hello world")
print(f"  shape : {vec.shape}, dtype : {vec.dtype}")
print("\nAll good.")
