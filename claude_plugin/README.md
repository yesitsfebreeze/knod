# shard-plugin

Claude Code plugin that connects directly to a running [shard](https://github.com/febreeze/shard) instance over MCP.

**Shard must be running with HTTP MCP transport:**
```bash
shard serve --mcp-transport streamable-http
```
MCP will be available at `http://localhost:8766/mcp` by default.

## Install

```bash
claude plugins install --github febreeze/shard-plugin
```

Or from a local directory:
```bash
claude plugins install /path/to/claude_plugin
```

## Configure

Default URL is `http://localhost:8766/mcp`. To use a different host or port:

```bash
claude plugins configure shard
# set SHARD_MCP_URL=http://localhost:9000/mcp
```

Or set the environment variable:
```bash
export SHARD_MCP_URL=http://192.168.1.10:8766/mcp
```

## Project-level override

Add to your project's `.mcp.json`:
```json
{
  "mcpServers": {
    "shard": { "type": "http", "url": "http://localhost:8766/mcp" }
  }
}
```

## Tools

| Tool | Description |
|---|---|
| `ask` | Semantic search + LLM answer grounded in the graph |
| `ingest` / `ingest_sync` | Store text as thoughts |
| `find_thoughts` | Semantic search without LLM answer |
| `graph_stats` | Thought/edge counts, maturity metrics |
| `list_shards` | All loaded shards |
| `explore_thought` | Inspect a thought and its edges |
| `traverse` | BFS walk from a thought |
| `set_purpose` | Set the graph's focus |
| `add_descriptor` / `remove_descriptor` | Manage context hints |
| `link` | Manually link two thoughts |
| `forget` | Remove a thought |
| `relink` | Rebuild missing edges |
| `create_shard` | Create a new specialist shard |
| `register_shard` | Load an existing .shard file |
| `rebootstrap_shards` | Re-run link reasoning on shards |
