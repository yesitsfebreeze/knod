# shard

A knowledge graph system that ingests raw text, distills it into atomic thoughts, and organizes them in a relevance graph that a GNN learns to navigate. An external LLM handles reasoning — decomposing text, evaluating worth, linking thoughts, and generating answers. A local GNN handles fast graph traversal and retrieval scoring.

Single binary, written in Odin. No runtime dependencies beyond an OpenAI API key.

## Architecture

```
raw text → external LLM decomposes → thoughts + embeddings → graph
                                                                ↓
query → embed → GNN scores + cosine search + edge search → merge → external LLM → answer
```

- **Graph** — Thoughts (nodes) linked by relevance edges. Each thought is an atomic, self-contained statement with an embedding. Each edge carries a weight, reasoning text, and its own embedding. The graph maintains a running profile embedding (mean of all thoughts) and labeled tag dimensions.
- **GNN** — A 3-layer Message Passing Neural Network (~17M params, hidden dim 512) trained on the graph via self-supervised link prediction. Scores node relevance for retrieval. Trained with AdamW, edge masking, and margin-based loss.
- **External LLM** — OpenAI API (configurable). Handles decomposition, evaluation, linking, labeling, and answer generation. Required — the system does not function without it.

## How It Works

### Ingestion

1. External LLM decomposes raw text into atomic thoughts.
2. Each thought is embedded (OpenAI text-embedding-3-small, 1536 dims).
3. For each thought, find similar existing thoughts by cosine similarity.
4. External LLM batch-evaluates link weights and reasoning for candidates.
5. MCMC acceptance gate: early on, all thoughts are kept (exploration). As the graph matures toward a configured threshold, unconnected thoughts are increasingly rejected (specialization). Acceptance follows exponential decay: `1.0 * 0.05^maturity`.
6. Accepted thoughts are added to the graph with edges above the min link weight.
7. Rejected thoughts go to a limbo graph (if enabled) for potential strand promotion.
8. GNN retrains on the updated graph. Checkpoint saved.

### Retrieval

When a question is asked:

1. Query is embedded.
2. Three parallel searches run:
   - **GNN scoring** — forward pass scores all nodes by relevance.
   - **Cosine similarity** — direct embedding search over thoughts.
   - **Edge embedding search** — find relevant edges (0.8 weight dampening).
3. Results are merged (best score per unique thought).
4. Top-k thoughts + edge reasoning assembled as context.
5. External LLM generates the answer.

### Multi-Store Queries

`shard ask` spawns subprocesses — one per registered strand graph. Each subprocess loads its graph, runs cosine similarity search, and returns scored results via stdout. The parent process merges, deduplicates, ranks, and calls the LLM for the final answer.

### Limbo & Strand Promotion

Thoughts rejected by the MCMC gate are stored in a limbo graph. A background scan (every 60s) checks for clusters that exceed `limbo_cluster_min`. When a cluster is large enough, the external LLM suggests a name and purpose, and the cluster can be promoted to a new strand store.

## Modules

| Module | Purpose |
|--------|---------|
| `graph/` | Core graph data structure — thoughts, edges, embeddings, cosine similarity search, adjacency lists, binary persistence (streaming + full save/load), tags, descriptors |
| `gnn/` | MPNN model — creation, Xavier init, forward/backward pass, layer norm, message passing, link prediction training, AdamW optimizer, checkpoint save/load |
| `provider/` | External LLM interface — OpenAI implementation with embedding, chat, decomposition, evaluation, linking, answer generation |
| `ingest/` | Ingestion pipeline — decompose, embed, link, MCMC gate, limbo routing, cluster scanning |
| `protocol/` | Request handling — TCP (prefix-based commands) and HTTP (REST API, background thread) |
| `registry/` | Strand store management — INI-format store/knid registry at `~/.config/shard/stores` |
| `config/` | Configuration — INI parsing from `~/.config/shard/config`, defaults for all parameters |
| `cli/` | CLI subcommands — `new`, `register`, `list`, `knid`, `ask` |
| `logger/` | Multi-channel file logging — shard.log, performance.log, error.log with rotation |
| `repl/` | Interactive REPL — non-blocking stdin polling with command dispatch |
| `http/` | Bundled HTTP server library — routing, request parsing, response building |
| `util/` | Minimal helpers |

## Usage

### Build & Run

```sh
just build          # optimized build → bin/shard.exe
just run            # build + run
just debug          # debug build
just test           # run all unit tests
```

### Configuration

Config lives at `~/.config/shard/config` (INI format, created on first run):

```ini
api_key = sk-...
embedding_model = text-embedding-3-small
chat_model = gpt-4o-mini
tcp_port = 7999
http_port = 8080
graph_path = shard.graph
max_thoughts = 10000
similarity_threshold = 0.7
```

### CLI

```sh
shard                        # start main process (TCP + HTTP + REPL)
shard new                    # create a new strand store
shard register <path>        # register an existing graph file
shard list                   # list all stores
shard list --knid=<name>     # list stores in a knowledge cluster
shard knid ...               # manage knowledge clusters
shard ask <query>            # query across all stores (subprocess per store)
shard ask --knid=<name> <q>  # query within a specific cluster
```

### TCP Commands

Connect to port 7999 (default):

```
PURPOSE:<text>                  set graph purpose/specialization
ASK:<query>                     ask a question, get LLM answer
DESCRIPTOR_ADD:<name>\n<text>   add a decomposition descriptor
DESCRIPTOR_REMOVE:<name>        remove a descriptor
DESCRIPTOR_LIST:                list all descriptors
INGEST_D:<name>\n<text>         ingest with a named descriptor
<plain text>                    ingest raw text
```

### HTTP API

Port 8080 (default):

```
POST   /ingest                  ingest text (body = raw text)
POST   /ingest?descriptor=name  ingest with descriptor
POST   /ask                     ask question (body = query text)
POST   /purpose                 set purpose (body = text)
POST   /descriptor/add?name=x   add descriptor (body = text)
DELETE /descriptor/<name>       remove descriptor
GET    /descriptor              list descriptors
GET    /health                  health check
```

### Justfile Commands

```sh
just ingest         # start shard in background
just send "text"    # send text to running node
just send-file f    # send file contents to running node
just ask "query"    # ask a question
just stop           # stop running shard
just status         # check if shard is running
just log            # view shard log
just clean          # remove build artifacts
```

## Persistence

Graph data uses a custom binary format:

- **Magic**: `0x6B6E6F67` ("knog"), **Version**: 2
- **Streaming**: append-only writes (thought/edge records appended as they arrive)
- **Full save**: header + all thoughts + all edges
- **GNN checkpoints**: separate file (`.gnn`), magic `0x6B6E6E67` ("knng")

All embeddings stored as `[1536]f32`. Strings as length-prefixed byte arrays.

## Design Principles

- **The graph is the product.** Every design decision serves the knowledge graph.
- **Thoughts are atomic.** One idea per node. Small nodes enable precise retrieval.
- **Hybrid by design.** Local GNN for fast navigation, external LLM for reasoning. Neither is optional.
- **MCMC drives specialization.** Acceptance gating — not pruning — shapes what a graph keeps.
- **One executable, many strands.** Single binary, one store per graph file, subprocess-based fan-out for queries.
- **Simple persistence.** Binary files, no database.
