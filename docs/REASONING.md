# Reason

Last updated: 2026-04-05

## Vision

Shard is a network of self-organizing specialist agents. Each specialist is one knowledge graph plus one reasoning agent — small, focused, and cheap to run. Specialists do not call each other directly. Instead they broadcast their learned identity into the network as a tag cloud and emit signals when they need more information. Other specialists hear those signals and respond if their tags match. Intelligence is not in any one node — it lives in the network topology and the protocol that lets nodes find each other.

The fundamental unit: `1 graph + 1 agent = 1 specialist`.

Scaling means acquiring more specialists, not bigger models. A new specialist auto-integrates by broadcasting its tag cloud. No manual wiring required.

## Purpose

People accumulate knowledge constantly — documents, notes, conversations — but have no good way to store it so it can be queried by meaning later. Search finds text matches. Databases find structured records. Neither understands relationships between ideas or can surface context for a question that doesn't use the same words as the stored information.

Shard solves this at two levels. Within a specialist: a graph of distilled thoughts linked by relevance, navigated by a GNN that learns the structure. Across specialists: a self-organizing network where specialists discover each other through tag broadcasts and reputation scoring. The result is an intelligence that scales horizontally — the more specialists exist, the better the network answers — without any single node needing to grow.

## The Broadcast Protocol

Each specialist periodically broadcasts its learned self-description. Packet format:

```
{ agent_id, tag, score }
```

Tags are **not hand-written**. The embedding model clusters the specialist's thought vectors; the LLM reads those clusters and names them. The tag cloud reflects what the specialist actually knows, updated as the graph learns.

Two score dimensions:
- `tag_score` — how strongly this specialist owns this tag (derived from vector space density)
- `agent_score` — how reliable this specialist is overall (accumulated from consumer feedback)

The registry becomes a tag index. Signal resolution: match tags, rank by `tag_score × agent_score`, route to the winner.

## Floating Edges

Specialists do not hard-code references to other specialists. A thought that needs more information emits a **signal** — a floating edge with a query and no target:

```
thought_A ──[signal: "more on X"]──► ???
```

At retrieval time, the routing layer resolves floating edges against the tag index. Unanswered signals remain visible as gaps. A new specialist that covers topic X will automatically attract those signals the moment it broadcasts its tag cloud.

## The Reputation Loop

```
query → tag match → specialist answers → consumer scores → agent_score updates → routing shifts
```

The network self-improves through use. No curation. No manual ranking. Bad specialists sink; good ones rise.

## Why Horizontal Scaling Changes Everything

| Vertical (today) | Horizontal (this) |
|---|---|
| Bigger model = more capable | More specialists = more capable |
| One model knows everything poorly | Each specialist knows its domain deeply |
| Expensive, centralized | Cheap, distributed |
| Wait for next model release | Acquire a new specialist |

This is how intelligence works at biological and social scale. No single neuron or person knows everything — the network does.

## Current State

The system is functional end-to-end. What exists today:

### Implemented
- **Graph storage** — Full thought/edge data structure with cosine similarity search, adjacency lists, running profile embedding, tag dimensions, and named descriptors. Binary persistence with streaming (append-only) and full save/load. Format version 2 with migration from v1.
- **GNN model** — 3-layer MPNN with 512 hidden dim (~17M params). Xavier initialization, forward/backward pass, layer normalization, message passing with scatter-add aggregation. Self-supervised training via edge masking (15% mask ratio) with link prediction and relevance scoring heads. AdamW optimizer with adaptive learning rate. Checkpoint save/load.
- **Ingestion pipeline** — External LLM decomposes text into thoughts, embeds them, batch-evaluates link candidates. MCMC acceptance gating drives specialization: exponential decay from full acceptance (empty graph) to 5% acceptance (mature graph). Limbo graph catches rejected thoughts for potential strand promotion.
- **Provider interface** — Vtable-based provider abstraction. OpenAI implementation with embedding (single + batch), chat completion, JSON mode support. Higher-level reasoning procs (decompose, evaluate, link, label, suggest) defined in the interface.
- **Protocol layer** — TCP with prefix-based commands (PURPOSE, ASK, DESCRIPTOR_ADD/REMOVE/LIST, INGEST_D, plain ingest). HTTP REST API on background thread with mutex-guarded handler access. Routes for ingest, ask, purpose, descriptors, and health.
- **Multi-store queries** — CLI `ask` command spawns subprocesses per registered store. Each subprocess loads its graph, runs cosine search, returns scored results via stdout pipe. Parent merges, deduplicates, ranks, and calls LLM for final answer.
- **Registry** — INI-format store/knid registry at `~/.config/Shard/stores`. Stores map names to graph file paths. Knids group stores into knowledge clusters.
- **Configuration** — INI parsing from `~/.config/Shard/config`. Auto-creates default config on first run. Covers provider settings, ports, graph limits, ingestion thresholds, limbo parameters.
- **CLI** — Subcommands: `new` (create strand), `register` (add existing graph), `list` (show stores/knids), `knid` (manage clusters), `ask` (cross-store query).
- **REPL** — Interactive non-blocking command loop with dispatch.
- **Logging** — Multi-channel (Shard.log, perf.log, error.log) with severity levels and timestamps.
- **Integration test** — Python script that starts Shard, sets purpose, fetches Wikipedia articles, ingests them, and asks questions to verify the full pipeline.

### Not Yet Implemented
- **Research loop** — Autonomous internet searching based on existing content. Described in early design but not built.
- **UDP gossip** — Tag vector broadcasting between nodes. The architecture shifted to subprocess fan-out for multi-store queries instead of live peer communication.
- **Edge decay** — Config field exists but decay is not applied during graph operations.
- **Graph limits enforcement** — `max_thoughts` and `max_edges` are configured but not enforced during insertion.
- **Strand GNN layers** — `StrandMPNN` type defined but strand layer training (base + strand split) is not implemented. Currently one model per process.

## Principles

1. **The graph is the product.** Every design decision serves the quality and navigability of the knowledge graph. The GNN, the external LLM, the ingestion pipeline — all are tools that serve the graph.

2. **Thoughts are atomic.** Each node in the graph is one idea, one fact, one observation. Never a paragraph, never a document. Small nodes make precise retrieval possible and re-linking cheap.

3. **Hybrid by design.** The GNN handles fast, cheap graph navigation and retrieval scoring. External LLMs handle reasoning, evaluation, and answer generation. Neither is optional — the system requires both.

4. **One executable, many strands.** A single Shard binary handles everything. Each strand is a graph file on disk. Cross-store queries spawn subprocesses of the same binary, each loading one graph. No distribution needed — subprocess fan-out handles parallelism on one machine.

5. **MCMC drives specialization.** Acceptance gating — not pruning — shapes what a graph keeps. Early graphs accept everything (exploration). Mature graphs are selective (specialization). The transition follows exponential decay.

6. **Two-part navigation model.** The GNN has a base architecture that learns graph traversal structure. Strand layers (per-graph fine-tuning) are planned but not yet implemented. Currently the GNN trains per-graph from scratch.

7. **Tags are learned, not declared.** A specialist's identity comes from its embedding space — cluster the vectors, name the clusters, broadcast the result. The tag cloud is always true to what the graph actually contains.

8. **Floating edges over hard links.** When a specialist needs more information, it emits a signal with no target. The network resolves it. This keeps each specialist's data model clean and makes the graph topology emergent rather than designed.

9. **Reputation accumulates from use.** Agent scores come from consumer feedback on real queries, never from manual curation. The network routes toward quality automatically.

10. **Scale by acquiring, not upgrading.** A new specialist integrates itself by broadcasting its tag cloud. The rest of the network discovers it automatically. Capability grows by addition, not replacement.

## Out of Scope

- **Not a chatbot.** Shard does not maintain conversation history or engage in dialogue. It answers questions from its knowledge graph.
- **Not a standalone LLM.** The GNN is a graph navigation tool, not a text generator. It never speaks directly to the user.
- **Not a database.** No arbitrary queries, schemas, joins, or transactions. Thoughts and relevance links only.
- **Not offline-capable.** The system requires an OpenAI API key. No degraded offline mode — if the API is unreachable, ingestion and querying do not work.
