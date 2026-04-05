# Reason

Last updated: 2026-04-05

## Vision

Knod is an AI-native knowledge graph system. It ingests raw text, distills it into atomic thoughts, and organizes them in a relevance graph that a GNN learns to navigate. A single Odin executable runs as one process: it owns the graph, the GNN model, and the ingestion pipeline. When querying across multiple specialists, the process spawns subprocesses of the same binary — each pointed at a different graph file — to retrieve in parallel. External LLMs (OpenAI) handle the expensive reasoning: decomposing text, evaluating connections, and generating answers from retrieved context. The result is an always-available knowledge store that specializes through MCMC acceptance gating and scales to many specialists via subprocess fan-out.

## Purpose

People accumulate knowledge constantly — documents, notes, conversations — but have no good way to store it so it can be queried by meaning later. Search finds text matches. Databases find structured records. Neither understands relationships between ideas or can surface context for a question that doesn't use the same words as the stored information.

Knod solves this by maintaining a graph of distilled thoughts linked by relevance. When asked a question, it doesn't just find the closest text match — it finds the most relevant thought and fans out along relationship edges to assemble context that a flat search would miss. The GNN makes retrieval fast and structure-aware; the external LLM makes the final answer accurate and reasoned.

## Current State

The system is functional end-to-end. What exists today:

### Implemented
- **Graph storage** — Full thought/edge data structure with cosine similarity search, adjacency lists, running profile embedding, tag dimensions, and named descriptors. Binary persistence with streaming (append-only) and full save/load. Format version 2 with migration from v1.
- **GNN model** — 3-layer MPNN with 512 hidden dim (~17M params). Xavier initialization, forward/backward pass, layer normalization, message passing with scatter-add aggregation. Self-supervised training via edge masking (15% mask ratio) with link prediction and relevance scoring heads. AdamW optimizer with adaptive learning rate. Checkpoint save/load.
- **Ingestion pipeline** — External LLM decomposes text into thoughts, embeds them, batch-evaluates link candidates. MCMC acceptance gating drives specialization: exponential decay from full acceptance (empty graph) to 5% acceptance (mature graph). Limbo graph catches rejected thoughts for potential specialist promotion.
- **Provider interface** — Vtable-based provider abstraction. OpenAI implementation with embedding (single + batch), chat completion, JSON mode support. Higher-level reasoning procs (decompose, evaluate, link, label, suggest) defined in the interface.
- **Protocol layer** — TCP with prefix-based commands (PURPOSE, ASK, DESCRIPTOR_ADD/REMOVE/LIST, INGEST_D, plain ingest). HTTP REST API on background thread with mutex-guarded handler access. Routes for ingest, ask, purpose, descriptors, and health.
- **Multi-store queries** — CLI `ask` command spawns subprocesses per registered store. Each subprocess loads its graph, runs cosine search, returns scored results via stdout pipe. Parent merges, deduplicates, ranks, and calls LLM for final answer.
- **Registry** — INI-format store/knid registry at `~/.config/knod/stores`. Stores map names to graph file paths. Knids group stores into knowledge clusters.
- **Configuration** — INI parsing from `~/.config/knod/config`. Auto-creates default config on first run. Covers provider settings, ports, graph limits, ingestion thresholds, limbo parameters.
- **CLI** — Subcommands: `new` (create specialist), `register` (add existing graph), `list` (show stores/knids), `knid` (manage clusters), `ask` (cross-store query).
- **REPL** — Interactive non-blocking command loop with dispatch.
- **Logging** — Multi-channel (knod.log, perf.log, error.log) with severity levels and timestamps.
- **Integration test** — Python script that starts knod, sets purpose, fetches Wikipedia articles, ingests them, and asks questions to verify the full pipeline.

### Not Yet Implemented
- **Research loop** — Autonomous internet searching based on existing content. Described in early design but not built.
- **UDP gossip** — Tag vector broadcasting between nodes. The architecture shifted to subprocess fan-out for multi-store queries instead of live peer communication.
- **Edge decay** — Config field exists but decay is not applied during graph operations.
- **Graph limits enforcement** — `max_thoughts` and `max_edges` are configured but not enforced during insertion.
- **Strand/specialist GNN layers** — `StrandMPNN` type defined but specialist layer training (base + specialist split) is not implemented. Currently one model per process.

## Principles

1. **The graph is the product.** Every design decision serves the quality and navigability of the knowledge graph. The GNN, the external LLM, the ingestion pipeline — all are tools that serve the graph.

2. **Thoughts are atomic.** Each node in the graph is one idea, one fact, one observation. Never a paragraph, never a document. Small nodes make precise retrieval possible and re-linking cheap.

3. **Hybrid by design.** The GNN handles fast, cheap graph navigation and retrieval scoring. External LLMs handle reasoning, evaluation, and answer generation. Neither is optional — the system requires both.

4. **One executable, many specialists.** A single knod binary handles everything. Each specialist is a graph file on disk. Cross-store queries spawn subprocesses of the same binary, each loading one graph. No distribution needed — subprocess fan-out handles parallelism on one machine.

5. **MCMC drives specialization.** Acceptance gating — not pruning — shapes what a graph keeps. Early graphs accept everything (exploration). Mature graphs are selective (specialization). The transition follows exponential decay.

6. **Two-part navigation model.** The GNN has a base architecture that learns graph traversal structure. Specialist layers (per-graph fine-tuning) are planned but not yet implemented. Currently the GNN trains per-graph from scratch.

7. **Odin stays.** The system is built in Odin. Performance-sensitive paths — graph operations, GNN inference/training, binary persistence — stay in Odin. Python is used only for integration testing.

## Out of Scope

- **Not a chatbot.** Knod does not maintain conversation history or engage in dialogue. It answers questions from its knowledge graph.
- **Not a standalone LLM.** The GNN is a graph navigation tool, not a text generator. It never speaks directly to the user.
- **Not a database.** No arbitrary queries, schemas, joins, or transactions. Thoughts and relevance links only.
- **Not offline-capable.** The system requires an OpenAI API key. No degraded offline mode — if the API is unreachable, ingestion and querying do not work.
