# py_knod — Cleanup Issues

Identified via structural audit of the Python codebase. Each issue includes the affected files, nature of the problem, and a concrete example. Ordered roughly by breadth of impact.

---

## 1. Cosine similarity duplicated across 8+ sites

**Files:** `specialist/graph.py`, `ingest/dedup.py`, `retrieval/score.py`, `retrieval/expand.py`, `retrieval/rate.py`, `handler.py`, `limbo/promote.py`

**Problem:** The same three-line normalize-then-dot-product pattern is copy-pasted at least 8 times. The epsilon `1e-10` appears in every copy. `retrieval/expand.py` has a private `_cosine()` helper that is never used outside that module.

**Example:**
```python
# dedup.py:36-39
query = pt.embedding / (np.linalg.norm(pt.embedding) + 1e-10)
for t in graph.thoughts.values():
    t_norm = t.embedding / (np.linalg.norm(t.embedding) + 1e-10)
    sim = float(np.dot(query, t_norm))

# retrieval/expand.py:69-72  (private, unused elsewhere)
def _cosine(a, b):
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a_norm, b_norm))
```

**Fix:** Promote `_cosine` to a public `cosine(a, b)` in a shared `retrieval/math.py` (or `specialist/math.py`). Replace all inline copies with a single import.

---

## 2. `dedup.py` re-implements `Graph.find_thoughts()` in O(N)

**Files:** `ingest/dedup.py:dedup()`, `specialist/graph.py:Graph.find_thoughts()`

**Problem:** `dedup()` hand-rolls the same O(N) cosine scan that `find_thoughts()` already does. Any future optimisation of `find_thoughts` (batched matrix multiply, approximate search) does not benefit dedup. The two copies can drift independently in both logic and epsilon.

**Example:**
```python
# dedup.py:36-47 — hand-rolled
query = pt.embedding / (np.linalg.norm(pt.embedding) + 1e-10)
for t in graph.thoughts.values():
    ...

# graph.py:100-111 — equivalent
query = embedding / (np.linalg.norm(embedding) + 1e-10)
for t in self.thoughts.values():
    ...
```

**Fix:** Replace the loop in `dedup()` with `graph.find_thoughts(pt.embedding, threshold=0, k=1)` and pick the top result.

---

## 3. Chain dedup + ranking duplicated between `ask()` and `ask_knid()`

**Files:** `handler.py:Handler.ask()` lines 254–263, `handler.py:Handler.ask_knid()` lines 380–388

**Problem:** An identical 8-line chain-merge/dedup block is copy-pasted verbatim. Additionally, `ask()` calls `rate_thoughts()` (re-ranking step) while `ask_knid()` silently omits it — knid-scoped queries produce differently ranked results with no visible API difference.

**Example:**
```python
# Both methods — identical block
surviving_ids = {t.id for t, _ in scored}
chain_by_terminal: dict[int, PathChain] = {}
for chains in all_chains:
    for chain in chains:
        if chain.terminal and chain.terminal.id in surviving_ids:
            tid = chain.terminal.id
            if tid not in chain_by_terminal or chain.score > chain_by_terminal[tid].score:
                chain_by_terminal[tid] = chain
best_chains = sorted(chain_by_terminal.values(), key=lambda c: c.score, reverse=True)
```

**Fix:** Extract `best_chains_from(all_chains, scored)` helper in `retrieval/`. Apply `rate_thoughts` in both paths, or document the intentional omission explicitly.

---

## 4. `.knod` binary parser duplicated in `load_knod` / `read_knod_metadata`

**Files:** `specialist/store.py:load_knod()` lines 451–558, `specialist/store.py:read_knod_metadata()` lines 561–599

**Problem:** Both functions open the same binary format and run the same section-loop parse. `read_knod_metadata` skips the version check that `load_knod` performs — passing a v3 file produces garbage instead of a clear error.

**Example:**
```python
# Both functions contain:
off = 8
while off < len(data):
    tag = data[off]; off += 1
    length = struct.unpack_from("<q", data, off)[0]; off += 8
    payload = data[off : off + length]; off += length
```

**Fix:** Extract `_parse_knod_sections(data) -> Iterator[tuple[int, bytes]]` that validates magic + version once and yields `(tag, payload)`. Both `load_knod` and `read_knod_metadata` consume this iterator.

---

## 5. Thought/Edge serialization triplicated across `save_graph`, `_graph_state`, `load_knod`

**Files:** `specialist/store.py:save_graph()` lines 183–196, `specialist/store.py:_graph_state()` lines 386–398, `specialist/store.py:load_knod()` lines 508–530, `specialist/store.py:GraphLog.replay()` lines 149–157

**Problem:** The `Thought → dict` serialization pattern is written identically in `save_graph` and `_graph_state`. The `Thought` reconstruction from dict in `load_graph` is repeated verbatim in `load_knod`. The `LimboThought` deserialization is triplicated. Adding a field to `Thought` requires touching at least three separate code paths.

**Fix:** Add `Thought.to_dict() -> dict` and `Thought.from_dict(d) -> Thought` class methods. Do the same for `Edge` and `LimboThought`. All serialization paths call these methods.

---

## 6. `_NON_ANSWER_PHRASES` tuple re-constructed on every `Handler.ask()` call

**Files:** `handler.py:Handler.ask()` lines 278–291

**Problem:** A literal tuple of LLM refusal phrases is defined as a local variable inside the hot `ask()` method body. It is re-constructed on every call. It is not configurable, not exposed at the module boundary, and cannot be injected in tests.

**Example:**
```python
# handler.py — inside ask(), executes on every query
_NON_ANSWER_PHRASES = (
    "the context does not",
    "the provided context",
    ...
)
```

**Fix:** Move to a module-level constant (or `Config` field). Makes the flywheel gate visible and testable.

---

## 7. `Config.load()` has three hard-coded key lists; several fields are silently unloadable

**Files:** `config.py:Config.load()` lines 65–130

**Problem:** Three separate hard-coded string lists (str keys, int keys, float keys) must stay in sync with the `Config` dataclass. Fields `margin`, `lr_max`, `lr_min`, `weight_decay`, `maturity_divisor` do not appear in any list — setting them in `~/.config/knod/config` silently has no effect.

**Example:**
```python
for key in ("api_key", "base_url", ...):          # strings
    if key in kv: setattr(cfg, key, kv[key])
for key in ("http_port", "tcp_port", ...):         # ints
    if key in kv: setattr(cfg, key, int(kv[key]))
for key in ("similarity_threshold", ...):          # floats — missing: margin, lr_max, lr_min, ...
    if key in kv: setattr(cfg, key, float(kv[key]))
```

**Fix:** Use `dataclasses.fields()` + `typing.get_type_hints()` to derive the load loops from the dataclass definition at runtime, eliminating the manual lists. Silent dead fields become impossible.

---

## 8. `Graph.maturity` hard-codes `50.0`; `Config.maturity_divisor` is dead code

**Files:** `specialist/graph.py:Graph.maturity` line 64, `config.py:Config.maturity_divisor` line 33

**Problem:** `Config` defines `maturity_divisor: int = 50`. `Graph.maturity` hard-codes `50.0` directly and never reads the config field. Changing `maturity_divisor` in the config file has no effect.

**Example:**
```python
# config.py:33
maturity_divisor: int = 50

# graph.py:64 — ignores config
@property
def maturity(self) -> float:
    return min(self.num_thoughts / 50.0, 1.0)
```

**Fix:** Pass `maturity_divisor` into `Graph` at construction (or store `cfg` reference). Remove the dead config field illusion.

---

## 9. `Specialist` dataclass lives in `handler.py`; imported upward by `limbo/promote.py`

**Files:** `handler.py:Specialist` lines 74–84, `limbo/promote.py:_spawn_specialist()` line 185

**Problem:** `limbo/promote.py` imports `Specialist` from `handler.py` — the top-level orchestrator. This creates an upward dependency from a low-level utility module into the highest-level module. `limbo` cannot be imported or tested without loading the full application graph. A latent circular import exists: `handler.py` imports `limbo` (line 27), and `limbo/promote.py` imports `handler`.

**Fix:** Move `Specialist` (and `SpecialistIndexEntry`, `GraphEvent`) to `specialist/types.py`. Both `handler.py` and `limbo/promote.py` import from there.

---

## 10. `bootstrap_thoughts` re-implements the ingest commit path, bypassing MCMC/dedup/snapshot

**Files:** `limbo/promote.py:bootstrap_thoughts()` lines 29–91, `ingest/commit.py:commit()` lines 60–69

**Problem:** `bootstrap_thoughts` manually reconstructs `PreparedThought` objects, calls `link()`, then iterates `pt.links` to call `graph.add_edge()` — duplicating the exact logic of `commit.py`. Critically, it bypasses the MCMC gate, dedup, and snapshot phases. Limbo-promoted thoughts enter the graph through a different code path with different semantics, with no documentation of the divergence.

**Example:**
```python
# limbo/promote.py:69-79 — manual commit
for link_data, emb in zip(pt.links, pt.link_embeddings):
    idx = link_data["index"]
    target_id = pt.candidate_ids[idx]
    graph.add_edge(source_id=tid, target_id=target_id, ...)

# ingest/commit.py:60-69 — the canonical path
for link, emb in zip(pt.links, pt.link_embeddings):
    target_id = pt.candidate_ids[link["index"]]
    if target_id in graph.thoughts:
        graph.add_edge(source_id=thought.id, target_id=target_id, ...)
```

**Fix:** Expose a `commit_prepared(graph, prepared_thoughts)` function from `ingest/commit.py` that `bootstrap_thoughts` calls directly. Document any intentional skips (MCMC bypass is intentional — limbo thoughts already passed the gate once).

---

## 11. MCP calls deprecated `handle_*` shims; HTTP calls canonical methods

**Files:** `protocol/mcp.py` lines 24, 30, 36, 46, 48, 62; `protocol/http.py` lines 55, 64, 82, 86, 91; `handler.py` lines 441–463

**Problem:** MCP and HTTP implement the same operations through different code paths. MCP uses deprecated `handle_ask`, `handle_ingest_queued`, `handle_set_purpose`, etc. shims. HTTP calls `ask()`, `ingest_sync()`, `set_purpose()` directly. A bug in the shim layer affects only MCP clients.

**Example:**
```python
# mcp.py — calls deprecated shim
answer, sources = handler.handle_ask(query)

# http.py — calls canonical method
answer, sources = handler.ask(req.query)
```

**Fix:** Update `mcp.py` to call canonical handler methods. Remove the deprecated shims.

---

## 12. `TCPServer._subscribers` mutated without a lock in `_dispatch`

**Files:** `protocol/tcp.py:_dispatch()` line 54, `protocol/tcp.py:TCPServer._on_event()` lines 191–202, `protocol/tcp.py:TCPServer._accept_loop()` lines 220–228

**Problem:** `_on_event` iterates `self._subscribers` under `self._subs_lock`, but `_dispatch` adds to the same set without acquiring any lock. Concurrent `SUBSCRIBE` commands from multiple connections race on the shared set.

**Example:**
```python
# tcp.py:220-228 — passes bare set to thread, no lock
args=(client, self.handler, self._subscribers)

# tcp.py:54-56 — mutates without lock
if stripped == "SUBSCRIBE":
    subscribers.add(sock)   # race condition
```

**Fix:** Pass `(self._subscribers, self._subs_lock)` to connection threads. Acquire the lock in `_dispatch` before mutating.

---

## 13. `Registry.__init__` has disk-write side effects via `_load()` → `self.save()`

**Files:** `registry.py:Registry.__init__()` line 48, `registry.py:Registry._load()` lines 103–111

**Problem:** Constructing a `Registry()` object may write to `~/.config/knod/stores` as a side effect of pruning stale knid members. Tests work around this with `Registry.__new__` to bypass `__init__` entirely — a clear symptom that the constructor is unsafe to call in a test environment.

**Example:**
```python
# registry.py — _load() called from __init__, may write to disk:
if changed:
    self.save()   # disk write inside constructor

# test_integration.py:188 — workaround
registry = Registry.__new__(Registry)
```

**Fix:** Separate `_load()` from `_prune_stale()`. Do not call `save()` inside the constructor. Expose `registry.prune_stale()` as an explicit operation callers opt into.

---

## 14. `Handler._ingest_sync` holds `self.mu` across GNN training and two disk writes

**Files:** `handler.py:Handler._ingest_sync()` lines 185–201

**Problem:** The mutex is held for edge decay, base model reload (disk I/O), full GNN training (`trainer.train_on_graph`), and `self.save()` (pickle + torch.save). Any concurrent `status()` call, TCP read, or HTTP request blocks for the full training duration. `status()` (line 319) reads graph state without acquiring `self.mu` — a data race.

**Example:**
```python
# handler.py:185-194
with self.mu:
    load_base_model(self.model)            # disk I/O
    loss = self.trainer.train_on_graph(...)  # training loop
    self.save()                            # two disk writes
```

**Fix:** Narrow the lock to cover only graph mutations. Perform training on a snapshot outside the lock. Acquire `self.mu` in `status()`.

---

## 15. Tests manually wire `Handler` internal state, bypassing `init()`

**Files:** `tests/test_integration.py` lines 307–311, `tests/test_edges.py` lines 72–79

**Problem:** Tests directly assign `handler.graph`, `handler.model`, `handler.strand`, `handler._queue`, `handler._in_flight`, `handler.ingester`, and `handler.trainer` because there is no factory or dependency injection. Any rename of these private attributes silently breaks the test suite. Tests also call `handle_ingest()` (the deprecated shim) rather than `ingest_sync()`.

**Example:**
```python
# test_integration.py:307-311
handler = Handler(cfg)
handler.graph = Graph()
handler.model = KnodMPNN(cfg)
handler.strand = StrandLayer(cfg.hidden_dim)
handler._queue = __import__("queue").Queue(maxsize=128)
```

**Fix:** Add `Handler.from_parts(cfg, graph, model, ...)` factory classmethod. Tests use the factory instead of directly assigning private state. Update test calls from `handle_ingest` to `ingest_sync`.
