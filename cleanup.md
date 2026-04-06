# Cleanup Opportunities

Audit of `src/` excluding vendored `src/http/`.

---

## 1. Dead Code (safe to delete)

| Function | File | Line | Reason |
|----------|------|------|--------|
| `create_empty_graph` | `src/cli/cli.odin` | 827 | Never called. Replaced by `create_empty_strand`. |
| `gnn_checkpoint_path` | `src/cli/cli.odin` | 743 | Never called anywhere. |
| `compute_loss_and_backward_masked` | `src/gnn/model.odin` | 564 | Never called. Replaced by `compute_loss_and_backward_strand`. |
| `train_on_graph` | `src/gnn/train.odin` | 231 | Never called. Replaced by separate strand/base training procs. |
| `evaluate_thought` | `src/provider/types.odin` | 64 | Provider vtable field — never assigned, never called. |
| `link_reason` | `src/provider/types.odin` | 67 | Provider vtable field — superseded by `batch_link_reason`. Never assigned, never called. |
| `logger.debug` | `src/logger/logger.odin` | 134 | Never called outside vendored `src/http/`. |
| `logger.perf` | `src/logger/logger.odin` | 130 | Never called anywhere. |
| `touch` | `src/graph/graph.odin` | 153 | Only called from `@test.odin`. No production usage. |

---

## 2. Duplicated Functions — Extract to Shared Location

### 2a. `home_dir` — 4 copies (all delegate to `util.home_dir`)

Thin private wrappers that just call `util.home_dir()`. Callers can import `util` directly.

| File | Line | Name |
|------|------|------|
| `src/cli/cli.odin` | 1049 | `home_dir` |
| `src/config/config.odin` | 101 | `home_dir` |
| `src/registry/registry.odin` | 240 | `home_dir` |
| `src/ingest/ingest.odin` | 568 | `home_dir_ingest` |

**Action:** Delete wrappers. Replace callsites with `util.home_dir()`.

### 2b. `ensure_dir` — 5 copies (4 delegate to `util.ensure_dir`, 1 is full reimplementation)

| File | Line | Name | Notes |
|------|------|------|-------|
| `src/cli/cli.odin` | 1054 | `ensure_dir` | Wrapper around `util.ensure_dir` |
| `src/config/config.odin` | 308 | `ensure_dir` | Wrapper around `util.ensure_dir` |
| `src/registry/registry.odin` | 245 | `ensure_dir` | Wrapper around `util.ensure_dir` |
| `src/ingest/ingest.odin` | 573 | `ensure_dir` | Wrapper around `util.ensure_dir` |
| `src/main.odin` | 281 | `ensure_data_dir` | **Full inline reimplementation** — does NOT call `util` |

**Action:** Delete all wrappers. Replace `ensure_data_dir` in `main.odin` with `util.ensure_dir`. Replace all callsites.

### 2c. `write_val` — 2 identical copies

Binary write helper: casts `^$T` to byte slice, writes to fd.

| File | Line |
|------|------|
| `src/cli/cli.odin` | 1042 |
| `src/graph/io.odin` | 136 |

**Action:** Keep in `graph/io.odin` (or move to `util`). Have `cli` import it.

### 2d. `save_graph` — 2 near-identical copies

Both check `.strand` extension, call `gnn.strand_save_bytes` + `graph.save_strand`, or fall back to `graph.save`.

| File | Line | Name | Difference |
|------|------|------|------------|
| `src/protocol/protocol.odin` | 218 | `save_graph` | Returns nothing, takes `^Handler` |
| `src/repl/repl.odin` | 521 | `repl_save_graph` | Returns `bool`, takes `^State`, nil-guards `s.strand` |

**Action:** Extract a shared `graph.save_auto(g, strand, path) -> bool` that both can call.

### 2e. `adamw_step` — 2 structurally identical copies

Same AdamW algorithm (BETA1, BETA2, WEIGHT_DECAY, ADAM_EPS), operating on different model types.

| File | Line | Name | Operates on |
|------|------|------|-------------|
| `src/gnn/model.odin` | 646 | `adamw_step` | `^MPNN` |
| `src/gnn/model.odin` | 1142 | `strand_adamw_step` | `^StrandMPNN` |

**Action:** Unify into a single proc that takes the common fields (params, grads, m, v, adam_t, num_parameters) directly, or use a shared interface.

### 2f. Cosine similarity — 2 copies

Same dot/(norm_a*norm_b) algorithm.

| File | Line | Form | Operates on |
|------|------|------|-------------|
| `src/graph/graph.odin` | 198-212 | Named proc `cosine_similarity` | `^Embedding` pointers |
| `src/gnn/model.odin` | 714-726 | Inline in `score_nodes` | Flat `[]f32` array with stride |

**Action:** Extract to `util.cosine_similarity(a, b: []f32) -> f32`. Adapt both callsites.

### 2g. Relevance score computation — 4 copies

Same block: iterate nodes, compute `b_relevance + dot(hidden[n], w_relevance)`.

| File | Line range |
|------|------------|
| `src/gnn/model.odin` | 695-704 |
| `src/gnn/train.odin` | 201-209 |
| `src/gnn/train.odin` | 287-295 |
| `src/gnn/train.odin` | 345-353 |

**Action:** Extract `gnn.compute_relevance_scores(model, cache, hidden, N, H)` helper.

---

## 3. Insertion Sort — 9 Identical Copies

Descending insertion sort on a scored array. All structurally identical (swap-based, inner loop compares adjacent `.score` or `.value`).

| # | File | Lines | Array name | Sort key |
|---|------|-------|------------|----------|
| 1 | `src/graph/graph.odin` | 229-235 | `results` | `.score` |
| 2 | `src/graph/graph.odin` | 259-265 | `results` | `.score` |
| 3 | `src/graph/tags.odin` | 30-36 | `candidates` | `.value` |
| 4 | `src/gnn/model.odin` | 734-740 | `results` | `.score` |
| 5 | `src/main.odin` | 504-510 | `deduped` | `.score` |
| 6 | `src/main.odin` | 512-518 | `all_edges` | `.score` |
| 7 | `src/cli/cli.odin` | 616-622 | `ranked` | `.score` |
| 8 | `src/protocol/protocol.odin` | 165-171 | `ranked` | `.score` |
| 9 | `src/ingest/ingest.odin` | 257-263 | `entries` | `.value` |

The pattern:
```odin
for i in 1 ..< len(arr) {
    j := i
    for j > 0 && arr[j].score > arr[j - 1].score {
        arr[j], arr[j - 1] = arr[j - 1], arr[j]
        j -= 1
    }
}
```

**Action:** Extract a generic `util.sort_descending` that takes a slice and a comparison proc, or use `slice.sort_by`.

---

## 4. Duplicated Constants

### 4a. `"text-embedding-3-small"` — defined independently in 2 places

| File | Line | Context |
|------|------|---------|
| `src/config/config.odin` | 52 | `embedding_model = "text-embedding-3-small"` (DEFAULT struct) |
| `src/provider/types.odin` | 21 | `embedding_model = "text-embedding-3-small"` (DEFAULT_CONFIG struct) |

**Action:** Define once (e.g., `DEFAULT_EMBEDDING_MODEL` in `util` or `config`) and reference from both.

### 4b. `0o644` file permissions — 7 occurrences across 5 files

| File | Line |
|------|------|
| `src/cli/cli.odin` | 828, 860 |
| `src/graph/stream.odin` | 46 |
| `src/gnn/model.odin` | 754 |
| `src/graph/io.odin` | 7, 234 |
| `src/logger/logger.odin` | 44 |

**Action:** Define `DEFAULT_FILE_MODE :: 0o644` in `util` and reference everywhere.

### 4c. `0.75` limbo threshold — still hardcoded in 1 place

Already defined as `util.LIMBO_THRESHOLD`, but `src/ingest/ingest.odin:35` still uses the raw literal `0.75`.

**Action:** Replace with `util.LIMBO_THRESHOLD`.

### 4d. `.strand` extension — still hardcoded in 2 places

Already defined as `util.STRAND_EXTENSION`, but:

| File | Line | Context |
|------|------|---------|
| `src/cli/cli.odin` | 823 | `fmt.tprintf("%s.strand", name)` |
| `src/config/config.odin` | 57 | `graph_path = "knod.strand"` (default config) |

**Action:** Use `util.STRAND_EXTENSION` via `fmt.tprintf("%s%s", name, util.STRAND_EXTENSION)` for cli. The config default is acceptable as a complete filename.

---

## Summary

| Category | Count | Impact |
|----------|-------|--------|
| Dead functions | 9 | Remove ~200 lines |
| `home_dir`/`ensure_dir` wrappers | 9 wrappers | Remove wrappers, use `util.` directly |
| `write_val` duplicate | 1 extra copy | Remove from cli, import from graph |
| `save_graph` duplicate | 2 copies | Extract shared helper |
| `adamw_step` duplicate | 2 copies | Unify into 1 parameterized proc |
| Cosine similarity duplicate | 2 copies | Extract to util |
| Relevance score computation | 4 copies | Extract helper in gnn |
| Insertion sort | 9 copies | Extract generic sort util |
| Duplicated constants | 4 patterns | Centralize in util |
