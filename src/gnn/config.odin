package gnn


EMBEDDING_DIM :: 1536 // must match graph.EMBEDDING_DIM

// Architecture defaults
DEFAULT_HIDDEN_DIM :: 512
DEFAULT_NUM_LAYERS :: 3

// AdamW
BETA1: f32 : 0.9
BETA2: f32 : 0.999
ADAM_EPS: f32 : 1e-8
WEIGHT_DECAY: f32 : 0.01

// Adaptive learning rate
ADAPT_LR_MAX: f32 : 1e-3
ADAPT_LR_MIN: f32 : 5e-5

// Training objective
LINK_PRED_WEIGHT: f32 : 1.0
RELEVANCE_WEIGHT: f32 : 1.0
MARGIN: f32 : 0.1

// Masking
EDGE_MASK_RATIO: f32 : 0.15

// Base MPNN checkpoint (~/.config/knod/knod, no extension)
GNN_MAGIC :: 0x6B6E6E67 // "knng"
GNN_VERSION :: i32(1)

// Strand MPNN checkpoint (SECTION_STRAND inside <name>.strand)
STRAND_MAGIC :: 0x6B6E7374 // "knst"
STRAND_VERSION :: i32(1)

// Training steps per ingestion event
STRAND_TRAIN_STEPS :: 10 // strand-only (base frozen), LR=ADAPT_LR_MAX
BASE_REFINE_STEPS :: 5 // base-only (strand frozen), LR=BASE_LR
BASE_LR: f32 : 1e-5

// RNG
RNG_SEED: u64 : 42
