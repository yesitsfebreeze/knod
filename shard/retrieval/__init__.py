"""RETRIEVAL — fan-out across all strands, merge signals, generate answer.

score.py  — Q_S1 GNN, Q_S2 cosine, Q_S3 edge scoring
merge.py  — Q_WGT adaptive weights, Q_BST access boost, Q_DED dedup
rate.py   — Q_RATE final re-ranking pass (direct cosine blend)
expand.py — Dijkstra path traversal with cumulative path scoring
answer.py — Q_CTX context assembly + access tracking, Q_LLM answer generation
"""

from .score import cosine_scores, edge_scores, gnn_scores
from .merge import merge, deduplicate, best_chains_from
from .rate import rate_thoughts
from .expand import expand, PathChain
from .answer import answer, synthesize_direct

__all__ = [
	"cosine_scores",
	"edge_scores",
	"gnn_scores",
	"merge",
	"deduplicate",
	"best_chains_from",
	"rate_thoughts",
	"expand",
	"PathChain",
	"answer",
	"synthesize_direct",
]
