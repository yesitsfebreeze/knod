"""RETRIEVAL — fan-out across all specialists, merge signals, generate answer.

score.py  — Q_S1 GNN, Q_S2 cosine, Q_S3 edge scoring
merge.py  — Q_WGT adaptive weights, Q_BST access boost, Q_DED dedup
answer.py — Q_CTX context assembly + access tracking, Q_LLM answer generation
"""

from .score import cosine_scores, edge_scores, gnn_scores
from .merge import merge, deduplicate
from .answer import answer, synthesize_direct

__all__ = [
	"cosine_scores",
	"edge_scores",
	"gnn_scores",
	"merge",
	"deduplicate",
	"answer",
	"synthesize_direct",
]
