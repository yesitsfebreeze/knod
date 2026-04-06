"""Base MPNN (3-layer message passing) + StrandLayer (per-specialist fine-tuning)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

from ..config import Config


class MPNNLayer(MessagePassing):
	"""Single message-passing layer: message → aggregate → update with gate."""

	def __init__(self, hidden_dim: int):
		super().__init__(aggr="add")
		# Message: concat [src, edge, dst] → hidden
		self.msg_mlp = nn.Linear(hidden_dim * 3, hidden_dim)
		# Update gate: concat [hidden, aggregated] → hidden
		self.update_mlp = nn.Linear(hidden_dim * 2, hidden_dim)
		self.norm = nn.LayerNorm(hidden_dim)

	def forward(self, x, edge_index, edge_attr):
		return self.propagate(edge_index, x=x, edge_attr=edge_attr)

	def message(self, x_i, x_j, edge_attr):
		# x_j = source, x_i = destination, edge_attr = edge features
		cat = torch.cat([x_j, edge_attr, x_i], dim=-1)
		return F.relu(self.msg_mlp(cat))

	def update(self, aggr_out, x):
		cat = torch.cat([x, aggr_out], dim=-1)
		h = F.relu(self.update_mlp(cat))
		return self.norm(h)


class KnodMPNN(nn.Module):
	"""Base MPNN: project → 3-layer message passing → score. Shared low LR."""

	def __init__(self, cfg: Config):
		super().__init__()
		self.embedding_dim = cfg.embedding_dim
		self.hidden_dim = cfg.hidden_dim

		# Projections
		self.node_proj = nn.Linear(cfg.embedding_dim, cfg.hidden_dim)
		self.edge_proj = nn.Linear(cfg.embedding_dim, cfg.hidden_dim)

		# Message-passing layers
		self.layers = nn.ModuleList([MPNNLayer(cfg.hidden_dim) for _ in range(cfg.num_layers)])

		# Scoring head
		self.score_head = nn.Linear(cfg.hidden_dim, 1)

		self._init_weights()

	def _init_weights(self):
		for p in self.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)

	def forward(self, node_features, edge_index, edge_features):
		"""
		Args:
			node_features: [N, embedding_dim]
			edge_index: [2, E]
			edge_features: [E, embedding_dim]
		Returns:
			hidden: [N, hidden_dim]
			scores: [N, 1]
		"""
		h = F.relu(self.node_proj(node_features))
		e = F.relu(self.edge_proj(edge_features))

		for layer in self.layers:
			h = layer(h, edge_index, e)

		scores = self.score_head(h)
		return h, scores


class StrandLayer(nn.Module):
	"""Per-specialist fine-tuning layer on top of base MPNN. Adaptive LR."""

	def __init__(self, hidden_dim: int):
		super().__init__()
		self.msg_mlp = nn.Linear(hidden_dim * 2, hidden_dim)
		self.update_mlp = nn.Linear(hidden_dim * 2, hidden_dim)
		self.norm = nn.LayerNorm(hidden_dim)
		self.score_head = nn.Linear(hidden_dim, 1)

	def forward(self, base_hidden, edge_index):
		# Simple one-layer specialization on base hidden states
		src, dst = edge_index
		if src.numel() == 0:
			return base_hidden, self.score_head(base_hidden)

		messages = torch.cat([base_hidden[src], base_hidden[dst]], dim=-1)
		messages = F.relu(self.msg_mlp(messages))

		# Scatter-add aggregation
		aggr = torch.zeros_like(base_hidden)
		aggr.index_add_(0, dst, messages)

		h = torch.cat([base_hidden, aggr], dim=-1)
		h = self.norm(F.relu(self.update_mlp(h)))

		scores = self.score_head(h)
		return h, scores
