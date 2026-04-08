"""GNN trainer — async retrain after each commit (link prediction + ranking loss)."""

import torch
import torch.nn.functional as F

from ..config import Config
from .gnn import ShardMPNN, StrandLayer


class GNNTrainer:
	"""Training loop with edge masking, link prediction, and adaptive LR.

	Retrained async after each commit (SP_GRAPH → SP_GNN in the flow).
	Base MPNN uses a fixed low LR (1e-5); StrandLayer uses adaptive LR.
	"""

	def __init__(self, model: ShardMPNN, strand: StrandLayer, cfg: Config):
		self.model = model
		self.strand = strand
		self.cfg = cfg
		self.optimizer = torch.optim.AdamW(
			[
				{"params": list(model.parameters()), "lr": 1e-5},  # base: fixed low LR
				{"params": list(strand.parameters()), "lr": cfg.lr_max},  # strand: adaptive LR
			],
			betas=(0.9, 0.999),
			weight_decay=cfg.weight_decay,
			eps=1e-8,
		)

	def adaptive_lr(self, num_thoughts: int) -> float:
		t = min(num_thoughts / 1024.0, 1.0)
		return self.cfg.lr_max * (1 - t) + self.cfg.lr_min * t

	def adaptive_steps(self, num_thoughts: int) -> int:
		t = min(num_thoughts / 1024.0, 1.0)
		return max(2, int(10 * (1 - t) + 2 * t))

	def train_step(self, node_features, edge_index, edge_features, num_thoughts: int):
		"""One training step with edge masking + link prediction loss."""
		self.model.train()
		self.strand.train()

		# Adaptive LR: only strand group (index 1) adapts; base (index 0) stays at 1e-5
		lr = self.adaptive_lr(num_thoughts)
		self.optimizer.param_groups[1]["lr"] = lr

		num_edges = edge_index.shape[1]
		if num_edges == 0:
			return 0.0

		# Edge masking: hide 15% of edges
		num_mask = max(1, int(num_edges * self.cfg.edge_mask_ratio))
		mask_indices = torch.randperm(num_edges)[:num_mask]
		keep_mask = torch.ones(num_edges, dtype=torch.bool)
		keep_mask[mask_indices] = False

		visible_edge_index = edge_index[:, keep_mask]
		visible_edge_features = edge_features[keep_mask]

		# Forward with visible edges
		hidden, scores = self.model(node_features, visible_edge_index, visible_edge_features)
		hidden, scores = self.strand(hidden, visible_edge_index)

		# Link prediction loss: masked edges should score high
		masked_src = edge_index[0, mask_indices]
		masked_dst = edge_index[1, mask_indices]
		pos_scores = (hidden[masked_src] * hidden[masked_dst]).sum(dim=-1)

		# Negative samples: random pairs
		neg_dst = torch.randint(0, node_features.shape[0], (num_mask,))
		neg_scores = (hidden[masked_src] * hidden[neg_dst]).sum(dim=-1)

		# Margin-based loss
		loss = F.relu(self.cfg.margin - pos_scores + neg_scores).mean()

		# Relevance ranking loss on scores
		if scores.shape[0] > 1:
			score_std = scores.std()
			if score_std > 0:
				loss = loss + 0.1 * (1.0 / (score_std + 1e-6))

		self.optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(
			list(self.model.parameters()) + list(self.strand.parameters()),
			max_norm=1.0,
		)
		self.optimizer.step()

		return loss.item()

	def train_on_graph(self, graph) -> float:
		"""Run adaptive training steps on the full graph.

		Returns average loss. Stores routing knowledge in self.last_routing
		for the caller to extract and merge into the base model.
		"""
		from .graph import Graph
		from .store import extract_routing_from_strand

		assert isinstance(graph, Graph)

		self.last_routing = None

		if graph.num_thoughts < 2 or graph.num_edges == 0:
			return 0.0

		node_features, edge_index, edge_features, ordered_ids, valid_edges = graph.to_tensors()

		if len(valid_edges) == 0:
			return 0.0

		steps = self.adaptive_steps(graph.num_thoughts)
		total_loss = 0.0
		for _ in range(steps):
			total_loss += self.train_step(node_features, edge_index, edge_features, graph.num_thoughts)

		self.last_routing = extract_routing_from_strand(graph, self.model, self.strand)

		return total_loss / steps

	def train_on_graph_with_routing(self, graph) -> tuple[float, dict]:
		"""Run training and return (loss, routing_dict).

		Convenience wrapper for callers that need both loss and routing.
		"""
		loss = self.train_on_graph(graph)
		return loss, self.last_routing or {}
