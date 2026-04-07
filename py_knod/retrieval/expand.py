"""Retrieval · Expand — semantic path traversal with cumulative path scoring.

Replaces the original BFS fan-out with Dijkstra-based path search that:

  1. Scores *paths* as a unit, not individual nodes.
     path_score = product(edge.weight × edge_reasoning_cosine for each hop)
                × terminal_thought_cosine
     Any weak link along the path suppresses the whole path — this prevents
     noise from leaking through via high-weight but irrelevant intermediate edges.

  2. Supports endpoint-aware search (Change 2).
     When `targets` are provided (distant high-scoring thoughts that didn't
     make it into seeds), the algorithm prioritises paths that terminate at
     a known target thought.  This enables multi-hop chaining between a seed
     and a semantically relevant but structurally disconnected thought.

  3. Returns PathChain objects (Change 3) alongside scored thoughts.
     A PathChain preserves the ordered sequence of (thought, edge_reasoning)
     pairs so the LLM can receive the chain as a narrative rather than a bag
     of isolated facts.

Algorithm
---------
  Dijkstra on a graph where edge *cost* = 1 - (edge.weight × edge_reasoning_cosine).
  Lower cost = stronger, more query-relevant edge.

  For each seed, we run a bounded shortest-path search:
    - priority queue ordered by cumulative path cost (min-heap)
    - each entry: (cum_cost, depth, current_id, path_so_far)
      where path_so_far = list of (thought_id, edge_reasoning) pairs
    - at each step, emit a PathChain when we reach a new node
    - stop at depth_limit hops or when fan_out_remaining budget is exhausted

PathChain
---------
  A dataclass carrying:
    steps  — ordered list of (Thought, edge_reasoning_str) pairs
             First step has no incoming edge (reasoning = "")
    score  — cumulative path score for ranking
    is_target_path — True when the terminal node was a supplied target
"""

import heapq
from dataclasses import dataclass, field

import time as _time

import numpy as np

from ..config import Config
from ..specialist.graph import Edge, Graph, Thought
from ..util.math import cosine


@dataclass
class PathChain:
	"""An ordered chain of thoughts connected by named reasoning steps."""

	steps: list[tuple[Thought, str]] = field(default_factory=list)
	score: float = 0.0
	is_target_path: bool = False
	traversed_edges: list[Edge] = field(default_factory=list)

	@property
	def terminal(self) -> Thought | None:
		return self.steps[-1][0] if self.steps else None

	@property
	def thoughts(self) -> list[Thought]:
		return [t for t, _ in self.steps]


def _edge_cost(edge, query_emb: np.ndarray) -> float:
	"""Edge traversal cost = 1 - (weight × edge_reasoning_cosine).

	Lower cost → stronger, more query-relevant edge.
	Clamped to [0, 1] so the cost is always a valid probability complement.
	"""
	edge_cos = max(cosine(edge.embedding, query_emb), 0.0)
	relevance = edge.weight * edge_cos
	return 1.0 - min(relevance, 1.0)


def expand(
	seeds: list[tuple[Thought, float]],
	query_emb: np.ndarray,
	graph: Graph,
	cfg: Config,
	targets: set[int] | None = None,
) -> tuple[list[tuple[Thought, float]], list[PathChain]]:
	"""Expand seed thoughts using Dijkstra cumulative path scoring.

	Args:
	    seeds:      Scored seed thoughts from merge(), each (Thought, score).
	    query_emb:  Query embedding (raw, not normalised).
	    graph:      The specialist graph to traverse.
	    cfg:        Config carrying traversal_depth and traversal_fan_out.
	    targets:    Optional set of thought IDs to treat as high-value endpoints.
	                Paths reaching a target get an is_target_path=True marker and
	                a small score bonus, ensuring they survive ranking.

	Returns:
	    (scored_thoughts, path_chains)

	    scored_thoughts — seeds ∪ discovered nodes as (Thought, score) pairs.
	                      Seeds retain their original scores.
	                      Discovered nodes get their best cumulative path score.
	    path_chains     — PathChain objects for each discovered node, ordered
	                      by score descending.  Includes seed-only single-step
	                      chains so every thought in scored_thoughts has a chain.
	"""
	if not seeds:
		return [], []

	depth_limit = cfg.traversal_depth
	fan_out = cfg.traversal_fan_out

	if fan_out <= 0 or depth_limit <= 0:
		# No expansion — wrap seeds in trivial single-step chains
		chains = [PathChain(steps=[(t, "")], score=s) for t, s in seeds]
		return list(seeds), chains

	targets = targets or set()

	# result: thought_id → (Thought, best_score)
	result: dict[int, tuple[Thought, float]] = {t.id: (t, s) for t, s in seeds}
	# best_chain: thought_id → PathChain with highest score seen so far
	best_chain: dict[int, PathChain] = {
		t.id: PathChain(steps=[(t, "")], score=s, is_target_path=(t.id in targets)) for t, s in seeds
	}

	visited_from: dict[int, set[int]] = {t.id: set() for t, _ in seeds}
	fan_out_remaining = fan_out

	# Min-heap: (cumulative_cost, depth, current_id, path_steps)
	# path_steps: list of (thought_id, edge_reasoning, edge_or_None)
	heap: list[tuple[float, int, int, list[tuple[int, str, Edge | None]]]] = []
	for seed_thought, _ in seeds:
		heap.append((0.0, 0, seed_thought.id, [(seed_thought.id, "", None)]))
	heapq.heapify(heap)

	now = _time.time()

	while heap and fan_out_remaining > 0:
		cum_cost, depth, current_id, path_steps = heapq.heappop(heap)

		if depth >= depth_limit:
			continue

		for neighbour_id, edge in graph.get_neighbors(current_id):
			if neighbour_id not in graph.thoughts:
				continue

			# Track which seeds have visited this neighbour to allow
			# convergent paths from different seeds.
			seed_origin = path_steps[0][0]
			already_visited = visited_from.setdefault(seed_origin, set())
			if neighbour_id in already_visited:
				continue
			if fan_out_remaining <= 0:
				break

			neighbour = graph.thoughts[neighbour_id]
			e_cost = _edge_cost(edge, query_emb)
			new_cum_cost = cum_cost + e_cost

			# Convert cumulative cost back to a score:
			# score = (1 - avg_edge_cost) × terminal_thought_cosine
			# avg_edge_cost = new_cum_cost / num_hops
			num_hops = depth + 1
			avg_edge_cost = new_cum_cost / num_hops
			thought_cos = max(cosine(neighbour.embedding, query_emb), 0.0)
			path_score = (1.0 - avg_edge_cost) * thought_cos

			# Target bonus: ensure target-reaching paths compete strongly
			is_target = neighbour_id in targets
			if is_target:
				path_score = min(path_score + 0.15, 1.0)

			already_visited.add(neighbour_id)
			fan_out_remaining -= 1

			# Record edge traversal
			edge.traversal_count += 1
			edge.last_traversed = now

			new_path = path_steps + [(neighbour_id, edge.reasoning, edge)]

			# Keep best score per thought across all paths
			if neighbour_id not in result or path_score > result[neighbour_id][1]:
				result[neighbour_id] = (neighbour, path_score)
				# Collect all edges traversed in this path
				path_edges = [e for _, _, e in new_path if e is not None]
				best_chain[neighbour_id] = PathChain(
					steps=[(graph.thoughts[sid], rsn) for sid, rsn, _ in new_path if sid in graph.thoughts],
					score=path_score,
					is_target_path=is_target,
					traversed_edges=path_edges,
				)

			# Push for further expansion
			heapq.heappush(heap, (new_cum_cost, depth + 1, neighbour_id, new_path))

	scored_thoughts = list(result.values())
	path_chains = sorted(best_chain.values(), key=lambda c: c.score, reverse=True)
	return scored_thoughts, path_chains
