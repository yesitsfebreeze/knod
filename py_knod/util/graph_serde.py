from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from ..specialist.graph import Graph


def graph_to_state(graph: "Graph", *, include_limbo: bool = False) -> dict:
	state = {
		"name": graph.name,
		"purpose": graph.purpose,
		"descriptors": graph.descriptors,
		"next_id": graph._next_id,
		"profile": graph._profile,
		"registry_nodes": graph._registry_nodes,
		"max_thoughts": graph.max_thoughts,
		"max_edges": graph.max_edges,
		"thoughts": {tid: t.to_dict() for tid, t in graph.thoughts.items()},
		"edges": [e.to_dict() for e in graph.edges],
	}
	if include_limbo:
		state["limbo"] = [lt.to_dict() for lt in graph.limbo]
	return state


def graph_from_state(state: dict, *, maturity_divisor: int = 50) -> "Graph":
	from ..specialist.graph import Graph, Thought, Edge, LimboThought

	graph = Graph(name=state.get("name", ""), purpose=state["purpose"])
	graph.descriptors = state.get("descriptors", {})
	graph._next_id = state["next_id"]
	graph._profile = state.get("profile")
	graph._registry_nodes = state.get("registry_nodes", {})
	graph.max_thoughts = state.get("max_thoughts", 0)
	graph.max_edges = state.get("max_edges", 0)
	graph.maturity_divisor = maturity_divisor

	for tid_str, tdata in state["thoughts"].items():
		tid = int(tid_str) if isinstance(tid_str, str) else tid_str
		graph.thoughts[tid] = Thought.from_dict(tid, tdata)

	for edata in state["edges"]:
		graph.edges.append(Edge.from_dict(edata))

	for ldata in state.get("limbo", []):
		graph.limbo.append(LimboThought.from_dict(ldata))

	return graph
