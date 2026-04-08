"""Retrieval · Answer — assemble top-k context and generate LLM answer.

Matches FLOW.md:
  Q_CTX — Assemble top-k context (also updates access_count + last_accessed)
  Q_LLM — LLM: generate answer
  Q_OUT — Answer + ranked sources

Change 3: Path-aware context assembly.

When PathChain objects are available from expand(), the context sent to the
LLM is structured as ordered reasoning chains rather than a flat bag of
thoughts.  Each multi-hop chain is serialised as:

    [thought A] → <edge reasoning> → [thought B] → <edge reasoning> → [thought C]

This lets the LLM reason along the chain rather than treating each fact as
independent.  Single-step chains (seed thoughts with no traversal) are
rendered as plain facts, preserving backward compatibility.

The ordering strategy:
  1. Multi-hop target chains first (is_target_path=True, len > 1) — these
     are the most valuable for complex questions.
  2. Multi-hop non-target chains (len > 1).
  3. Single-step thoughts (direct matches), in score order.

Any thought that appears in a chain is NOT repeated in the flat facts section,
preventing duplication.
"""

import time as _time
from typing import TYPE_CHECKING

from ..provider import Provider
from ..strand.graph import Thought

if TYPE_CHECKING:
	from .expand import PathChain


def _build_sources(scored: list[tuple[Thought, float]]) -> tuple[list[str], list[dict]]:
	"""Update access tracking and build context parts + sources list."""
	sources = []
	context_parts = []

	for thought, score in scored:
		thought.access_count += 1
		thought.last_accessed = _time.time()
		context_parts.append(thought.text)
		sources.append(
			{
				"id": thought.id,
				"text": thought.text,
				"similarity": round(score, 3),
				"source": thought.source,
			}
		)

	return context_parts, sources


def _render_chain(chain: "PathChain") -> str:
	"""Render a PathChain as a readable reasoning chain string.

	Single-step chains: just the thought text.
	Multi-step chains: thought → reasoning → thought → ...
	"""
	if len(chain.steps) <= 1:
		return chain.steps[0][0].text if chain.steps else ""

	parts = []
	for i, (thought, reasoning) in enumerate(chain.steps):
		parts.append(thought.text)
		if i < len(chain.steps) - 1 and reasoning:
			parts.append(f"  → ({reasoning}) →")
		elif i < len(chain.steps) - 1:
			parts.append("  →")
	return "\n".join(parts)


def _build_path_context(
	scored: list[tuple[Thought, float]],
	chains: list["PathChain"],
) -> tuple[str, list[dict]]:
	"""Build path-aware context from chains + flat fallback for uncovered thoughts.

	Returns (context_str, sources_list).
	Access tracking is updated for all thoughts that appear in context.
	"""
	now = _time.time()
	sources = []
	seen_ids: set[int] = set()
	context_sections: list[str] = []

	# Sort chains: target multi-hop first, then other multi-hop, then single-step
	multi_target = [c for c in chains if c.is_target_path and len(c.steps) > 1]
	multi_other = [c for c in chains if not c.is_target_path and len(c.steps) > 1]
	single_step = [c for c in chains if len(c.steps) <= 1]

	ordered_chains = multi_target + multi_other + single_step

	if ordered_chains:
		chain_parts = []
		for chain in ordered_chains:
			if not chain.steps:
				continue
			rendered = _render_chain(chain)
			chain_parts.append(rendered)
			chain_used = False
			for thought, _ in chain.steps:
				if thought.id not in seen_ids:
					seen_ids.add(thought.id)
					thought.access_count += 1
					thought.last_accessed = now
					chain_used = True
					# Find score from scored list for this thought
					score = next((s for t, s in scored if t.id == thought.id), chain.score)
					sources.append(
						{
							"id": thought.id,
							"text": thought.text,
							"similarity": round(score, 3),
							"source": thought.source,
						}
					)
			# Mark edges in this chain as successful if the chain contributed context
			if chain_used and hasattr(chain, "traversed_edges"):
				for edge in chain.traversed_edges:
					edge.success_count += 1
		if chain_parts:
			context_sections.append("Reasoning chains:\n" + "\n\n".join(chain_parts))

	# Any scored thoughts not covered by chains go in as flat facts
	flat_parts = []
	for thought, score in scored:
		if thought.id not in seen_ids:
			seen_ids.add(thought.id)
			thought.access_count += 1
			thought.last_accessed = now
			flat_parts.append(thought.text)
			sources.append(
				{
					"id": thought.id,
					"text": thought.text,
					"similarity": round(score, 3),
					"source": thought.source,
				}
			)
	if flat_parts:
		context_sections.append("Supporting facts:\n" + "\n\n".join(flat_parts))

	context = "\n\n---\n\n".join(context_sections)
	return context, sources


def synthesize_direct(
	scored: list[tuple[Thought, float]],
) -> tuple[str, list[dict]]:
	"""Return thought texts directly without LLM generation.

	Used when top scores exceed the confidence threshold — the graph
	already has high-quality knowledge and no LLM call is needed.
	"""
	context_parts, sources = _build_sources(scored)
	return "\n\n".join(context_parts), sources


def answer(
	query: str,
	scored: list[tuple[Thought, float]],
	provider: Provider,
	chains: "list[PathChain] | None" = None,
) -> tuple[str, list[dict]]:
	"""Assemble context from top-k thoughts, update access tracking, generate answer.

	When `chains` are provided, context is structured as ordered reasoning
	chains so the LLM can follow multi-hop logic.  Falls back to flat context
	if no chains are available.

	Returns (answer_text, sources_list).
	"""
	if chains:
		context, sources = _build_path_context(scored, chains)
	else:
		context_parts, sources = _build_sources(scored)
		context = "\n\n".join(context_parts)

	answer_text = provider.generate_answer(query, context)
	return answer_text, sources
