"""Retrieval · Answer — assemble top-k context and generate LLM answer.

Matches FLOW.md:
  Q_CTX — Assemble top-k context (also updates access_count + last_accessed)
  Q_LLM — LLM: generate answer
  Q_OUT — Answer + ranked sources
"""

import time as _time

from ..provider import Provider
from ..specialist.graph import Thought


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
) -> tuple[str, list[dict]]:
	"""Assemble context from top-k thoughts, update access tracking, generate answer.

	Returns (answer_text, sources_list).
	"""
	context_parts, sources = _build_sources(scored)
	context = "\n\n".join(context_parts)
	answer_text = provider.generate_answer(query, context)
	return answer_text, sources
