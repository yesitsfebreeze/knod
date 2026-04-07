"""OpenAI API wrapper — embed, decompose, link, answer.

Uses the primary provider (OpenAI) for all calls.  When a chat completion
hits a rate-limit, timeout, or connection error *and* a local fallback
(Ollama) is configured, the request is retried against the fallback.
Embeddings always use the primary provider to keep dimensions consistent.
"""

import json
import logging
import numpy as np
from openai import OpenAI, APIStatusError, APITimeoutError, APIConnectionError, RateLimitError

from .config import Config

log = logging.getLogger(__name__)

_FALLBACK_ERRORS = (RateLimitError, APITimeoutError, APIConnectionError)


class Provider:
	def __init__(self, cfg: Config):
		self.client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
		self.embedding_model = cfg.embedding_model
		self.chat_model = cfg.chat_model
		self.embedding_dim = cfg.embedding_dim

		# Fallback client (Ollama or other local provider)
		if cfg.fallback_base_url:
			self._fallback = OpenAI(api_key=cfg.fallback_api_key, base_url=cfg.fallback_base_url)
			self._fallback_chat_model = cfg.fallback_chat_model or cfg.chat_model
		else:
			self._fallback = None
			self._fallback_chat_model = ""

	def _chat(self, **kwargs) -> "openai.types.chat.ChatCompletion":
		"""Try primary client; fall back to local on rate-limit / timeout."""
		try:
			return self.client.chat.completions.create(**kwargs)
		except _FALLBACK_ERRORS as exc:
			if self._fallback is None:
				raise
			log.warning("Primary provider failed (%s), falling back to local", exc)
			kwargs["model"] = self._fallback_chat_model
			return self._fallback.chat.completions.create(**kwargs)

	def embed_text(self, text: str) -> np.ndarray:
		resp = self.client.embeddings.create(model=self.embedding_model, input=text)
		return np.array(resp.data[0].embedding, dtype=np.float32)

	def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
		if not texts:
			return []
		resp = self.client.embeddings.create(model=self.embedding_model, input=texts)
		# API returns embeddings in order of index
		sorted_data = sorted(resp.data, key=lambda d: d.index)
		return [np.array(d.embedding, dtype=np.float32) for d in sorted_data]

	def decompose_text(self, text: str, purpose: str = "", descriptors: dict[str, str] | None = None) -> list[str]:
		system = (
			"You decompose text into atomic, self-contained thoughts. "
			"Each thought should be a single factual statement that stands alone."
		)
		if purpose:
			system += f"\n\nThis knowledge base focuses on: {purpose}"
		if descriptors:
			hints = "\n".join(f"- {k}: {v}" for k, v in descriptors.items())
			system += f"\n\nContext hints:\n{hints}"

		resp = self._chat(
			model=self.chat_model,
			messages=[
				{"role": "system", "content": system},
				{
					"role": "user",
					"content": (f"Decompose this text into atomic thoughts. Return a JSON array of strings.\n\n{text}"),
				},
			],
			response_format={"type": "json_object"},
		)
		data = json.loads(resp.choices[0].message.content)
		# Handle {"thoughts": [...]} or {"items": [...]} or just [...]
		if isinstance(data, list):
			return data
		for key in ("thoughts", "items", "statements", "facts"):
			if key in data and isinstance(data[key], list):
				return data[key]
		# Fallback: take first list value
		for v in data.values():
			if isinstance(v, list):
				return v
		return []

	def batch_link_reason(self, thought_text: str, candidate_texts: list[str]) -> list[dict]:
		if not candidate_texts:
			return []

		numbered = "\n".join(f"{i}: {t}" for i, t in enumerate(candidate_texts))
		resp = self._chat(
			model=self.chat_model,
			messages=[
				{
					"role": "system",
					"content": (
						"You evaluate semantic relationships between thoughts. "
						"For each candidate, decide if it's related to the given thought. "
						"Return a JSON object with a 'links' array. Each link has: "
						"'index' (int), 'weight' (0.0-1.0 relevance), 'reasoning' (brief explanation). "
						"Only include candidates with weight > 0.1."
					),
				},
				{"role": "user", "content": (f"Thought: {thought_text}\n\nCandidates:\n{numbered}")},
			],
			response_format={"type": "json_object"},
		)
		data = json.loads(resp.choices[0].message.content)
		links = data.get("links", [])
		return [
			{"index": l["index"], "weight": l["weight"], "reasoning": l["reasoning"]}
			for l in links
			if isinstance(l, dict) and "index" in l and "weight" in l and "reasoning" in l
		]

	def generate_answer(self, query: str, context: str) -> str:
		resp = self._chat(
			model=self.chat_model,
			messages=[
				{
					"role": "system",
					"content": (
						"You answer questions using the provided context. "
						"Be concise and accurate. If the context doesn't contain "
						"enough information, say so."
					),
				},
				{"role": "user", "content": (f"Context:\n{context}\n\nQuestion: {query}")},
			],
		)
		return resp.choices[0].message.content

	def suggest_store(self, thought_texts: list[str]) -> tuple[str, str]:
		combined = "\n".join(f"- {t}" for t in thought_texts[:20])
		resp = self._chat(
			model=self.chat_model,
			messages=[
				{
					"role": "system",
					"content": (
						"Given a cluster of related thoughts, suggest a specialist name "
						'and purpose. Return JSON: {"name": "...", "purpose": "..."}'
					),
				},
				{"role": "user", "content": f"Thoughts:\n{combined}"},
			],
			response_format={"type": "json_object"},
		)
		data = json.loads(resp.choices[0].message.content)
		return data.get("name", "unnamed"), data.get("purpose", "")
