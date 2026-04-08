import json
import logging
import numpy as np
from openai import OpenAI, APIStatusError, APITimeoutError, APIConnectionError, RateLimitError, AuthenticationError

from .config import Config

log = logging.getLogger(__name__)

_FALLBACK_ERRORS = (RateLimitError, APITimeoutError, APIConnectionError, APIStatusError, AuthenticationError)


QUERY_TOOL_NAMES = {"ask", "find_thoughts", "explore_thought", "traverse", "graph_stats", "list_shards"}

TOOL_DEFINITIONS = [
	{
		"type": "function",
		"function": {
			"name": "ask",
			"description": "Ask a question against the knowledge graph. Returns an answer grounded in stored thoughts.",
			"parameters": {
				"type": "object",
				"properties": {
					"query": {"type": "string", "description": "The question to ask"},
				},
				"required": ["query"],
			},
		},
	},
	{
		"type": "function",
		"function": {
			"name": "ingest",
			"description": "Ingest text into the knowledge graph. The text is decomposed into atomic thoughts, linked, and stored.",
			"parameters": {
				"type": "object",
				"properties": {
					"text": {"type": "string", "description": "Text to ingest"},
					"source": {"type": "string", "description": "Source of the text (e.g., 'user', 'web')", "default": ""},
					"descriptor": {"type": "string", "description": "Optional descriptor to guide decomposition", "default": ""},
				},
				"required": ["text"],
			},
		},
	},
	{
		"type": "function",
		"function": {
			"name": "find_thoughts",
			"description": "Search for thoughts semantically similar to the query. Returns top-k matches without generating an LLM answer.",
			"parameters": {
				"type": "object",
				"properties": {
					"query": {"type": "string", "description": "Search query"},
					"k": {"type": "integer", "description": "Number of results", "default": 5},
				},
				"required": ["query"],
			},
		},
	},
	{
		"type": "function",
		"function": {
			"name": "set_purpose",
			"description": "Set the purpose / focus of the knowledge graph.",
			"parameters": {
				"type": "object",
				"properties": {
					"purpose": {"type": "string", "description": "Purpose description"},
				},
				"required": ["purpose"],
			},
		},
	},
	{
		"type": "function",
		"function": {
			"name": "add_descriptor",
			"description": "Add a descriptor (named context hint) to guide future ingestion.",
			"parameters": {
				"type": "object",
				"properties": {
					"name": {"type": "string", "description": "Descriptor name"},
					"description": {"type": "string", "description": "Descriptor text"},
				},
				"required": ["name", "description"],
			},
		},
	},
	{
		"type": "function",
		"function": {
			"name": "remove_descriptor",
			"description": "Remove a descriptor by name.",
			"parameters": {
				"type": "object",
				"properties": {
					"name": {"type": "string", "description": "Descriptor name"},
				},
				"required": ["name"],
			},
		},
	},
	{
		"type": "function",
		"function": {
			"name": "explore_thought",
			"description": "Explore a single thought: see its text, edges, neighbors, and reasoning.",
			"parameters": {
				"type": "object",
				"properties": {
					"thought_id": {"type": "integer", "description": "Thought ID"},
				},
				"required": ["thought_id"],
			},
		},
	},
	{
		"type": "function",
		"function": {
			"name": "traverse",
			"description": "Walk the graph from a starting thought via BFS. Returns the local subgraph.",
			"parameters": {
				"type": "object",
				"properties": {
					"start_id": {"type": "integer", "description": "Starting thought ID"},
					"depth": {"type": "integer", "description": "Max depth", "default": 2},
					"max_nodes": {"type": "integer", "description": "Max nodes", "default": 50},
				},
				"required": ["start_id"],
			},
		},
	},
	{
		"type": "function",
		"function": {
			"name": "graph_stats",
			"description": "Get aggregate statistics for the knowledge graph.",
			"parameters": {
				"type": "object",
				"properties": {},
			},
		},
	},
	{
		"type": "function",
		"function": {
			"name": "list_shards",
			"description": "List all loaded shards with their purpose, counts, and descriptors.",
			"parameters": {
				"type": "object",
				"properties": {},
			},
		},
	},
	{
		"type": "function",
		"function": {
			"name": "ingest_sync",
			"description": "Ingest text synchronously and return detailed results.",
			"parameters": {
				"type": "object",
				"properties": {
					"text": {"type": "string", "description": "Text to ingest"},
					"source": {"type": "string", "description": "Source", "default": ""},
					"descriptor": {"type": "string", "description": "Descriptor", "default": ""},
				},
				"required": ["text"],
			},
		},
	},
	{
		"type": "function",
		"function": {
			"name": "relink",
			"description": "Scan all existing thoughts and create missing edges between similar pairs.",
			"parameters": {
				"type": "object",
				"properties": {},
			},
		},
	},
]


class Provider:
	def __init__(self, cfg: Config):
		self._cfg = cfg
		self._use_openai = bool(cfg.openai_api_key)
		self._use_local = bool(cfg.local_base_url)

		if self._use_openai:
			self._openai = OpenAI(api_key=cfg.openai_api_key, base_url=cfg.openai_base_url)
		else:
			self._openai = None

		if self._use_local:
			self._local = OpenAI(api_key=cfg.local_api_key, base_url=cfg.local_base_url)
		else:
			self._local = None

		self.chat_model = cfg.openai_model
		self.embedding_model = cfg.openai_model
		self.embedding_dim = cfg.embedding_dim
		self._embed_provider: str | None = None  # None = not yet probed

	def _chat(self, **kwargs) -> "openai.types.chat.ChatCompletion":
		if self._use_openai:
			try:
				return self._openai.chat.completions.create(**kwargs)
			except _FALLBACK_ERRORS as exc:
				log.warning("OpenAI failed (%s), trying local", exc)

		if self._local is not None:
			messages = kwargs.pop("messages", [])
			kwargs["model"] = self._cfg.local_model or self.chat_model
			return self._local.chat.completions.create(messages=messages, **kwargs)

		raise RuntimeError("No providers available: configure openai_api_key or local_base_url")

	def _probe_embed(self) -> None:
		"""Status-check each embedding tier in order; cache the first working one."""
		probe = "probe"
		log.info(
			"Probing embed — openai: use=%s key=%s model=%s | local: use=%s url=%s model=%s",
			self._use_openai,
			(self._cfg.openai_api_key[:12] + "...") if self._cfg.openai_api_key else "UNSET",
			self._cfg.openai_embedding_model or "UNSET",
			self._use_local,
			self._cfg.local_base_url or "UNSET",
			self._cfg.local_embedding_model or self._cfg.local_model or "UNSET",
		)
		if self._openai is not None:
			try:
				self._openai.embeddings.create(model=self._cfg.openai_embedding_model, input=probe)
				self._embed_provider = "openai"
				log.info("Embedding provider: openai (%s)", self._cfg.openai_embedding_model)
				return
			except Exception as e:
				log.warning("OpenAI embedding unavailable (%s), trying local", e)
				self._openai = None

		if self._local is not None and self._use_local:
			local_model = self._cfg.local_embedding_model or self._cfg.local_model
			if local_model:
				try:
					self._local.embeddings.create(model=local_model, input=probe)
					self._embed_provider = "local"
					log.info("Embedding provider: local (%s)", local_model)
					return
				except Exception as e:
					log.warning("Local embedding unavailable (%s)", e)
			else:
				log.warning("Local embedding skipped — no local_embedding_model or local_model set")
		else:
			log.warning("Local embedding skipped — _local=%s _use_local=%s", self._local, self._use_local)

		self._embed_provider = ""
		raise RuntimeError(
			"No embedding provider available — configure openai_api_key+openai_embedding_model "
			"or local_base_url+local_embedding_model"
		)

	def embed_text(self, text: str) -> np.ndarray:
		if self._embed_provider is None:
			self._probe_embed()
		if self._embed_provider == "openai":
			resp = self._openai.embeddings.create(model=self._cfg.openai_embedding_model, input=text)
			return np.array(resp.data[0].embedding, dtype=np.float32)
		if self._embed_provider == "local":
			model = self._cfg.local_embedding_model or self._cfg.local_model
			resp = self._local.embeddings.create(model=model, input=text)
			return np.array(resp.data[0].embedding, dtype=np.float32)
		raise RuntimeError("No embedding provider available")

	def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
		if not texts:
			return []
		if self._embed_provider is None:
			self._probe_embed()
		if self._embed_provider == "openai":
			resp = self._openai.embeddings.create(model=self._cfg.openai_embedding_model, input=texts)
			return [np.array(d.embedding, dtype=np.float32) for d in sorted(resp.data, key=lambda d: d.index)]
		if self._embed_provider == "local":
			model = self._cfg.local_embedding_model or self._cfg.local_model
			resp = self._local.embeddings.create(model=model, input=texts)
			return [np.array(d.embedding, dtype=np.float32) for d in sorted(resp.data, key=lambda d: d.index)]
		raise RuntimeError("No embedding provider available")

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
		content = resp.choices[0].message.content.strip()
		if content.startswith("```"):
			content = content.split("```")[1]
			if content.startswith("json"):
				content = content[4:]
		content = content.strip()
		data = json.loads(content)
		if isinstance(data, list):
			return data
		for key in ("thoughts", "items", "statements", "facts"):
			if key in data and isinstance(data[key], list):
				return data[key]
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
		content = resp.choices[0].message.content.strip()
		if content.startswith("```"):
			content = content.split("```")[1]
			if content.startswith("json"):
				content = content[4:]
		content = content.strip()
		data = json.loads(content)
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
						"Given a cluster of related thoughts, suggest a short name "
						"that describes the content and a one-sentence purpose. "
						'Return JSON: {"name": "...", "purpose": "..."}'
					),
				},
				{"role": "user", "content": f"Thoughts:\n{combined}"},
			],
			response_format={"type": "json_object"},
		)
		data = json.loads(resp.choices[0].message.content)
		return data.get("name", "unnamed"), data.get("purpose", "")

	def chat(self, messages: list[dict], **kwargs) -> str:
		resp = self._chat(model=self.chat_model, messages=messages, **kwargs)
		return resp.choices[0].message.content

	def chat_with_tools(self, messages: list[dict], tool_handler, tools: list[dict] | None = None, **kwargs) -> tuple[str, list[dict]]:
		tool_calls = []
		max_turns = 10
		turn = 0
		active_tools = tools if tools is not None else TOOL_DEFINITIONS

		while turn < max_turns:
			turn += 1
			resp = self._chat(
				model=self.chat_model,
				messages=messages,
				tools=active_tools,
				**kwargs,
			)

			# Unified OpenAI-style path (_anthropic_chat wraps Anthropic responses in OpenAIMessage)
			choices = resp.choices
			if not choices:
				return "", tool_calls
			msg = choices[0].message
			if not hasattr(msg, "tool_calls") or not msg.tool_calls:
				return msg.content, tool_calls

			# Assistant message with tool_calls must precede tool results (OpenAI requirement)
			messages.append({
				"role": "assistant",
				"content": msg.content,
				"tool_calls": [
					{
						"id": tc.id,
						"type": "function",
						"function": {"name": tc.function.name, "arguments": tc.function.arguments},
					}
					for tc in msg.tool_calls
				],
			})

			# Process tool calls
			for tc in msg.tool_calls:
				tool_name = tc.function.name
				tool_args = json.loads(tc.function.arguments)
				try:
					result = self._call_tool(tool_name, tool_args, tool_handler)
					tool_calls.append({"name": tool_name, "result": result})
					messages.append(
						{
							"role": "tool",
							"tool_call_id": tc.id,
							"content": json.dumps(result),
						}
					)
				except Exception as e:
					messages.append(
						{
							"role": "tool",
							"tool_call_id": tc.id,
							"content": f"Error: {str(e)}",
						}
					)

		return "Max tool turns exceeded", tool_calls

	def _call_tool(self, name: str, args: dict, handler) -> dict:
		method = getattr(handler, name, None)
		if method is None:
			raise ValueError(f"Unknown tool: {name}")
		result = method(**args)
		if isinstance(result, str):
			return {"result": result}
		return result


