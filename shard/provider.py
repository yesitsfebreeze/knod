import json
import logging
import numpy as np
from openai import OpenAI, APIStatusError, APITimeoutError, APIConnectionError, RateLimitError
from anthropic import Anthropic

from .config import Config

log = logging.getLogger(__name__)

_FALLBACK_ERRORS = (RateLimitError, APITimeoutError, APIConnectionError)


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
			"name": "list_Shards",
			"description": "List all loaded Shards with their purpose, counts, and descriptors.",
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
		self._use_anthropic = bool(cfg.anthropic_api_key)

		if self._use_anthropic:
			self._anthropic = Anthropic(api_key=cfg.anthropic_api_key, base_url=cfg.anthropic_base_url or None)
			self._openai = None
		else:
			self._anthropic = None
			self._openai = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)

		self.embedding_model = cfg.embedding_model
		self.chat_model = cfg.chat_model
		self.embedding_dim = cfg.embedding_dim

		if cfg.fallback_base_url:
			self._fallback = OpenAI(api_key=cfg.fallback_api_key, base_url=cfg.fallback_base_url)
			self._fallback_chat_model = cfg.fallback_chat_model or cfg.chat_model
		else:
			self._fallback = None
			self._fallback_chat_model = ""

	def _chat(self, **kwargs) -> "openai.types.chat.ChatCompletion":
		if self._use_anthropic:
			return self._anthropic_chat(**kwargs)
		return self._openai_chat(**kwargs)

	def _openai_chat(self, **kwargs) -> "openai.types.chat.ChatCompletion":
		try:
			return self._openai.chat.completions.create(**kwargs)
		except _FALLBACK_ERRORS as exc:
			if self._fallback is None:
				raise
			log.warning("Primary provider failed (%s), falling back to local", exc)
			kwargs["model"] = self._fallback_chat_model
			return self._fallback.chat.completions.create(**kwargs)

	def _anthropic_chat(self, **kwargs) -> "openai.types.chat.ChatCompletion":
		messages = kwargs.pop("messages", [])
		model = kwargs.pop("model", self.chat_model)
		system = ""
		user_messages = []

		for msg in messages:
			if msg.get("role") == "system":
				system = msg.get("content", "")
			else:
				user_messages.append(msg)

		anthropic_messages = [{"role": m["role"], "content": m["content"]} for m in user_messages]

		try:
			resp = self._anthropic.messages.create(
				model=model,
				max_tokens=4096,
				system=system,
				messages=anthropic_messages,
				**kwargs,
			)
			content = resp.content[0].text if resp.content else ""
			return OpenAIMessage(content=content, model=model)
		except Exception as exc:
			if self._fallback is None:
				raise
			log.warning("Anthropic failed (%s), falling back to local", exc)
			kwargs["model"] = self._fallback_chat_model
			return self._fallback.chat.completions.create(**kwargs)

	def embed_text(self, text: str) -> np.ndarray:
		if self._use_anthropic:
			raise NotImplementedError("Embedding not supported with Anthropic")
		resp = self._openai.embeddings.create(model=self.embedding_model, input=text)
		return np.array(resp.data[0].embedding, dtype=np.float32)

	def embed_texts(self, texts: list[str]) -> list[np.ndarray]:
		if self._use_anthropic:
			raise NotImplementedError("Embedding not supported with Anthropic")
		if not texts:
			return []
		resp = self._openai.embeddings.create(model=self.embedding_model, input=texts)
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
		if self._use_anthropic:
			return resp.content
		return resp.choices[0].message.content

	def chat_with_tools(self, messages: list[dict], tool_handler, **kwargs) -> tuple[str, list[dict]]:
		"""Chat with tool calling support. tool_handler should have methods matching tool names."""
		tool_calls = []
		max_turns = 10
		turn = 0

		while turn < max_turns:
			turn += 1
			resp = self._chat(
				model=self.chat_model,
				messages=messages,
				tools=TOOL_DEFINITIONS,
				**kwargs,
			)

			if self._use_anthropic:
				# Check for tool use in response
				content = resp.content
				tool_use = getattr(resp, "tool_use", None)
				if tool_use:
					# Anthropic returns tool_use blocks
					for tool in tool_use:
						tool_name = tool.name
						tool_input = tool.input
						try:
							result = self._call_tool(tool_name, tool_input, tool_handler)
							messages.append({"role": "user", "content": json.dumps(result)})
						except Exception as e:
							messages.append({"role": "user", "content": f"Error: {str(e)}"})
					continue
				# No tool use, return the content
				return content, tool_calls
			else:
				# OpenAI style
				choices = resp.choices
				if not choices:
					return "", tool_calls
				msg = choices[0].message
				if not hasattr(msg, "tool_calls") or not msg.tool_calls:
					return msg.content, tool_calls

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


class OpenAIMessage:
	class _Choice:
		class _Message:
			def __init__(self, content: str, model: str):
				self.content = content
				self.model = model

		def __init__(self, content: str, model: str):
			self.message = self._Message(content, model)

		def __getitem__(self, key):
			if key == "message":
				return self.message
			raise KeyError(key)

	def __init__(self, content: str, model: str):
		self.choices = [self._Choice(content, model)]
