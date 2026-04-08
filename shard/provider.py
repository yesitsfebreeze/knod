import json
import logging
import numpy as np
from anthropic import Anthropic
import voyageai
from voyageai.error import AuthenticationError as VoyageAuthError, RateLimitError as VoyageRateLimitError
from openai import OpenAI, APIStatusError, APITimeoutError, APIConnectionError, RateLimitError, AuthenticationError

from .config import Config

log = logging.getLogger(__name__)

_FALLBACK_ERRORS = (RateLimitError, APITimeoutError, APIConnectionError, APIStatusError, AuthenticationError)


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
		self._use_voyage = bool(cfg.voyage_api_key) and cfg.voyage_enabled
		self._use_anthropic = bool(cfg.anthropic_api_key) and cfg.anthropic_enabled
		self._use_openai = bool(cfg.openai_api_key) and cfg.openai_enabled
		self._use_local = bool(cfg.local_base_url) and cfg.local_enabled

		if self._use_voyage:
			self._voyage = voyageai.Client(api_key=cfg.voyage_api_key)
		else:
			self._voyage = None

		if self._use_anthropic:
			self._anthropic = Anthropic(
				api_key=cfg.anthropic_api_key,
				**({"base_url": cfg.anthropic_base_url} if cfg.anthropic_base_url else {}),
			)
		else:
			self._anthropic = None

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
		if self._use_anthropic:
			try:
				return self._anthropic_chat(**{**kwargs, "model": self._cfg.anthropic_model})
			except _FALLBACK_ERRORS as exc:
				log.warning("Anthropic failed (%s), trying OpenAI", exc)

		if self._use_openai:
			try:
				return self._openai_chat(**kwargs)
			except _FALLBACK_ERRORS as exc:
				log.warning("OpenAI failed (%s), trying local", exc)

		if self._local is not None:
			return self._local_chat(**kwargs)

		raise RuntimeError("No providers available: configure anthropic_api_key, openai_api_key, or local_base_url")

	def _openai_chat(self, **kwargs) -> "openai.types.chat.ChatCompletion":
		return self._openai.chat.completions.create(**kwargs)

	def _local_chat(self, **kwargs) -> "openai.types.chat.ChatCompletion":
		messages = kwargs.pop("messages", [])
		kwargs["model"] = self._cfg.local_model or self.chat_model
		return self._local.chat.completions.create(messages=messages, **kwargs)

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

		# Anthropic doesn't support response_format - remove it
		kwargs.pop("response_format", None)

		resp = self._anthropic.messages.create(
			model=model,
			max_tokens=4096,
			system=system,
			messages=anthropic_messages,
			**kwargs,
		)
		content = resp.content[0].text if resp.content else ""
		return OpenAIMessage(content=content, model=model)

	def _probe_embed(self) -> None:
		"""Status-check each embedding tier in order; cache the first working one."""
		probe = "probe"
		if self._voyage is not None:
			try:
				self._voyage.embed([probe], model=self._cfg.voyage_model)
				self._embed_provider = "voyage"
				log.info("Embedding provider: voyage (%s)", self._cfg.voyage_model)
				return
			except VoyageRateLimitError as e:
				log.warning("Voyage rate limited: %s", e)
			except VoyageAuthError as e:
				log.warning("Voyage auth error: %s", e)
				self._voyage = None
			except Exception as e:
				log.warning("Voyage embedding unavailable (%s), trying next provider", e)
				self._voyage = None

		if self._openai is not None:
			try:
				self._openai.embeddings.create(model=self._cfg.openai_embedding_model, input=probe)
				self._embed_provider = "openai"
				log.info("Embedding provider: openai (%s)", self._cfg.openai_embedding_model)
				return
			except Exception as e:
				log.warning("OpenAI embedding unavailable (%s), trying next provider", e)
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

		self._embed_provider = ""
		raise RuntimeError(
			"No embedding provider available — configure voyage_api_key, "
			"openai_api_key+openai_embedding_model, or local_base_url+local_embedding_model"
		)

	def embed_text(self, text: str) -> np.ndarray:
		if self._embed_provider is None:
			self._probe_embed()
		if self._embed_provider == "voyage":
			result = self._voyage.embed([text], model=self._cfg.voyage_model)
			return np.array(result.embeddings[0], dtype=np.float32)
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
		if self._embed_provider == "voyage":
			result = self._voyage.embed(texts, model=self._cfg.voyage_model)
			return [np.array(e, dtype=np.float32) for e in result.embeddings]
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
