"""FastAPI HTTP server — delegates to shared Handler."""

import logging
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..handler import Handler
from ..provider import TOOL_DEFINITIONS, QUERY_TOOL_NAMES

_QUERY_TOOLS = [t for t in TOOL_DEFINITIONS if t["function"]["name"] in QUERY_TOOL_NAMES]


class SessionStore:
	def __init__(self):
		self._lock = threading.Lock()
		self._sessions: dict[str, list[dict]] = {}

	def get(self, session_id: str) -> list[dict] | None:
		with self._lock:
			msgs = self._sessions.get(session_id)
			return list(msgs) if msgs is not None else None

	def get_or_create(self, session_id: str) -> list[dict]:
		with self._lock:
			if session_id not in self._sessions:
				self._sessions[session_id] = []
			return list(self._sessions[session_id])

	def set(self, session_id: str, messages: list[dict]) -> None:
		with self._lock:
			self._sessions[session_id] = messages

	def list_ids(self) -> list[str]:
		with self._lock:
			return list(self._sessions.keys())

	def delete(self, session_id: str) -> bool:
		with self._lock:
			return bool(self._sessions.pop(session_id, None))


def _build_system_prompt(handler: "Handler") -> str:
	info = handler.graph_info
	purpose = info.get("purpose", "")
	descriptors = info.get("descriptors", {})
	stats = handler.graph_stats()
	lines = [
		"You are a shard AI assistant with access to a persistent knowledge graph memory.",
		f"The graph currently holds {stats.get('thoughts', 0)} thoughts and {stats.get('edges', 0)} edges.",
	]
	if purpose:
		lines.append(f"Shard purpose: {purpose}")
	if descriptors:
		lines.append(f"Knowledge domains: {', '.join(descriptors.keys())}")
	lines += [
		"",
		"Before answering any question, always query the knowledge graph to retrieve relevant information:",
		"  - Use `ask` for a full semantic answer grounded in stored knowledge.",
		"  - Use `find_thoughts` to locate specific nodes, then `explore_thought` or `traverse` to follow edges.",
		"  - Use `list_shards` / `graph_stats` to orient yourself.",
		"Ground your final answer in what the graph returns. Never ingest or modify the graph during chat.",
	]
	return "\n".join(lines)

log = logging.getLogger(__name__)

# --- Request/Response models ---


class IngestRequest(BaseModel):
	text: str
	source: str = ""
	descriptor: str = ""


class IngestResponse(BaseModel):
	thoughts_added: int
	total_thoughts: int
	total_edges: int


class AskRequest(BaseModel):
	query: str


class AskResponse(BaseModel):
	answer: str
	sources: list[dict]


class PurposeRequest(BaseModel):
	purpose: str


class DescriptorRequest(BaseModel):
	name: str
	description: str = ""


class MessageRequest(BaseModel):
	messages: list[dict]
	model: str | None = None
	temperature: float | None = None
	max_tokens: int | None = None


class MessageResponse(BaseModel):
	content: str
	model: str


class ChatRequest(BaseModel):
	messages: list[dict]
	system_prompt: str | None = None
	temperature: float | None = None
	max_tokens: int | None = None


class ChatResponse(BaseModel):
	content: str
	tool_calls: list[dict] | None = None


class AgentRequest(BaseModel):
	message: str
	session_id: str | None = None


class AgentResponse(BaseModel):
	content: str
	session_id: str
	tool_calls: list[dict] | None = None


class ChatToolHandler:
	"""Adapter to expose Handler methods as MCP-style tools."""

	def __init__(self, handler: "Handler"):
		self._handler = handler

	def ask(self, query: str, knid: str | None = None) -> dict:
		answer, sources = self._handler.ask(query, knid=knid)
		return {"answer": answer, "sources": sources}

	def ingest(self, text: str, source: str = "", descriptor: str = "") -> dict:
		result = self._handler.ingest(text, source=source, descriptor=descriptor)
		return {"status": result}

	def ingest_sync(self, text: str, source: str = "", descriptor: str = "") -> dict:
		return self._handler.ingest_sync(text, source=source, descriptor=descriptor)

	def find_thoughts(self, query: str, k: int = 5) -> dict:
		return {"thoughts": self._handler.find_thoughts_by_query(query, k=k)}

	def set_purpose(self, purpose: str) -> dict:
		self._handler.set_purpose(purpose)
		return {"purpose": purpose}

	def add_descriptor(self, name: str, description: str = "") -> dict:
		self._handler.add_descriptor(name, description)
		return {"name": name, "description": description}

	def remove_descriptor(self, name: str) -> dict:
		ok = self._handler.remove_descriptor(name)
		return {"removed": ok, "name": name}

	def explore_thought(self, thought_id: int) -> dict:
		result = self._handler.explore_thought(thought_id)
		if result is None:
			return {"error": f"Thought {thought_id} not found"}
		return result

	def traverse(self, start_id: int, depth: int = 2, max_nodes: int = 50) -> dict:
		result = self._handler.traverse(start_id, depth=depth, max_nodes=max_nodes)
		if result is None:
			return {"error": f"Thought {start_id} not found"}
		return result

	def graph_stats(self) -> dict:
		return self._handler.graph_stats()

	def list_shards(self) -> dict:
		return {"shards": self._handler.list_shards()}

	def relink(self) -> dict:
		return self._handler.relink()

	def status(self) -> dict:
		return {"status": self._handler.status()}

	def get_diff(self) -> dict:
		return self._handler.get_diff()

	def graph_info(self) -> dict:
		return self._handler.graph_info

	def graph_full(self) -> dict:
		return self._handler.graph_full()

	def list_descriptors(self) -> dict:
		return {"descriptors": self._handler.graph_info.get("descriptors", {})}

	def create_shard(self, name: str, purpose: str, cluster: str | None = None) -> dict:
		"""Create a new shard alongside the main graph."""
		path = self._handler.create_shard(name, purpose, cluster=cluster)
		return {"name": name, "purpose": purpose, "path": path}

	def ingest_into_shard(self, shard_name: str, text: str, source: str = "", descriptor: str = "") -> dict:
		"""Ingest text directly into a named shard."""
		try:
			count = self._handler.ingest_into_shard(shard_name, text, source=source, descriptor=descriptor)
			return {"shard": shard_name, "committed": count}
		except KeyError as e:
			return {"error": str(e)}

	def get_ingested_sources(self) -> dict:
		return {"sources": list(self._handler.ingested_sources())}

	def process(self, operations: list[dict]) -> dict:
		"""Execute a batch of operations. Each operation is a dict with 'tool' and 'args'."""
		results = []
		for op in operations:
			tool_name = op.get("tool")
			args = op.get("args", {})
			if not tool_name or not hasattr(self, tool_name):
				results.append({"op": op, "error": f"Unknown tool: {tool_name}"})
				continue
			try:
				tool = getattr(self, tool_name)
				result = tool(**args)
				results.append({"op": op, "result": result})
			except Exception as e:
				results.append({"op": op, "error": str(e)})
		return {"results": results}


def http(handler: Handler) -> FastAPI:
	@asynccontextmanager
	async def lifespan(app: FastAPI):
		yield

	app = FastAPI(title="shard", lifespan=lifespan)

	@app.post("/ingest", response_model=IngestResponse)
	def ingest(req: IngestRequest):
		stats = handler.ingest_sync(req.text, req.source, req.descriptor)
		return IngestResponse(
			thoughts_added=stats.get("committed", 0),
			total_thoughts=stats["thoughts"],
			total_edges=stats["edges"],
		)

	@app.post("/ask", response_model=AskResponse)
	def ask(req: AskRequest):
		answer, sources = handler.ask(req.query)
		return AskResponse(answer=answer, sources=sources)

	@app.get("/explore")
	def explore():
		return handler.graph_info

	@app.get("/stats")
	def stats():
		return handler.graph_stats()

	@app.get("/shards")
	def shards():
		return handler.list_shards()

	@app.get("/thought/{thought_id}")
	def get_thought(thought_id: int):
		result = handler.explore_thought(thought_id)
		if result is None:
			from fastapi.responses import JSONResponse

			return JSONResponse(status_code=404, content={"error": f"Thought {thought_id} not found"})
		return result

	@app.get("/traverse/{start_id}")
	def traverse(start_id: int, depth: int = 2, max_nodes: int = 50):
		result = handler.traverse(start_id, depth=depth, max_nodes=max_nodes)
		if result is None:
			from fastapi.responses import JSONResponse

			return JSONResponse(status_code=404, content={"error": f"Thought {start_id} not found"})
		return result

	@app.post("/ingest/sync", response_model=None)
	def ingest_sync(req: IngestRequest):
		return handler.ingest_sync(req.text, req.source, req.descriptor)

	@app.get("/health")
	def health():
		return {"status": "ok"}

	@app.get("/diff")
	def get_diff():
		"""Poll for changes since last call. Returns new thoughts and current state."""
		return handler.get_diff()

	@app.get("/status")
	def status():
		return {"status": handler.status()}

	@app.post("/purpose")
	def set_purpose(req: PurposeRequest):
		handler.set_purpose(req.purpose)
		return {"purpose": handler.graph_info["purpose"]}

	@app.post("/descriptor/add")
	def add_descriptor(req: DescriptorRequest):
		handler.add_descriptor(req.name, req.description)
		return {"descriptors": handler.graph_info["descriptors"]}

	@app.post("/descriptor/remove")
	def remove_descriptor(req: DescriptorRequest):
		handler.remove_descriptor(req.name)
		return {"descriptors": handler.graph_info["descriptors"]}

	@app.get("/descriptor/list")
	def list_descriptors():
		return {"descriptors": handler.graph_info["descriptors"]}

	@app.get("/graph/full")
	def graph_full():
		return handler.graph_full()

	@app.get("/graph/meta")
	def graph_meta():
		return handler.graph_meta()

	@app.get("/graph/seed")
	def graph_seed(n: int = 12):
		return handler.graph_seed(n=n)

	@app.post("/graph/expand")
	def graph_expand(body: dict):
		keys = body.get("keys", [])
		known = set(body.get("known", []))
		return handler.graph_expand(keys=keys, known=known)

	@app.get("/graph/knn_edges")
	def graph_knn_edges(k: int = 3):
		return handler.graph_knn_edges(k=k)

	@app.get("/graph/thoughts")
	def graph_thoughts(offset: int = 0, limit: int = 200):
		return handler.graph_thoughts(offset=offset, limit=limit)

	@app.post("/relink")
	def relink():
		return handler.relink()

	@app.post("/message", response_model=MessageResponse)
	def message(req: MessageRequest):
		kwargs = {}
		if req.temperature is not None:
			kwargs["temperature"] = req.temperature
		if req.max_tokens is not None:
			kwargs["max_tokens"] = req.max_tokens
		model = req.model or handler.provider.chat_model
		content = handler.provider.chat(req.messages, **kwargs)
		return MessageResponse(content=content, model=model)

	@app.post("/chat", response_model=ChatResponse)
	def chat(req: ChatRequest):
		tool_handler = ChatToolHandler(handler)
		build_messages = []

		if req.system_prompt:
			build_messages.append({"role": "system", "content": req.system_prompt})
		build_messages.extend(req.messages)

		kwargs = {}
		if req.temperature is not None:
			kwargs["temperature"] = req.temperature
		if req.max_tokens is not None:
			kwargs["max_tokens"] = req.max_tokens

		content, tool_calls = handler.provider.chat_with_tools(build_messages, tool_handler, **kwargs)
		return ChatResponse(content=content, tool_calls=tool_calls or None)

	_sessions = SessionStore()

	@app.post("/agent", response_model=AgentResponse)
	def agent(req: AgentRequest):
		session_id = req.session_id or str(uuid.uuid4())
		messages = _sessions.get_or_create(session_id)
		if not messages:
			messages.append({"role": "system", "content": _build_system_prompt(handler)})
		messages.append({"role": "user", "content": req.message})
		tool_handler = ChatToolHandler(handler)
		content, tool_calls = handler.provider.chat_with_tools(messages, tool_handler, tools=_QUERY_TOOLS)
		messages.append({"role": "assistant", "content": content})
		_sessions.set(session_id, messages)
		return AgentResponse(content=content, session_id=session_id, tool_calls=tool_calls or None)

	@app.get("/agent/sessions")
	def list_agent_sessions():
		return {"sessions": _sessions.list_ids()}

	@app.get("/agent/session/{session_id}")
	def get_agent_session(session_id: str):
		msgs = _sessions.get(session_id)
		if msgs is None:
			from fastapi.responses import JSONResponse
			return JSONResponse(status_code=404, content={"error": "Session not found"})
		return {"session_id": session_id, "messages": msgs, "length": len(msgs)}

	@app.delete("/agent/session/{session_id}")
	def delete_agent_session(session_id: str):
		return {"deleted": _sessions.delete(session_id), "session_id": session_id}

	# Static files (CSS, JS) for the web UI
	_web_dir = Path(__file__).resolve().parent.parent / "web"
	app.mount("/public", StaticFiles(directory=str(_web_dir / "_public")), name="public")

	_NO_CACHE = {"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"}

	@app.get("/view", response_class=HTMLResponse)
	def view():
		return HTMLResponse((_web_dir / "view" / "index.html").read_text(encoding="utf-8"), headers=_NO_CACHE)

	@app.get("/chat", response_class=HTMLResponse)
	def chat_ui():
		return HTMLResponse((_web_dir / "chat" / "index.html").read_text(encoding="utf-8"), headers=_NO_CACHE)

	return app
