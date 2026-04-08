"""FastAPI HTTP server — delegates to shared Handler."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..handler import Handler

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


class ChatToolHandler:
	"""Adapter to expose Handler methods as MCP-style tools."""

	def __init__(self, handler: "Handler"):
		self._handler = handler

	def ask(self, query: str) -> dict:
		answer, sources = self._handler.ask(query)
		return {"answer": answer, "sources": sources}

	def ingest(self, text: str, source: str = "", descriptor: str = "") -> dict:
		result = self._handler.ingest(text, source=source, descriptor=descriptor)
		return {"status": result}

	def find_thoughts(self, query: str, k: int = 5) -> dict:
		return self._handler.find_thoughts_by_query(query, k=k)

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

	def list_strands(self) -> dict:
		return {"strands": self._handler.list_strands()}

	def ingest_sync(self, text: str, source: str = "", descriptor: str = "") -> dict:
		return self._handler.ingest_sync(text, source=source, descriptor=descriptor)

	def relink(self) -> dict:
		return self._handler.relink()


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

	@app.get("/strands")
	def strands():
		return handler.list_strands()

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

	# Static files (CSS, JS) for the web UI
	_web_dir = Path(__file__).resolve().parent.parent / "web"
	app.mount("/public", StaticFiles(directory=str(_web_dir / "_public")), name="public")

	@app.get("/view", response_class=HTMLResponse)
	def view():
		return (_web_dir / "view" / "index.html").read_text(encoding="utf-8")

	@app.get("/chat", response_class=HTMLResponse)
	def chat():
		return (_web_dir / "chat" / "index.html").read_text(encoding="utf-8")

	return app
