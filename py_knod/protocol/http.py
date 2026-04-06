"""FastAPI HTTP server — delegates to shared Handler."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
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


def http(handler: Handler) -> FastAPI:
	@asynccontextmanager
	async def lifespan(app: FastAPI):
		yield

	app = FastAPI(title="knod", lifespan=lifespan)

	@app.post("/ingest", response_model=IngestResponse)
	def ingest(req: IngestRequest):
		stats = handler.handle_ingest(req.text, req.source, req.descriptor)
		return IngestResponse(
			thoughts_added=stats.get("committed", 0),
			total_thoughts=stats["thoughts"],
			total_edges=stats["edges"],
		)

	@app.post("/ask", response_model=AskResponse)
	def ask(req: AskRequest):
		answer, sources = handler.handle_ask(req.query)
		return AskResponse(answer=answer, sources=sources)

	@app.get("/explore")
	def explore():
		return {
			"purpose": handler.graph.purpose,
			"thought_count": handler.graph.num_thoughts,
			"edge_count": handler.graph.num_edges,
			"descriptors": handler.graph.descriptors,
		}

	@app.get("/health")
	def health():
		return {"status": "ok"}

	@app.get("/status")
	def status():
		return {"status": handler.handle_status()}

	@app.post("/purpose")
	def set_purpose(req: PurposeRequest):
		handler.handle_set_purpose(req.purpose)
		return {"purpose": handler.graph.purpose}

	@app.post("/descriptor/add")
	def add_descriptor(req: DescriptorRequest):
		handler.handle_descriptor_add(req.name, req.description)
		return {"descriptors": handler.graph.descriptors}

	@app.post("/descriptor/remove")
	def remove_descriptor(req: DescriptorRequest):
		handler.handle_descriptor_remove(req.name)
		return {"descriptors": handler.graph.descriptors}

	@app.get("/descriptor/list")
	def list_descriptors():
		return {"descriptors": handler.graph.descriptors}

	return app
