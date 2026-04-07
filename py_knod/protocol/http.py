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

	@app.get("/health")
	def health():
		return {"status": "ok"}

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

	return app
