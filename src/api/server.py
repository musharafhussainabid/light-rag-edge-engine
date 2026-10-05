"""FastAPI service exposing grounded question answering over local documents.

The vector store and quantized model are loaded once at startup and shared by
all requests. Configuration comes from environment variables:

==========================  ====================================================
``LIGHTRAG_MODEL``          Preset key (default ``qwen2.5-1.5b``)
``LIGHTRAG_QUANT``          ``q4_k_m`` (default), ``q8_0`` or ``f16``
``LIGHTRAG_MODEL_PATH``     Local ``.gguf`` file; overrides model/quant
``LIGHTRAG_DOCS_DIR``       Documents to index (default ``data/docs``)
``LIGHTRAG_INDEX_DIR``      Saved FAISS index (default ``data/index``)
``LIGHTRAG_N_CTX``          Model context window (default ``2048``)
``LIGHTRAG_THREADS``        llama.cpp CPU threads (default: library choice)
==========================  ====================================================
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.core.quant_engine import DEFAULT_MODEL, DEFAULT_QUANT, QuantizedSLMEngine
from src.retrieval.vector_store import DEFAULT_DOCS_DIR, DEFAULT_INDEX_DIR, VectorStore

logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    """Body of ``POST /query``."""

    prompt: str = Field(..., min_length=1, max_length=4000, description="The user's question.")
    top_k: int = Field(3, ge=1, le=10, description="Number of context chunks to retrieve.")
    max_tokens: int = Field(256, ge=1, le=1024, description="Maximum tokens to generate.")
    temperature: float = Field(0.2, ge=0.0, le=2.0)


class SourceChunk(BaseModel):
    """A retrieved chunk that was given to the model as context."""

    source: str
    chunk_id: int
    score: float
    text: str


class LatencyMetrics(BaseModel):
    """Timing breakdown for one request."""

    retrieval_ms: float = Field(description="Query embedding + FAISS search.")
    queue_ms: float = Field(description="Wait for the model while other requests were generating.")
    ttft_ms: float = Field(description="Time to first token, including prompt processing.")
    generation_ms: float = Field(description="Total model time, first token through last.")
    total_ms: float = Field(description="End-to-end server time for the request.")
    completion_tokens: int
    tokens_per_second: float


class QueryResponse(BaseModel):
    """Body returned by ``POST /query``."""

    answer: str
    model: str
    sources: list[SourceChunk]
    metrics: LatencyMetrics


def _optional_int(name: str) -> int | None:
    value = os.getenv(name)
    return int(value) if value else None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the index and model once, before the server accepts requests."""
    docs_dir = Path(os.getenv("LIGHTRAG_DOCS_DIR", str(DEFAULT_DOCS_DIR)))
    index_dir = Path(os.getenv("LIGHTRAG_INDEX_DIR", str(DEFAULT_INDEX_DIR)))
    model_path = os.getenv("LIGHTRAG_MODEL_PATH")

    app.state.store = VectorStore.load_or_build(docs_dir, index_dir)
    app.state.engine = QuantizedSLMEngine(
        model=os.getenv("LIGHTRAG_MODEL", DEFAULT_MODEL),
        quant=os.getenv("LIGHTRAG_QUANT", DEFAULT_QUANT),
        model_path=Path(model_path) if model_path else None,
        n_ctx=_optional_int("LIGHTRAG_N_CTX") or 2048,
        n_threads=_optional_int("LIGHTRAG_THREADS"),
    )
    # A llama.cpp context holds one KV cache and is not thread-safe, so generations
    # are serialized. Concurrent requests queue here; see LatencyMetrics.queue_ms.
    app.state.engine_lock = threading.Lock()
    logger.info(
        "Ready: %s with %d indexed chunks", app.state.engine.model_name, app.state.store.index.ntotal
    )
    yield


app = FastAPI(
    title="Light-RAG Edge Engine",
    description="On-device RAG with a quantized small language model on CPU.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request) -> dict[str, object]:
    """Liveness probe that also reports what is loaded."""
    state = request.app.state
    return {
        "status": "ok",
        "model": state.engine.model_name,
        "indexed_chunks": state.store.index.ntotal,
    }


# Declared sync so FastAPI runs it in its threadpool: retrieval and generation are
# blocking CPU work and would otherwise stall the event loop.
@app.post("/query", response_model=QueryResponse)
def query(body: QueryRequest, request: Request) -> QueryResponse:
    """Retrieve context for ``prompt`` and return a grounded answer with latency metrics."""
    state = request.app.state
    start = time.perf_counter()

    results = state.store.search(body.prompt, body.top_k)
    retrieval_done = time.perf_counter()

    with state.engine_lock:
        lock_acquired = time.perf_counter()
        try:
            result = state.engine.generate(
                body.prompt,
                [r.chunk.text for r in results],
                max_tokens=body.max_tokens,
                temperature=body.temperature,
            )
        except ValueError as exc:  # prompt too long for the context window
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    end = time.perf_counter()

    return QueryResponse(
        answer=result.text.strip(),
        model=state.engine.model_name,
        sources=[
            SourceChunk(source=r.chunk.source, chunk_id=r.chunk.chunk_id, score=r.score, text=r.chunk.text)
            for r in results
        ],
        metrics=LatencyMetrics(
            retrieval_ms=(retrieval_done - start) * 1000,
            queue_ms=(lock_acquired - retrieval_done) * 1000,
            ttft_ms=result.time_to_first_token_s * 1000,
            generation_ms=result.total_time_s * 1000,
            total_ms=(end - start) * 1000,
            completion_tokens=result.completion_tokens,
            tokens_per_second=result.tokens_per_second,
        ),
    )
