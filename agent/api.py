"""
FastAPI application for the Agentic RAG system.
Provides REST and Server-Sent Events (SSE) streaming endpoints with full explainability traces.
Serves an interactive dark-themed web chat interface.
"""
import os
import json
import asyncio
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from agent.config import settings
from agent.db import get_pool, close_pool, check_database_health
from agent.graph_utils import initialize_graph, close_graph, test_graph_connection
from agent.schemas import RetrievalTrace
from agent.agent import ask_agent, ask_agent_with_trace, execute_corrective_rag

logger = logging.getLogger(__name__)

STATIC_INDEX = Path(__file__).parent / "static" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager:
    Runs startup tasks when FastAPI boots, and shutdown tasks when it stops.
    """
    logger.info("Starting up Agentic RAG API server...")
    # 1. Warm up database pool
    await get_pool()
    # 2. Warm up knowledge graph
    await initialize_graph()
    logger.info("All database connections established!")
    
    yield  # The server is now actively taking requests
    
    logger.info("Shutting down API server...")
    await close_pool()
    await close_graph()
    logger.info("All connections closed gracefully.")


app = FastAPI(
    title="Agentic RAG with Knowledge Graph API",
    description="Enterprise Corrective RAG API combining PostgreSQL vector search and Neo4j Graphiti.",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS so browser apps can connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    """Schema for incoming chat messages."""
    message: str = Field(..., description="The user's query or prompt", min_length=1)
    session_id: str | None = Field(default=None, description="Optional session UUID")


class ChatResponse(BaseModel):
    """Schema for non-streaming chat responses with audit trace."""
    response: str
    trace: RetrievalTrace | None = None
    session_id: str | None = None


@app.get("/")
async def serve_index():
    """
    Serves the interactive dark-themed web chat application.
    """
    if STATIC_INDEX.exists():
        return FileResponse(STATIC_INDEX)
    return {"message": "Agentic RAG API is live. Visit /docs for Swagger UI."}


@app.get("/health")
async def health_check():
    """
    System health check endpoint.
    Verifies that both PostgreSQL and Neo4j are operational.
    """
    db_ok = await check_database_health()
    graph_ok = await test_graph_connection()
    
    status = "healthy" if (db_ok and graph_ok) else "degraded"
    return {
        "status": status,
        "postgres_vector_db": "connected" if db_ok else "unreachable",
        "neo4j_knowledge_graph": "connected" if graph_ok else "unreachable"
    }


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Standard non-streaming chat endpoint.
    Returns complete synthesized response along with the full explainability trace.
    """
    try:
        answer, trace = await execute_corrective_rag(request.message)
        return ChatResponse(
            response=answer,
            trace=trace,
            session_id=request.session_id
        )
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    Server-Sent Events (SSE) streaming chat endpoint.
    Streams token deltas in real-time, then emits the full explainability trace.
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            answer, trace = await execute_corrective_rag(request.message)
            
            # Stream response text smoothly
            words = answer.split(" ")
            for i, word in enumerate(words):
                prefix = "" if i == 0 else " "
                yield json.dumps({"token": prefix + word})
                await asyncio.sleep(0.015)
                
            # Emit the full explainability trace before closing
            yield json.dumps({"trace": trace.model_dump()})
            yield json.dumps({"done": True})
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield json.dumps({"error": str(e)})

    return EventSourceResponse(event_generator())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "agent.api:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False
    )
