from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from edukg.api.routers import agents, dashboard, graph, knowledge_tracing, pipeline, quiz, review, roster
from edukg.api.state import lifespan
from edukg.core.config import get_settings
from edukg.core.exceptions import EduKGError
from edukg.core.logging_setup import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

app = FastAPI(
    title="EduKG API",
    description="HTTP layer over the EduKG pipeline and agent layer — RAG retrieval, intent classification, "
    "the knowledge graph, the human-in-the-loop review queues, and pipeline run orchestration.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    # Comma-separated EDUKG_API_CORS_ORIGINS, not hardcoded — hardcoding this
    # to exactly ["localhost:3000"] is what caused a real, live "Couldn't
    # load the graph from the API" bug the moment the frontend dev server
    # had to fall back to port 3001 (3000 was occupied by an unrelated
    # project). See core/config/settings.py's api_cors_origins.
    allow_origins=get_settings().api_cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(EduKGError)
async def edukg_error_handler(request: Request, exc: EduKGError) -> JSONResponse:
    # Several routers call straight through to Neo4j/Postgres-backed
    # services with no try/except at all (graph.py, dashboard.py,
    # knowledge_tracing.py, review.py, the primary calls in agents.py) — an
    # outage there used to become an unhandled 500 with zero log entry
    # anywhere in the app's structlog setup. One handler here closes that
    # gap project-wide instead of adding a try/except to every endpoint.
    logger.error("unhandled_edukg_error", path=request.url.path, error=exc.message, details=exc.details)
    return JSONResponse(status_code=502, content={"detail": exc.message})


app.include_router(agents.router)
app.include_router(graph.router)
app.include_router(review.router)
app.include_router(pipeline.router)
app.include_router(dashboard.router)
app.include_router(knowledge_tracing.router)
app.include_router(quiz.router)
app.include_router(roster.router)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
