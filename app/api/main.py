from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.rag import RAGPipeline
from app.api.routes.query import (
    router as query_router,
    set_rag_pipeline,
)
from app.core.logging import (
    configure_logging,
    get_logger,
)


configure_logging()

logger = get_logger(__name__)

rag_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):

    global rag_pipeline

    logger.info(
        "Starting Production GenAI API"
    )

    try:

        rag_pipeline = RAGPipeline()

        set_rag_pipeline(
            rag_pipeline
        )

        logger.info(
            "RAG pipeline initialized successfully"
        )

    except Exception:

        logger.exception(
            "Failed to initialize RAG pipeline"
        )

        raise

    yield

    logger.info(
        "Shutting down RAG pipeline"
    )

    if rag_pipeline is not None:
        rag_pipeline.close()

    logger.info(
        "Shutdown complete"
    )


app = FastAPI(
    title="Production GenAI RAG API",
    description=(
        "Production-oriented RAG and agent API "
        "with hybrid retrieval, reranking, "
        "LangGraph orchestration, and Docker."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get(
    "/health",
    tags=["health"],
)
def health():

    if rag_pipeline is None:

        return {
            "status": "starting"
        }

    return {
        "status": "healthy"
    }


app.include_router(
    query_router
)