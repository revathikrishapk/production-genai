from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    QueryRequest,
    QueryResponse,
)

router = APIRouter()

rag_pipeline = None


def set_rag_pipeline(pipeline):
    global rag_pipeline
    rag_pipeline = pipeline


@router.post(
    "/query",
    response_model=QueryResponse,
)
def query(
    request: QueryRequest,
):

    if rag_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is not initialized.",
        )

    try:

        result = rag_pipeline.ask(
            request.question
        )

        sources = []
        seen_sources = set()

        for source in result["sources"]:

            document = source["document"]
            metadata = document["metadata"]

            source_name = metadata.get(
                "source",
                "unknown",
            )

            page = metadata.get(
                "page",
                "unknown",
            )

            source_key = (
                source_name,
                page,
            )

            # Avoid returning the same
            # PDF page multiple times.
            if source_key in seen_sources:
                continue

            seen_sources.add(
                source_key
            )

            sources.append(
                {
                    "source": source_name,
                    "page": page,
                    "rerank_score": source[
                        "rerank_score"
                    ],
                }
            )

        return {
            "question": result["question"],
            "answer": result["answer"],
            "sources": sources,
            "attempts": result["attempts"],
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )