import json
import time
from pathlib import Path

from app.ingestion.storage import load_documents
from app.embeddings.embedder import Embedder
from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import Reranker
from app.evaluation.metrics import (
    recall_at_k,
    reciprocal_rank,
)


DATASET_PATH = Path(
    "app/evaluation/dataset.json"
)

DOCUMENTS_PATH = (
    "data/processed/documents.json"
)


def load_dataset():
    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def get_document_id(result):
    """
    Extract the chunk/document ID from
    dense, BM25, hybrid, or reranker results.
    """

    if isinstance(result, dict):
        document = result.get("document")

        if document:
            return document["id"]

        return result.get("id")

    return str(result.id)


def evaluate_retrieval():

    print("=" * 80)
    print("RETRIEVAL EVALUATION")
    print("=" * 80)

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    print("\nLoading processed documents...")

    documents = load_documents(
        DOCUMENTS_PATH
    )

    print(
        f"Loaded {len(documents)} chunks."
    )

    print("\nLoading evaluation dataset...")

    dataset = load_dataset()

    print(
        f"Loaded {len(dataset)} evaluation questions."
    )

    # ---------------------------------------------------------
    # Initialize components
    # ---------------------------------------------------------

    print("\nLoading embedding model...")

    embedder = Embedder()

    print("\nConnecting to Qdrant...")

    vector_store = VectorStore()

    print("\nBuilding BM25 index...")

    bm25 = BM25Retriever(
        documents
    )

    print("\nBuilding hybrid retriever...")

    hybrid = HybridRetriever(
        vector_store=vector_store,
        embedder=embedder,
        bm25_retriever=bm25,
    )

    print("\nLoading reranker...")

    reranker = Reranker()

    # ---------------------------------------------------------
    # Metric storage
    # ---------------------------------------------------------

    methods = [
        "dense",
        "bm25",
        "hybrid",
        "reranker",
    ]

    results = {
        method: {
            "retrieved": [],
            "recall_1": [],
            "recall_3": [],
            "recall_5": [],
            "mrr": [],
            "latency_ms": [],
        }
        for method in methods
    }

    # ---------------------------------------------------------
    # Evaluation loop
    # ---------------------------------------------------------

    try:

        for question_number, item in enumerate(
            dataset,
            start=1,
        ):

            question = item["question"]

            relevant_ids = set(
                item["relevant_chunk_ids"]
            )

            print("\n")
            print("=" * 80)
            print(
                f"QUESTION {question_number}/{len(dataset)}"
            )
            print("=" * 80)

            print(
                f"\nQuestion: {question}"
            )

            print(
                "\nRelevant chunks:"
            )

            for chunk_id in relevant_ids:
                print(
                    f"  - {chunk_id}"
                )

            # =================================================
            # 1. Dense Retrieval
            # =================================================

            start = time.perf_counter()

            query_vector = (
                embedder.embed_query(
                    question
                )
            )

            dense_results = (
                vector_store.search(
                    query_vector,
                    limit=5,
                )
            )

            dense_latency = (
                time.perf_counter() - start
            ) * 1000

            dense_ids = [
                get_document_id(result)
                for result in dense_results
            ]

            results["dense"][
                "retrieved"
            ].append(dense_ids)

            results["dense"][
                "latency_ms"
            ].append(dense_latency)

            results["dense"][
                "recall_1"
            ].append(
                recall_at_k(
                    dense_ids,
                    relevant_ids,
                    1,
                )
            )

            results["dense"][
                "recall_3"
            ].append(
                recall_at_k(
                    dense_ids,
                    relevant_ids,
                    3,
                )
            )

            results["dense"][
                "recall_5"
            ].append(
                recall_at_k(
                    dense_ids,
                    relevant_ids,
                    5,
                )
            )

            results["dense"][
                "mrr"
            ].append(
                reciprocal_rank(
                    dense_ids,
                    relevant_ids,
                )
            )

            print(
                "\nDense:"
            )

            for rank, chunk_id in enumerate(
                dense_ids,
                start=1,
            ):
                print(
                    f"  {rank}. {chunk_id}"
                )

            # =================================================
            # 2. BM25
            # =================================================

            start = time.perf_counter()

            bm25_results = bm25.search(
                question,
                limit=5,
            )

            bm25_latency = (
                time.perf_counter() - start
            ) * 1000

            bm25_ids = [
                get_document_id(result)
                for result in bm25_results
            ]

            results["bm25"][
                "retrieved"
            ].append(bm25_ids)

            results["bm25"][
                "latency_ms"
            ].append(bm25_latency)

            results["bm25"][
                "recall_1"
            ].append(
                recall_at_k(
                    bm25_ids,
                    relevant_ids,
                    1,
                )
            )

            results["bm25"][
                "recall_3"
            ].append(
                recall_at_k(
                    bm25_ids,
                    relevant_ids,
                    3,
                )
            )

            results["bm25"][
                "recall_5"
            ].append(
                recall_at_k(
                    bm25_ids,
                    relevant_ids,
                    5,
                )
            )

            results["bm25"][
                "mrr"
            ].append(
                reciprocal_rank(
                    bm25_ids,
                    relevant_ids,
                )
            )

            print(
                "\nBM25:"
            )

            for rank, chunk_id in enumerate(
                bm25_ids,
                start=1,
            ):
                print(
                    f"  {rank}. {chunk_id}"
                )

            # =================================================
            # 3. Hybrid Retrieval
            #
            # IMPORTANT:
            # Retrieve 10 candidates.
            # Reranker will later reduce them to top 5.
            # =================================================

            start = time.perf_counter()

            hybrid_results = hybrid.search(
                question,
                limit=10,
                candidate_limit=10,
            )

            hybrid_latency = (
                time.perf_counter() - start
            ) * 1000

            hybrid_ids = [
                get_document_id(result)
                for result in hybrid_results
            ]

            results["hybrid"][
                "retrieved"
            ].append(hybrid_ids)

            results["hybrid"][
                "latency_ms"
            ].append(hybrid_latency)

            results["hybrid"][
                "recall_1"
            ].append(
                recall_at_k(
                    hybrid_ids,
                    relevant_ids,
                    1,
                )
            )

            results["hybrid"][
                "recall_3"
            ].append(
                recall_at_k(
                    hybrid_ids,
                    relevant_ids,
                    3,
                )
            )

            results["hybrid"][
                "recall_5"
            ].append(
                recall_at_k(
                    hybrid_ids,
                    relevant_ids,
                    5,
                )
            )

            results["hybrid"][
                "mrr"
            ].append(
                reciprocal_rank(
                    hybrid_ids,
                    relevant_ids,
                )
            )

            print(
                "\nHybrid (10 candidates):"
            )

            for rank, chunk_id in enumerate(
                hybrid_ids,
                start=1,
            ):
                print(
                    f"  {rank}. {chunk_id}"
                )

            # =================================================
            # 4. Reranker
            #
            # Takes the 10 hybrid candidates
            # and returns top 5.
            # =================================================

            start = time.perf_counter()

            reranked_results = (
                reranker.rerank(
                    query=question,
                    documents=hybrid_results,
                    top_k=5,
                )
            )

            reranker_latency = (
                time.perf_counter() - start
            ) * 1000

            reranked_ids = [
                get_document_id(result)
                for result in reranked_results
            ]

            results["reranker"][
                "retrieved"
            ].append(reranked_ids)

            results["reranker"][
                "latency_ms"
            ].append(
                hybrid_latency
                + reranker_latency
            )

            results["reranker"][
                "recall_1"
            ].append(
                recall_at_k(
                    reranked_ids,
                    relevant_ids,
                    1,
                )
            )

            results["reranker"][
                "recall_3"
            ].append(
                recall_at_k(
                    reranked_ids,
                    relevant_ids,
                    3,
                )
            )

            results["reranker"][
                "recall_5"
            ].append(
                recall_at_k(
                    reranked_ids,
                    relevant_ids,
                    5,
                )
            )

            results["reranker"][
                "mrr"
            ].append(
                reciprocal_rank(
                    reranked_ids,
                    relevant_ids,
                )
            )

            print(
                "\nReranker (top 5):"
            )

            for rank, result in enumerate(
                reranked_results,
                start=1,
            ):

                chunk_id = get_document_id(
                    result
                )

                score = result[
                    "rerank_score"
                ]

                print(
                    f"  {rank}. "
                    f"{chunk_id} "
                    f"(score={score:.4f})"
                )

    finally:

        vector_store.close()

    # ---------------------------------------------------------
    # Aggregate results
    # ---------------------------------------------------------

    print("\n\n")
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    print()

    print(
        f"{'Method':<15}"
        f"{'Recall@1':<15}"
        f"{'Recall@3':<15}"
        f"{'Recall@5':<15}"
        f"{'MRR':<15}"
        f"{'Latency(ms)':<15}"
    )

    print("-" * 95)

    for method in methods:

        data = results[method]

        recall_1 = sum(
            data["recall_1"]
        ) / len(data["recall_1"])

        recall_3 = sum(
            data["recall_3"]
        ) / len(data["recall_3"])

        recall_5 = sum(
            data["recall_5"]
        ) / len(data["recall_5"])

        mrr = sum(
            data["mrr"]
        ) / len(data["mrr"])

        latency = sum(
            data["latency_ms"]
        ) / len(data["latency_ms"])

        print(
            f"{method:<15}"
            f"{recall_1:<15.3f}"
            f"{recall_3:<15.3f}"
            f"{recall_5:<15.3f}"
            f"{mrr:<15.3f}"
            f"{latency:<15.2f}"
        )

    print("\n")
    print("=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    evaluate_retrieval()