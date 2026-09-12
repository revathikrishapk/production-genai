from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(
        self,
        model_name: str = (
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ),
    ):
        self.model = CrossEncoder(
            model_name
        )

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 5,
    ) -> list[dict]:

        if not documents:
            return []

        # -----------------------------------------------------
        # Deduplicate by chunk ID
        # -----------------------------------------------------

        unique_documents = []
        seen_ids = set()

        for document in documents:

            chunk_id = document[
                "document"
            ]["id"]

            if chunk_id in seen_ids:
                continue

            seen_ids.add(chunk_id)

            unique_documents.append(
                document
            )

        # -----------------------------------------------------
        # Cross-encoder scoring
        # -----------------------------------------------------

        pairs = [
            [
                query,
                document["document"]["text"],
            ]
            for document in unique_documents
        ]

        scores = self.model.predict(
            pairs
        )

        reranked = []

        for document, score in zip(
            unique_documents,
            scores,
        ):

            reranked.append(
                {
                    "document": document[
                        "document"
                    ],
                    "retrieval_score": document[
                        "score"
                    ],
                    "rerank_score": float(
                        score
                    ),
                }
            )

        # -----------------------------------------------------
        # Sort by relevance
        # -----------------------------------------------------

        reranked.sort(
            key=lambda x: x[
                "rerank_score"
            ],
            reverse=True,
        )

        return reranked[:top_k]