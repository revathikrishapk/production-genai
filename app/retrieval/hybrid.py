class HybridRetriever:

    def __init__(
        self,
        vector_store,
        embedder,
        bm25_retriever,
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.bm25_retriever = bm25_retriever

    def search(
        self,
        query: str,
        limit: int = 5,
        candidate_limit: int = 10,
        k: int = 60,
    ):
        # --------------------------------------------------
        # Dense retrieval
        # --------------------------------------------------

        query_vector = self.embedder.embed_query(
            query
        )

        dense_results = self.vector_store.search(
            query_vector,
            limit=candidate_limit,
        )

        # --------------------------------------------------
        # Sparse retrieval
        # --------------------------------------------------

        sparse_results = self.bm25_retriever.search(
            query,
            limit=candidate_limit,
        )

        # --------------------------------------------------
        # Reciprocal Rank Fusion
        # --------------------------------------------------

        scores = {}
        documents = {}

        # --------------------------------------------------
        # Dense results
        # --------------------------------------------------

        for rank, result in enumerate(
            dense_results,
            start=1,
        ):
            document_id = str(
                result.id
            )

            scores[document_id] = (
                scores.get(
                    document_id,
                    0,
                )
                + 1 / (k + rank)
            )

            documents[document_id] = {
                "id": document_id,

                "text": result.payload[
                    "text"
                ],

                "metadata": {
                    key: value
                    for key, value in result.payload.items()
                    if key != "text"
                },
            }

        # --------------------------------------------------
        # Sparse results
        # --------------------------------------------------

        for rank, result in enumerate(
            sparse_results,
            start=1,
        ):
            document = result[
                "document"
            ]

            document_id = document[
                "id"
            ]

            scores[document_id] = (
                scores.get(
                    document_id,
                    0,
                )
                + 1 / (k + rank)
            )

            documents[
                document_id
            ] = document

        # --------------------------------------------------
        # Sort by RRF score
        # --------------------------------------------------

        ranked_documents = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # --------------------------------------------------
        # Page diversity
        # --------------------------------------------------
        #
        # Don't allow the same page to consume
        # the entire candidate pool.
        #
        # Maximum 2 chunks per page.
        # --------------------------------------------------

        results = []
        page_counts = {}

        for document_id, score in ranked_documents:

            document = documents[
                document_id
            ]

            metadata = document.get(
                "metadata",
                {},
            )

            page = metadata.get(
                "page"
            )

            # Count chunks from this page
            current_count = page_counts.get(
                page,
                0,
            )

            # Maximum 2 chunks per page
            if current_count >= 2:
                continue

            results.append(
                {
                    "document": document,
                    "score": score,
                }
            )

            page_counts[page] = (
                current_count + 1
            )

            # Stop once we have enough results
            if len(results) >= limit:
                break

        return results