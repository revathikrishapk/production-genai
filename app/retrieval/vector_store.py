from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)


class VectorStore:

    def __init__(
        self,
        collection_name: str = "documents",
        vector_size: int = 384,
        path: str = "data/qdrant",
    ):

        self.client = QdrantClient(
            path=path
        )

        self.collection_name = collection_name

        collections = (
            self.client
            .get_collections()
            .collections
        )

        existing = [
            collection.name
            for collection in collections
        ]

        if collection_name not in existing:

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def add_documents(
        self,
        documents,
        embeddings,
    ):

        points = []

        for document, embedding in zip(
            documents,
            embeddings,
        ):

            points.append(
                PointStruct(
                    id=document["id"],
                    vector=embedding.tolist(),
                    payload={
                        "text": document["text"],
                        **document["metadata"],
                    },
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def search(
        self,
        query_vector,
        limit=5,
    ):

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=limit,
        )

        return results.points

    def close(self):
        """Explicitly close the Qdrant client."""

        if self.client is not None:
            self.client.close()
            self.client = None