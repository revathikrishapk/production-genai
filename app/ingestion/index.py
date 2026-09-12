from app.ingestion.process import process_pdf
from app.ingestion.storage import save_documents

from app.embeddings.embedder import Embedder
from app.retrieval.vector_store import VectorStore


PDF_PATH = "data/raw/fds_assessment__7__26MAG0065.pdf"


def index_document():

    print("=" * 70)
    print("DOCUMENT INDEXING")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Load + chunk PDF
    # --------------------------------------------------

    print("\n[1/4] Loading and chunking document...")

    documents = process_pdf(PDF_PATH)

    print(
        f"Created {len(documents)} chunks."
    )

    # --------------------------------------------------
    # 2. Save processed documents
    # --------------------------------------------------

    print("\n[2/4] Saving processed documents...")

    save_documents(
        documents,
        "data/processed/documents.json",
    )

    # --------------------------------------------------
    # 3. Generate embeddings
    # --------------------------------------------------

    print("\n[3/4] Loading embedding model...")

    embedder = Embedder()

    texts = [
        document["text"]
        for document in documents
    ]

    print("Generating embeddings...")

    embeddings = embedder.embed_documents(
        texts
    )

    print(
        f"Generated {len(embeddings)} embeddings."
    )

    # --------------------------------------------------
    # 4. Store in Qdrant
    # --------------------------------------------------

    print("\n[4/4] Indexing into Qdrant...")

    vector_store = VectorStore()

    try:

        vector_store.add_documents(
            documents,
            embeddings,
        )

        print(
            f"Successfully indexed "
            f"{len(documents)} chunks into Qdrant."
        )

    finally:

        vector_store.close()

    print("\n" + "=" * 70)
    print("INDEXING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    index_document()