from app.ingestion.process import process_pdf
from app.embeddings.embedder import Embedder
from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.hybrid import HybridRetriever


# --------------------------------
# Load documents
# --------------------------------

documents = process_pdf(
    "data/raw/fds_assessment__7__26MAG0065.pdf"
)

print(f"Loaded {len(documents)} chunks")


# --------------------------------
# Embeddings
# --------------------------------

embedder = Embedder()

texts = [
    document["text"]
    for document in documents
]

embeddings = embedder.embed_documents(texts)


# --------------------------------
# Qdrant
# --------------------------------

vector_store = VectorStore()

vector_store.add_documents(
    documents,
    embeddings,
)


# --------------------------------
# BM25
# --------------------------------

bm25 = BM25Retriever(documents)


# --------------------------------
# Hybrid
# --------------------------------

retriever = HybridRetriever(
    vector_store=vector_store,
    embedder=embedder,
    bm25_retriever=bm25,
)


# --------------------------------
# Search
# --------------------------------

query = "What are the company's AI initiatives?"

results = retriever.search(
    query,
    limit=5,
)


# --------------------------------
# Display
# --------------------------------

print("\nHYBRID RESULTS\n")

for i, result in enumerate(results, start=1):

    document = result["document"]

    print("=" * 70)

    print(f"Rank: {i}")
    print(f"RRF Score: {result['score']:.6f}")
    print(f"Source: {document['metadata']['source']}")
    print(f"Page: {document['metadata']['page']}")

    print("\nText:")
    print(document["text"][:700])