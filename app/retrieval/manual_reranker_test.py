from app.ingestion.process import process_pdf
from app.embeddings.embedder import Embedder
from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import Reranker


# ==========================================
# 1. Load documents
# ==========================================

documents = process_pdf(
    "data/raw/fds_assessment__7__26MAG0065.pdf"
)

print(f"Loaded {len(documents)} chunks")


# ==========================================
# 2. Embeddings
# ==========================================

embedder = Embedder()

texts = [
    document["text"]
    for document in documents
]

embeddings = embedder.embed_documents(texts)


# ==========================================
# 3. Vector store
# ==========================================

vector_store = VectorStore()

vector_store.add_documents(
    documents,
    embeddings,
)


# ==========================================
# 4. BM25
# ==========================================

bm25 = BM25Retriever(documents)


# ==========================================
# 5. Hybrid retrieval
# ==========================================

hybrid = HybridRetriever(
    vector_store=vector_store,
    embedder=embedder,
    bm25_retriever=bm25,
)


# ==========================================
# 6. Query
# ==========================================

query = "What are the company's main AI initiatives?"


# ==========================================
# 7. Hybrid candidates
# ==========================================

candidates = hybrid.search(
    query,
    limit=10,
    candidate_limit=10,
)

print("\n")
print("=" * 70)
print("HYBRID RETRIEVAL")
print("=" * 70)

for i, result in enumerate(candidates, start=1):

    document = result["document"]

    print(
        f"\n{i}. "
        f"RRF={result['score']:.6f} "
        f"page={document['metadata']['page']}"
    )

    print(document["text"][:300])


# ==========================================
# 8. Reranking
# ==========================================

reranker = Reranker()

reranked = reranker.rerank(
    query=query,
    documents=candidates,
    top_k=5,
)


# ==========================================
# 9. Final results
# ==========================================

print("\n")
print("=" * 70)
print("RERANKED RESULTS")
print("=" * 70)

for i, result in enumerate(reranked, start=1):

    document = result["document"]

    print(
        f"\n{i}. "
        f"rerank={result['rerank_score']:.4f} "
        f"retrieval={result['retrieval_score']:.6f} "
        f"page={document['metadata']['page']}"
    )

    print(document["text"][:500])