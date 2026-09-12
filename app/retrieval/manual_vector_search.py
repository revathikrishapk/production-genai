from app.ingestion.process import process_pdf
from app.embeddings.embedder import Embedder
from app.retrieval.vector_store import VectorStore




pdf_path = (
    r"C:\Users\User\Desktop\production-genai"
    r"\data\raw\fds_assessment__7__26MAG0065.pdf"
)



print("\n[1/5] Loading and chunking PDF...")

documents = process_pdf(pdf_path)

print(f"Documents/chunks: {len(documents)}")


if not documents:
    raise ValueError("No documents/chunks were created from the PDF.")




print("\n[2/5] Loading embedding model...")

embedder = Embedder()

print("Embedding model loaded.")



print("\n[3/5] Generating embeddings...")

texts = [
    document["text"]
    for document in documents
]

embeddings = embedder.embed_documents(texts)

print(f"Embeddings generated: {len(embeddings)}")


if embeddings is None or len(embeddings) == 0:
    raise ValueError("No embeddings were generated.")




print("\n[4/5] Creating vector store...")

vector_store = VectorStore()

print("Vector store created.")




print("Indexing documents...")

vector_store.add_documents(
    documents,
    embeddings,
)

print("Documents indexed successfully.")




print("\n[5/5] Running semantic search...")

query = "What are the company's main AI initiatives?"

print(f"\nQuery: {query}")



query_vector = embedder.embed_query(query)



results = vector_store.search(
    query_vector,
    limit=5,
)




print("\n" + "=" * 70)
print("TOP SEARCH RESULTS")
print("=" * 70)


if not results:
    print("No search results found.")

else:

    for i, result in enumerate(results, start=1):

        print("\n" + "=" * 70)

        print(f"Rank       : {i}")

        # Handle different result formats safely
        if hasattr(result, "score"):
            print(f"Score      : {result.score}")

        if hasattr(result, "payload"):

            payload = result.payload

            print(f"Source     : {payload.get('source', 'Unknown')}")
            print(f"Page       : {payload.get('page', 'Unknown')}")

            print("\nText:")
            print("-" * 70)

            text = payload.get("text", "")

            print(text[:1000])

        else:
            print("Result:")
            print(result)


print("\n" + "=" * 70)
print("VECTOR SEARCH TEST COMPLETED")
print("=" * 70)