from app.ingestion.process import process_pdf
from app.retrieval.bm25 import BM25Retriever


documents = process_pdf(
    "data/raw/fds_assessment__7__26MAG0065.pdf"
)

print(f"Loaded {len(documents)} chunks")


retriever = BM25Retriever(documents)


query = "What are the company's AI initiatives?"

results = retriever.search(
    query,
    limit=5,
)


print("\nBM25 RESULTS\n")


for i, result in enumerate(results, start=1):

    document = result["document"]

    print("=" * 60)

    print(f"Rank: {i}")
    print(f"Score: {result['score']}")
    print(f"Source: {document['metadata']['source']}")
    print(f"Page: {document['metadata']['page']}")

    print("\nText:")
    print(document["text"][:700])