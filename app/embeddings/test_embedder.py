from app.embeddings.embedder import Embedder


embedder = Embedder()

texts = [
    "The company invested heavily in artificial intelligence.",
    "The weather is sunny today.",
    "Artificial intelligence investment increased significantly.",
]

vectors = embedder.embed_documents(texts)

print("Number of vectors:", len(vectors))
print("Vector dimension:", len(vectors[0]))

query = "How much did the company invest in AI?"

query_vector = embedder.embed_query(query)

print("Query vector dimension:", len(query_vector))
print("First 10 values:", query_vector[:10])