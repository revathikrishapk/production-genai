from app.retrieval.vector_store import VectorStore


store = VectorStore()

try:
    collections = (
        store.client
        .get_collections()
        .collections
    )

    print("Collections:")

    for collection in collections:
        print("-", collection.name)

finally:
    store.client.close()