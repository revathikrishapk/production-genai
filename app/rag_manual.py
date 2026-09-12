from app.rag import RAGPipeline


rag = RAGPipeline()


try:

    question = input(
        "\nAsk a question about the document: "
    )

    result = rag.ask(
        question
    )

    print("\n")
    print("=" * 70)
    print("QUESTION")
    print("=" * 70)

    print(result["question"])

    print("\n")
    print("=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(result["answer"])

    print("\n")
    print("=" * 70)
    print("SOURCES")
    print("=" * 70)

    for i, source in enumerate(
        result["sources"],
        start=1,
    ):

        document = source["document"]

        metadata = document["metadata"]

        print(
            f"\n[{i}] "
            f"{metadata.get('source', 'unknown')} "
            f"- Page {metadata.get('page', 'unknown')}"
        )

        print(
            f"Rerank score: "
            f"{source['rerank_score']:.4f}"
        )

finally:

    rag.close()