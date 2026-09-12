def build_context(
    documents: list[dict],
) -> str:

    context_parts = []

    for i, result in enumerate(documents, start=1):

        document = result["document"]

        source = document["metadata"].get(
            "source",
            "unknown",
        )

        page = document["metadata"].get(
            "page",
            "unknown",
        )

        text = document["text"].strip()

        context_parts.append(
            f"""
SOURCE {i}
Document: {source}
Page: {page}

{text}
""".strip()
        )

    return "\n\n---\n\n".join(context_parts)