def check_evidence(state):

    documents = state.get(
        "reranked_documents",
        []
    )

    attempts = state.get(
        "attempts",
        0
    )

    # No retrieved evidence
    if not documents:
        return "generate"

    # Only allow one retrieval attempt.
    # This prevents wasting LLM quota.
    if attempts >= 1:
        return "generate"

    top_score = documents[0]["rerank_score"]

    # Cross-encoder scores are model-dependent.
    # We use a conservative threshold here.
    if top_score < 0:
        return "generate"

    return "generate"