def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """
    Calculate Recall@K.

    Recall@K measures how many of the relevant
    chunks were retrieved within the top K results.
    """

    if not relevant_ids:
        return 0.0

    retrieved = set(
        retrieved_ids[:k]
    )

    relevant = set(
        relevant_ids
    )

    return len(
        retrieved & relevant
    ) / len(relevant)


def reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> float:
    """
    Calculate Reciprocal Rank.

    Returns 1/rank of the first relevant result.
    Returns 0 if no relevant result was retrieved.
    """

    relevant = set(
        relevant_ids
    )

    for rank, chunk_id in enumerate(
        retrieved_ids,
        start=1,
    ):

        if chunk_id in relevant:

            return 1.0 / rank

    return 0.0