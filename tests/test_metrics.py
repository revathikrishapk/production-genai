from app.evaluation.metrics import (
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k():

    retrieved = [
        "a",
        "b",
        "c",
        "d",
    ]

    relevant = {
        "b",
        "c",
    }

    assert (
        recall_at_k(
            retrieved,
            relevant,
            3,
        )
        == 1.0
    )


def test_recall_at_k_partial():

    retrieved = [
        "a",
        "b",
        "c",
    ]

    relevant = {
        "b",
        "d",
    }

    assert (
        recall_at_k(
            retrieved,
            relevant,
            3,
        )
        == 0.5
    )


def test_reciprocal_rank():

    retrieved = [
        "a",
        "b",
        "c",
    ]

    relevant = {
        "b",
    }

    assert (
        reciprocal_rank(
            retrieved,
            relevant,
        )
        == 0.5
    )


def test_reciprocal_rank_not_found():

    retrieved = [
        "a",
        "b",
        "c",
    ]

    relevant = {
        "x",
    }

    assert (
        reciprocal_rank(
            retrieved,
            relevant,
        )
        == 0.0
    )