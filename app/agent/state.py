from typing import TypedDict


class AgentState(TypedDict):

    question: str

    retrieved_documents: list

    reranked_documents: list

    answer: str

    attempts: int