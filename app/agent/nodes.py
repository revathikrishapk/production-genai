from app.llm.context import build_context
from app.llm.prompts import build_messages


# Conservative threshold for the
# MS-MARCO cross encoder.
RELEVANCE_THRESHOLD = -5.0


class RAGNodes:

    def __init__(
        self,
        hybrid_retriever,
        reranker,
        llm,
    ):
        self.hybrid = hybrid_retriever
        self.reranker = reranker
        self.llm = llm

    def retrieve(self, state):

        question = state["question"]

        results = self.hybrid.search(
            query=question,
            limit=10,
            candidate_limit=10,
        )

        return {
            "retrieved_documents": results,
            "attempts": state.get(
                "attempts",
                0,
            ),
        }

    def rerank(self, state):

        question = state["question"]

        candidates = state[
            "retrieved_documents"
        ]

        results = self.reranker.rerank(
            query=question,
            documents=candidates,
            top_k=5,
        )

        return {
            "reranked_documents": results,
        }

    def generate(self, state):

        question = state["question"]

        documents = state[
            "reranked_documents"
        ]

        # -----------------------------------------------------
        # Deterministic relevance gate
        # -----------------------------------------------------

        if not documents:

            return {
                "answer": (
                    "I don't have enough information "
                    "in the provided documents to answer "
                    "this question."
                )
            }

        top_score = documents[0][
            "rerank_score"
        ]

        if top_score < RELEVANCE_THRESHOLD:

            return {
                "answer": (
                    "I don't have enough information "
                    "in the provided documents to answer "
                    "this question."
                )
            }

        # -----------------------------------------------------
        # Generate grounded answer
        # -----------------------------------------------------

        context = build_context(
            documents
        )

        messages = build_messages(
            question=question,
            context=context,
        )

        answer = self.llm.generate(
            messages
        )

        return {
            "answer": answer
        }

    def rewrite_query(self, state):

        question = state["question"]

        attempts = state.get(
            "attempts",
            0,
        )

        rewrite_prompt = f"""
Rewrite the following user question to make it
more precise for document retrieval.

Do not answer the question.

Original question:
{question}

Return only the rewritten question.
"""

        response = self.llm.generate(
            [
                {
                    "role": "user",
                    "content": rewrite_prompt,
                }
            ]
        )

        return {
            "question": response.strip(),
            "attempts": attempts + 1,
        }