from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self, documents: list[dict]):
        self.documents = documents

        tokenized_documents = [
            self._tokenize(document["text"])
            for document in documents
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    def search(self, query: str, limit: int = 5):

        tokenized_query = self._tokenize(query)

        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = scores.argsort()[::-1][:limit]

        results = []

        for index in ranked_indices:

            results.append(
                {
                    "document": self.documents[index],
                    "score": float(scores[index]),
                }
            )

        return results