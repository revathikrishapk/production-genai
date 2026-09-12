from app.ingestion.storage import load_documents

from app.embeddings.embedder import Embedder

from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import Reranker

from app.llm.client import OpenRouterClient

from app.agent.graph import build_graph


class RAGPipeline:

    def __init__(
        self,
        documents_path: str = "data/processed/documents.json",
    ):

        print("Loading processed documents...")

        self.documents = load_documents(
            documents_path
        )

        print(
            f"Loaded {len(self.documents)} chunks."
        )

        # --------------------------------------------------
        # 1. Embedding model
        # --------------------------------------------------

        print("Loading embedding model...")

        self.embedder = Embedder()

        # --------------------------------------------------
        # 2. Existing Qdrant index
        # --------------------------------------------------

        print("Connecting to Qdrant...")

        self.vector_store = VectorStore()

        # --------------------------------------------------
        # 3. BM25
        # --------------------------------------------------

        print("Building BM25 index...")

        self.bm25 = BM25Retriever(
            self.documents
        )

        # --------------------------------------------------
        # 4. Hybrid retrieval
        # --------------------------------------------------

        self.hybrid = HybridRetriever(
            vector_store=self.vector_store,
            embedder=self.embedder,
            bm25_retriever=self.bm25,
        )

        # --------------------------------------------------
        # 5. Reranker
        # --------------------------------------------------

        print("Loading reranker...")

        self.reranker = Reranker()

        # --------------------------------------------------
        # 6. LLM
        # --------------------------------------------------

        print("Initializing LLM...")

        self.llm = OpenRouterClient()

        # --------------------------------------------------
        # 7. LangGraph
        # --------------------------------------------------

        print("Building LangGraph agent...")

        self.graph = build_graph(
            hybrid_retriever=self.hybrid,
            reranker=self.reranker,
            llm=self.llm,
        )

        print("RAG pipeline ready.")

    # ------------------------------------------------------
    # Query
    # ------------------------------------------------------

    def ask(
        self,
        question: str,
    ):

        if not question or not question.strip():

            raise ValueError(
                "Question cannot be empty."
            )

        initial_state = {

            "question": question.strip(),

            "retrieved_documents": [],

            "reranked_documents": [],

            "answer": "",

            "attempts": 0,
        }

        result = self.graph.invoke(
            initial_state
        )

        return {

            "answer": result["answer"],

            "sources": result[
                "reranked_documents"
            ],

            "question": result[
                "question"
            ],

            "attempts": result[
                "attempts"
            ],
        }

    # ------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------

    def close(self):

        if self.vector_store is not None:

            self.vector_store.close()