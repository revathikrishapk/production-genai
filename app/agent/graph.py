from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from app.agent.state import AgentState
from app.agent.nodes import RAGNodes


def build_graph(
    hybrid_retriever,
    reranker,
    llm,
):

    nodes = RAGNodes(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        llm=llm,
    )

    graph = StateGraph(AgentState)

    graph.add_node(
        "retrieve",
        nodes.retrieve,
    )

    graph.add_node(
        "rerank",
        nodes.rerank,
    )

    graph.add_node(
        "generate",
        nodes.generate,
    )

    graph.add_edge(
        START,
        "retrieve",
    )

    graph.add_edge(
        "retrieve",
        "rerank",
    )

    graph.add_edge(
        "rerank",
        "generate",
    )

    graph.add_edge(
        "generate",
        END,
    )

    return graph.compile()