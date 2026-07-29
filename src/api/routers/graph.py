"""Knowledge graph endpoints."""

import networkx as nx
from fastapi import APIRouter

from src.graph.knowledge_graph import build_graph, to_mermaid

router = APIRouter(prefix="/api/graph", tags=["graph"])


@router.get("/mermaid")
def graph_mermaid():
    g = build_graph()
    return {"mermaid": to_mermaid(g), "n_nodes": g.number_of_nodes(), "n_edges": g.number_of_edges()}


@router.get("/data")
def graph_data():
    g = build_graph()
    return nx.node_link_data(g, edges="edges")
