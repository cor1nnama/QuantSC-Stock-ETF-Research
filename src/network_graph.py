import networkx as nx
import pandas as pd


def build_graph(A: pd.DataFrame) -> nx.DiGraph:
    """
    Build directed graph from adjacency matrix A.
    A_ij = strength of i leading j.
    """
    G = nx.DiGraph()

    for i in A.index:
        for j in A.columns:
            weight = A.loc[i, j]
            if weight > 0:
                G.add_edge(i, j, weight=float(weight))

    return G


def get_edge_list(G: nx.DiGraph) -> pd.DataFrame:
    """
    Return sorted edge list DataFrame.
    """
    edge_rows = []

    for u, v, data in G.edges(data=True):
        edge_rows.append({
            "source": u,
            "target": v,
            "weight": data["weight"]
        })

    edges_df = pd.DataFrame(edge_rows)

    if edges_df.empty:
        return edges_df

    return edges_df.sort_values("weight", ascending=False)


def compute_centrality(G: nx.DiGraph) -> pd.DataFrame:
    """
    Compute in-degree, out-degree, and PageRank.
    """
    in_deg = dict(G.in_degree(weight="weight"))
    out_deg = dict(G.out_degree(weight="weight"))
    pagerank = nx.pagerank(G, weight="weight")

    df = pd.DataFrame({
        "in_degree": pd.Series(in_deg),
        "out_degree": pd.Series(out_deg),
        "pagerank": pd.Series(pagerank)
    })

    return df.sort_values("pagerank", ascending=False)


def compute_leading_score(A: pd.DataFrame) -> pd.Series:
    """
    Leading score = outgoing flow - incoming flow.
    """
    return A.sum(axis=1) - A.sum(axis=0)


def filter_graph(G: nx.DiGraph, threshold: float = 0.1) -> nx.DiGraph:
    """
    Keep only edges with weight >= threshold.
    """
    G_filtered = nx.DiGraph()

    for u, v, data in G.edges(data=True):
        if data["weight"] >= threshold:
            G_filtered.add_edge(u, v, weight=data["weight"])

    return G_filtered