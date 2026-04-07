from build_returns import load_all_returns
from leadlag_matrix import compute_ccf_auc
from cluster_graph import hermitian_rw_clustering, compute_cluster_leadingness, build_meta_flow_matrix
from network_graph import (
    build_graph,
    get_edge_list,
    compute_centrality,
    compute_leading_score,
)


def main():

    returns = load_all_returns()
    print("Returns preview:")
    print(returns.head())
    print("\nReturns shape:", returns.shape)

    print("\nComputing lead-lag matrix...")
    S = compute_ccf_auc(returns, max_lag=5)

    print("\nLead-lag matrix S (top-left 5x5):")
    print(S.iloc[:5, :5])

    A = S.clip(lower=0)

    print("\nAdjacency matrix A (top-left 5x5):")
    print(A.iloc[:5, :5])

    skew_check = (S + S.T).abs().max().max()
    print("\nSkew-symmetry check for S:", skew_check)
    print("Any negative values in A:", (A < 0).any().any())

    G = build_graph(A)

    print("\nGraph info:")
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    edges_df = get_edge_list(G)
    print("\nTop 10 edges:")
    print(edges_df.head(10))

    centrality_df = compute_centrality(G)
    print("\nTop 10 nodes by PageRank:")
    print(centrality_df.head(10))

    leading_score = compute_leading_score(A)

    print("\nTop 10 leaders:")
    print(leading_score.sort_values(ascending=False).head(10))

    print("\nTop 10 laggers:")
    print(leading_score.sort_values(ascending=True).head(10))
    print("\nRunning Hermitian RW clustering...")
    labels, embedding_df = hermitian_rw_clustering(A, n_clusters=10)

    print("\nCluster assignments (first 20):")
    print(labels.head(20))

    cluster_summary = compute_cluster_leadingness(A, labels)
    print("\nCluster leadingness:")
    print(cluster_summary)

    meta_flow = build_meta_flow_matrix(A, labels)
    print("\nMeta-flow matrix:")
    print(meta_flow)

    labels.to_csv("cluster_labels.csv", header=["cluster"])
    embedding_df.to_csv("cluster_embedding.csv")
    cluster_summary.to_csv("cluster_leadingness.csv", index=False)
    meta_flow.to_csv("meta_flow_matrix.csv")

    print("\nSaved additional clustering files:")
    print("cluster_labels.csv")
    print("cluster_embedding.csv")
    print("cluster_leadingness.csv")
    print("meta_flow_matrix.csv")

    returns.to_csv("returns_matrix.csv")
    S.to_csv("leadlag_matrix_S.csv")
    A.to_csv("adjacency_matrix_A.csv")
    edges_df.to_csv("leadlag_edges.csv", index=False)
    centrality_df.to_csv("graph_centrality.csv")
    leading_score.to_csv("leading_score.csv", header=["leading_score"])

    print("\nSaved:")
    print("returns_matrix.csv")
    print("leadlag_matrix_S.csv")
    print("adjacency_matrix_A.csv")
    print("leadlag_edges.csv")
    print("graph_centrality.csv")
    print("leading_score.csv")


if __name__ == "__main__":
    main()