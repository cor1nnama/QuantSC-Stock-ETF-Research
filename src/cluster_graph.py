import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.cluster import KMeans


def hermitian_rw_clustering(A: pd.DataFrame, n_clusters: int = 10, n_evecs: int | None = None):
    """
    Approximate Hermitian RW clustering inspired by the paper.

    Parameters
    ----------
    A : pd.DataFrame
        Directed adjacency matrix with nonnegative weights.
    n_clusters : int
        Number of clusters to return.
    n_evecs : int | None
        Number of eigenvectors to use. If None, uses n_clusters.

    Returns
    -------
    labels : pd.Series
        Cluster label for each ticker.
    embedding_df : pd.DataFrame
        Spectral embedding used for clustering.
    """

    tickers = list(A.index)
    W = A.values.astype(float)

    if n_evecs is None:
        n_evecs = n_clusters

    H = 1j * (W - W.T)

    d = W.sum(axis=1) + W.sum(axis=0)
    d = np.where(d == 0, 1.0, d)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d))

    H_rw = D_inv_sqrt @ H @ D_inv_sqrt

    evals, evecs = eigh(H_rw)

    idx = np.argsort(np.abs(evals))[::-1][:n_evecs]
    top_vecs = evecs[:, idx]

    X = np.hstack([top_vecs.real, top_vecs.imag])

    row_norms = np.linalg.norm(X, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    X = X / row_norms

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    cluster_ids = km.fit_predict(X)

    labels = pd.Series(cluster_ids, index=tickers, name="cluster")
    embedding_df = pd.DataFrame(X, index=tickers)

    return labels, embedding_df


def compute_cluster_leadingness(A: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """
    Compute cluster-level leadingness using average outgoing minus incoming flow.
    """
    clusters = sorted(labels.unique())
    rows = []

    for c in clusters:
        members = labels[labels == c].index
        non_members = labels[labels != c].index

        out_flow = A.loc[members, :].sum().sum()
        in_flow = A.loc[:, members].sum().sum()
        score = out_flow - in_flow

        rows.append({
            "cluster": c,
            "size": len(members),
            "out_flow": out_flow,
            "in_flow": in_flow,
            "leadingness": score
        })

    out = pd.DataFrame(rows).sort_values("leadingness", ascending=False).reset_index(drop=True)
    return out


def build_meta_flow_matrix(A: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """
    Cluster-to-cluster net flow matrix.
    """
    clusters = sorted(labels.unique())
    F = pd.DataFrame(0.0, index=clusters, columns=clusters)

    for ci in clusters:
        for cj in clusters:
            if ci == cj:
                continue

            members_i = labels[labels == ci].index
            members_j = labels[labels == cj].index

            flow_ij = A.loc[members_i, members_j].sum().sum()
            flow_ji = A.loc[members_j, members_i].sum().sum()

            F.loc[ci, cj] = flow_ij - flow_ji

    return F