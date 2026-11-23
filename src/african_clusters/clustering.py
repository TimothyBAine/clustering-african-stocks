"""Clustering methods."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler


def prepare(features: pd.DataFrame) -> np.ndarray:
    """Median-impute and robust-scale (African returns are fat-tailed)."""
    features = features.dropna(axis=1, how="all")
    return make_pipeline(SimpleImputer(strategy="median"), RobustScaler()).fit_transform(features)


def kmeans(features: pd.DataFrame, k: int, seed: int = 42) -> pd.Series:
    X = prepare(features)
    labels = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(X)
    return pd.Series(labels, index=features.index, name="cluster")


def choose_k(features: pd.DataFrame, k_range=range(2, 11), seed: int = 42) -> pd.DataFrame:
    """Inertia + silhouette for a range of k, to pick k."""
    X = prepare(features)
    rows = []
    for k in k_range:
        if k >= len(X):
            break
        km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X, km.labels_)})
    return pd.DataFrame(rows).set_index("k")


def choose_k_distance(
    distance: pd.DataFrame, method: str = "average", k_range=range(2, 11)
) -> pd.DataFrame:
    """Silhouette (on the precomputed distance) for hierarchical clusterings."""
    rows = []
    for k in k_range:
        if k >= len(distance):
            break
        labels, _ = hierarchical(distance, k, method=method)
        if labels.nunique() < 2:
            continue
        s = silhouette_score(distance.values, labels.values, metric="precomputed")
        rows.append({"k": k, "silhouette": s, "smallest_cluster": labels.value_counts().min()})
    return pd.DataFrame(rows).set_index("k")


def hierarchical(distance: pd.DataFrame, k: int, method: str = "average") -> tuple[pd.Series, np.ndarray]:
    """Agglomerative clustering on a precomputed distance matrix.

    Returns labels and the linkage matrix (for plotting a dendrogram).
    """
    condensed = squareform(distance.values, checks=False)
    Z = linkage(condensed, method=method)
    labels = fcluster(Z, t=k, criterion="maxclust") - 1
    return pd.Series(labels, index=distance.index, name="cluster"), Z
