"""Similarity views: different answers to "what makes two stocks alike?".

Each view turns the data into a distance matrix between stocks, so every view
is clustered the same way (hierarchical linkage) and the results can be compared fairly.

* ``returns``      co-movement: sqrt(2(1 - rho)) between daily log returns
* ``risk``         risk profile: vol, downside vol, beta, drawdown, skew, kurtosis
* ``liquidity``    trading behaviour: zero-return days, lag-1 autocorr, volume
* ``profile``      every history feature at once (the original "features" mode)
* ``fundamentals`` balance-sheet / valuation fields from ``/fundamentals``

All views use Ward linkage, which gives balanced clusters (close to KMeans'
objective). Ward needs Euclidean distances. Feature views use Euclidean distance on
median-imputed, robust-scaled features, capped at +/-``CLIP`` so a few extreme
stocks cannot claim clusters for themselves. The returns view qualifies too:
sqrt(2(1 - rho)) is the Euclidean distance between standardised return series.
Average linkage was tried and chains on illiquid stocks (one giant cluster plus singletons).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from . import clustering, features

VIEW_NAMES = ("returns", "risk", "liquidity", "profile", "fundamentals")
HISTORY_VIEWS = ("returns", "risk", "liquidity", "profile")


@dataclass
class View:
    name: str
    distance: pd.DataFrame
    method: str
    features: pd.DataFrame | None = None

    def subset(self, keys) -> "View":
        keys = [k for k in keys if k in self.distance.index]
        feats = None if self.features is None else self.features.loc[keys]
        return View(self.name, self.distance.loc[keys, keys], self.method, feats)


CLIP = 4.0  # robust-scaled units (IQRs from the median)


def feature_distance(feats: pd.DataFrame) -> pd.DataFrame:
    feats = feats.dropna(axis=1, how="all")
    X = np.clip(clustering.prepare(feats), -CLIP, CLIP)
    return pd.DataFrame(squareform(pdist(X)), index=feats.index, columns=feats.index)


def build_view(
    name: str,
    close: pd.DataFrame | None = None,
    volume: pd.DataFrame | None = None,
    fundamentals: pd.DataFrame | None = None,
    local_close: pd.DataFrame | None = None,
) -> View:
    """``close`` may be in USD; ``local_close`` (traded prices) feeds the liquidity measures."""
    if name in HISTORY_VIEWS and close is None:
        raise ValueError(f"View '{name}' needs price history (run `acs fetch`).")

    if name == "returns":
        return View(name, features.correlation_distance(close), "ward")

    if name in ("risk", "liquidity", "profile"):
        feats = features.return_features(close, volume, local_close)
        cols = {
            "risk": features.RISK_COLS,
            "liquidity": features.LIQUIDITY_COLS,
            "profile": list(feats.columns),
        }[name]
        feats = feats[[c for c in cols if c in feats.columns]].dropna(axis=1, how="all")
        return View(name, feature_distance(feats), "ward", feats)

    if name == "fundamentals":
        if fundamentals is None or fundamentals.empty:
            raise ValueError("No fundamentals found (run `acs fetch --fundamentals`).")
        feats = features.fundamental_features(fundamentals)
        if feats.empty:
            raise ValueError("Fundamentals have no numeric field with enough coverage.")
        if close is not None:  # cluster the same stocks the other views use
            feats = feats.loc[feats.index.intersection(close.columns)]
        feats = feats.dropna(how="all")
        return View(name, feature_distance(feats), "ward", feats)

    raise ValueError(f"Unknown view '{name}'. Choose from {VIEW_NAMES}.")


def cluster(view: View, k: int) -> pd.Series:
    labels, _ = clustering.hierarchical(view.distance, k, method=view.method)
    return labels


def linkage_matrix(view: View) -> np.ndarray:
    return clustering.hierarchical(view.distance, 2, method=view.method)[1]
