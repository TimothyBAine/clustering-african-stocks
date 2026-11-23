"""Test what "similar" means by comparing views with each other and with what happens next.

Three questions:

1. **Do the views agree?** Compare partitions with ARI/NMI, and the raw distance
   matrices with a Mantel test, which needs no choice of k.
2. **Are they just exchange or sector in disguise?** Compare each partition
   with the ``exchange`` and ``sector`` labels.
3. **What does each view predict?** Build the views on a *training* window,
   then measure on the following *test* window whether stocks in the same cluster
   still (a) move together, (b) have similar risk, (c) have similar liquidity.
   Each score is compared with random partitions of the same cluster sizes,
   so the scores can be read as z-scores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from . import features


# --------------------------------------------------------------- agreement
def partition_agreement(labelings: dict[str, pd.Series], metric: str = "ari") -> pd.DataFrame:
    """Pairwise ARI (or NMI) between labelings, on the stocks they share."""
    score = adjusted_rand_score if metric == "ari" else normalized_mutual_info_score
    names = list(labelings)
    out = pd.DataFrame(np.nan, index=names, columns=names)
    for a in names:
        for b in names:
            common = labelings[a].dropna().index.intersection(labelings[b].dropna().index)
            if len(common) >= 3:
                out.loc[a, b] = score(labelings[a][common], labelings[b][common])
    return out


def mantel(d1: pd.DataFrame, d2: pd.DataFrame, n_perm: int = 499, seed: int = 0) -> tuple[float, float]:
    """Spearman correlation between two distance matrices + permutation p-value."""
    common = d1.index.intersection(d2.index)
    a, b = d1.loc[common, common].to_numpy(), d2.loc[common, common].to_numpy()
    iu = np.triu_indices(len(common), 1)
    r = spearmanr(a[iu], b[iu]).statistic
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        p = rng.permutation(len(common))
        hits += spearmanr(a[iu], b[p][:, p][iu]).statistic >= r
    return float(r), (hits + 1) / (n_perm + 1)


def distance_agreement(distances: dict[str, pd.DataFrame], n_perm: int = 199) -> pd.DataFrame:
    """Mantel r for every pair of views (upper triangle) with p-values (lower)."""
    names = list(distances)
    out = pd.DataFrame(np.nan, index=names, columns=names)
    for i, a in enumerate(names):
        out.loc[a, a] = 1.0
        for b in names[i + 1:]:
            r, p = mantel(distances[a], distances[b], n_perm=n_perm)
            out.loc[a, b], out.loc[b, a] = r, p
    return out


# ------------------------------------------------- out-of-sample prediction
def _eta2(values: pd.Series, labels: pd.Series) -> float:
    """Share of the variance in ``values`` that the grouping explains (between-group SS / total SS)."""
    v = values.dropna()
    g = labels.reindex(v.index)
    total = ((v - v.mean()) ** 2).sum()
    if total == 0 or len(v) < 3:
        return np.nan
    between = v.groupby(g).transform("mean")
    return float(((between - v.mean()) ** 2).sum() / total)


def outcome_scores(labels: pd.Series, test_corr: pd.DataFrame, test_feats: pd.DataFrame) -> dict:
    """How well a partition (made on the train window) describes the test window."""
    keys = labels.index.intersection(test_corr.index)
    lab = labels[keys].to_numpy()
    c = test_corr.loc[keys, keys].to_numpy()
    same = lab[:, None] == lab[None, :]
    off = ~np.eye(len(keys), dtype=bool)
    within = np.nanmean(c[same & off]) if (same & off).any() else np.nan
    between = np.nanmean(c[~same]) if (~same).any() else np.nan
    return {
        "comovement_gap": within - between,
        "within_corr": within,
        "between_corr": between,
        "risk_eta2": _eta2(test_feats["ann_vol"], labels),
        "drawdown_eta2": _eta2(test_feats["max_drawdown"], labels),
        "liquidity_eta2": _eta2(test_feats["zero_return_frac"], labels),
    }


SCORE_COLS = ["comovement_gap", "risk_eta2", "drawdown_eta2", "liquidity_eta2"]


def evaluate_partitions(
    partitions: dict[str, pd.Series],
    test_close: pd.DataFrame,
    n_perm: int = 200,
    seed: int = 0,
    test_local: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Score each partition on the test window, with z-scores against random partitions of the same sizes."""
    rets = features.log_returns(test_close)
    test_corr = rets.corr(min_periods=40)
    test_feats = features.return_features(test_close, local_close=test_local)
    rng = np.random.default_rng(seed)
    rows = []
    for name, labels in partitions.items():
        labels = labels.dropna()
        labels = labels[labels.index.isin(test_close.columns)]
        if labels.nunique() < 2:
            continue
        obs = outcome_scores(labels, test_corr, test_feats)
        null = pd.DataFrame(
            [
                outcome_scores(pd.Series(rng.permutation(labels.values), index=labels.index),
                               test_corr, test_feats)
                for _ in range(n_perm)
            ]
        )
        row = {"partition": name, "n_stocks": len(labels), "n_clusters": labels.nunique()}
        for col in SCORE_COLS:
            sd = null[col].std()
            row[col] = obs[col]
            row[f"{col}_z"] = (obs[col] - null[col].mean()) / sd if sd > 0 else np.nan
        row["within_corr"], row["between_corr"] = obs["within_corr"], obs["between_corr"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("partition")


def split_by_time(close: pd.DataFrame, train_frac: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    cut = int(len(close) * train_frac)
    return close.iloc[:cut], close.iloc[cut:]
