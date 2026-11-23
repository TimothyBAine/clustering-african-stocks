"""Is a cluster real, or an accident of this particular sample?

* **Bootstrap stability** (Hennig's ``clusterboot`` idea). History-based views
  resample *days* in blocks, which keeps short-range autocorrelation and the
  stale-price effects of illiquid stocks. The fundamentals view has one row per
  stock, so it resamples *stocks* instead. Each resample is reclustered and scored:
    - ARI against the full-sample clustering;
    - per-cluster Jaccard recovery (>0.75 stable, 0.6-0.75 a real pattern
      whose exact members are uncertain, <0.5 the cluster falls apart);
    - per-stock confidence: how often a stock stays with its cluster-mates.
* **Temporal stability**: cluster rolling windows and measure how much the
  membership changes from one window to the next.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from . import features, views


def block_bootstrap_index(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, max(1, n - block + 1), size=int(np.ceil(n / block)))
    return np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]


def _resampled_view(
    name: str,
    close: pd.DataFrame | None,
    volume: pd.DataFrame | None,
    fundamentals: pd.DataFrame | None,
    rng: np.random.Generator,
    block: int,
    local_close: pd.DataFrame | None = None,
) -> views.View:
    if name in views.HISTORY_VIEWS:
        rets = features.log_returns(close)
        idx = block_bootstrap_index(len(rets), block, rng)
        r = rets.iloc[idx].reset_index(drop=True)
        r.index = pd.bdate_range("2000-01-03", periods=len(r))
        boot_close = features.prices_from_returns(r)
        boot_local = None
        if local_close is not None:  # resample the same days in local currency
            lr = features.log_returns(local_close).reindex(rets.index).iloc[idx]
            lr.index = r.index
            boot_local = features.prices_from_returns(lr)
        boot_vol = None
        if volume is not None and not volume.empty:
            v = volume.reindex(rets.index).iloc[idx].reset_index(drop=True)
            v.index = r.index
            boot_vol = v
        return views.build_view(name, boot_close, boot_vol, local_close=boot_local)
    # Cross-sectional view: resample stocks with replacement, keep unique ones.
    full = views.build_view(name, close, volume, fundamentals)
    keys = full.distance.index.to_numpy()
    picked = np.unique(rng.choice(keys, size=len(keys), replace=True))
    return full.subset(picked)


def jaccard_recovery(base: pd.Series, boot: pd.Series) -> pd.Series:
    """For each base cluster, best Jaccard overlap with any bootstrap cluster."""
    common = base.index.intersection(boot.index)
    b, o = base[common], boot[common]
    out = {}
    for c in sorted(base.unique()):
        members = set(b.index[b == c])
        if not members:
            out[c] = np.nan
            continue
        out[c] = max(
            len(members & set(o.index[o == d])) / len(members | set(o.index[o == d]))
            for d in o.unique()
        )
    return pd.Series(out)


def bootstrap(
    view_name: str,
    k: int,
    close: pd.DataFrame | None = None,
    volume: pd.DataFrame | None = None,
    fundamentals: pd.DataFrame | None = None,
    n_boot: int = 100,
    block: int = 20,
    seed: int = 0,
    local_close: pd.DataFrame | None = None,
) -> dict:
    """Run the bootstrap; returns base labels, ARI per draw, Jaccard table, stock confidence, co-assignment."""
    rng = np.random.default_rng(seed)
    base_view = views.build_view(view_name, close, volume, fundamentals, local_close)
    base = views.cluster(base_view, k)
    keys = base.index
    pos = {key: i for i, key in enumerate(keys)}
    together = np.zeros((len(keys), len(keys)))
    seen = np.zeros((len(keys), len(keys)))
    aris, jacc = [], []

    for _ in range(n_boot):
        bv = _resampled_view(view_name, close, volume, fundamentals, rng, block, local_close)
        bl = views.cluster(bv, k)
        bl = bl[bl.index.isin(pos)]
        common = bl.index
        aris.append(adjusted_rand_score(base[common], bl))
        jacc.append(jaccard_recovery(base, bl))
        ix = np.array([pos[c] for c in common])
        same = (bl.to_numpy()[:, None] == bl.to_numpy()[None, :]).astype(float)
        together[np.ix_(ix, ix)] += same
        seen[np.ix_(ix, ix)] += 1

    coassign = pd.DataFrame(np.divide(together, seen, out=np.full_like(together, np.nan),
                                      where=seen > 0), index=keys, columns=keys)
    jacc_df = pd.DataFrame(jacc)
    clusters = pd.DataFrame({
        "n": base.value_counts().sort_index(),
        "jaccard_mean": jacc_df.mean(),
        "jaccard_p10": jacc_df.quantile(0.10),
        "dissolved_frac": (jacc_df < 0.5).mean(),
    })
    clusters["verdict"] = pd.cut(
        clusters["jaccard_mean"], [-np.inf, 0.5, 0.6, 0.75, np.inf],
        labels=["dissolves", "weak", "pattern", "stable"],
    )
    confidence = []
    for key in keys:
        mates = base.index[(base == base[key]) & (base.index != key)]
        confidence.append(coassign.loc[key, mates].mean() if len(mates) else np.nan)
    stocks = pd.DataFrame({"cluster": base, "confidence": confidence}, index=keys)
    return {
        "labels": base,
        "ari": pd.Series(aris, name="ari"),
        "clusters": clusters,
        "stocks": stocks.sort_values(["cluster", "confidence"]),
        "coassign": coassign,
    }


def stability_by_k(view_name: str, k_values, close=None, volume=None, fundamentals=None,
                   n_boot: int = 30, block: int = 20, seed: int = 0,
                   local_close: pd.DataFrame | None = None) -> pd.DataFrame:
    """Mean bootstrap ARI for each k. Picking k by stability rather than silhouette alone."""
    rows = []
    for k in k_values:
        res = bootstrap(view_name, k, close, volume, fundamentals, n_boot=n_boot, block=block,
                        seed=seed, local_close=local_close)
        rows.append({"k": k, "ari_mean": res["ari"].mean(), "ari_p10": res["ari"].quantile(0.1),
                     "min_cluster_jaccard": res["clusters"]["jaccard_mean"].min()})
    return pd.DataFrame(rows).set_index("k")


def temporal(
    view_name: str,
    k: int,
    close: pd.DataFrame,
    volume: pd.DataFrame | None = None,
    window: int = 250,
    step: int = 63,
    min_obs: int = 120,
    local_close: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster rolling windows. Returns (ARI between consecutive windows, labels per window)."""
    if view_name not in views.HISTORY_VIEWS:
        raise ValueError("Temporal stability needs a history-based view.")
    labels = {}
    for end in range(window, len(close) + 1, step):
        win = features.clean_panel(close.iloc[end - window:end], min_obs=min(min_obs, window // 2))
        vol = None if volume is None else volume.reindex(win.index)
        if win.shape[1] <= k:
            continue
        loc = None if local_close is None else local_close.reindex(index=win.index, columns=win.columns)
        labels[close.index[end - 1].date()] = views.cluster(
            views.build_view(view_name, win, vol, local_close=loc), k)
    lab = pd.DataFrame(labels)
    dates = list(lab.columns)
    rows = []
    for a, b in zip(dates, dates[1:]):
        both = lab[[a, b]].dropna()
        rows.append({"from": a, "to": b, "n_stocks": len(both),
                     "ari": adjusted_rand_score(both[a], both[b]) if len(both) > k else np.nan})
    return pd.DataFrame(rows), lab
