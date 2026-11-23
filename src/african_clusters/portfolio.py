"""Turn clusters into portfolio decisions, and check whether they help out of sample.

Strategies, each re-estimated on a trailing window and then held for the next one:

* ``equal_weight``    1/N across the tradable universe (baseline)
* ``inverse_vol``     weight proportional to 1/sigma (baseline)
* ``hrp``             Hierarchical Risk Parity (Lopez de Prado, 2016)
* ``cluster_parity``  equal budget per cluster, inverse-vol inside each cluster
* ``cluster_reps``    one liquid representative per cluster, equal weight; a
                      compact portfolio for markets where holding hundreds of
                      thinly traded names is impractical

"Tradable" means at most ``max_zero_frac`` of days with no price change and
enough history in the window, judged on *local* prices. Returns are in the
analysis currency (USD by default). Weights are reset to target every day
(no drift), so treat results as relative comparisons between strategies, not achievable P&L.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from . import features, views

TRADING_DAYS = 252


# ------------------------------------------------------------------ weights
def _ivp(cov: pd.DataFrame) -> pd.Series:
    iv = 1.0 / np.diag(cov.to_numpy()).clip(min=1e-12)
    return pd.Series(iv / iv.sum(), index=cov.index)


def _cluster_var(cov: pd.DataFrame, keys) -> float:
    sub = cov.loc[keys, keys]
    w = _ivp(sub).to_numpy()
    return float(w @ sub.to_numpy() @ w)


def hrp_weights(cov: pd.DataFrame) -> pd.Series:
    sd = np.sqrt(np.diag(cov.to_numpy())).clip(min=1e-12)
    corr = (cov.to_numpy() / np.outer(sd, sd)).clip(-1, 1)
    dist = np.sqrt(0.5 * (1 - corr))
    np.fill_diagonal(dist, 0.0)
    order = cov.index[leaves_list(linkage(squareform(dist, checks=False), "single"))]
    w = pd.Series(1.0, index=order)
    stack = [list(order)]
    while stack:
        stack = [half for c in stack if len(c) > 1
                 for half in (c[: len(c) // 2], c[len(c) // 2:])]
        for left, right in zip(stack[::2], stack[1::2]):
            vl, vr = _cluster_var(cov, left), _cluster_var(cov, right)
            alpha = 1 - vl / (vl + vr)
            w[left] *= alpha
            w[right] *= 1 - alpha
    return w.reindex(cov.index)


def cluster_parity_weights(cov: pd.DataFrame, labels: pd.Series) -> pd.Series:
    labels = labels.reindex(cov.index).dropna()
    out = pd.Series(0.0, index=cov.index)
    groups = labels.groupby(labels).groups
    for members in groups.values():
        out[members] = _ivp(cov.loc[members, members]) / len(groups)
    return out / out.sum()


def representatives(
    labels: pd.Series, distance: pd.DataFrame, liquidity: pd.DataFrame | None = None
) -> pd.Series:
    """One stock per cluster: the medoid of the cluster's most liquid half."""
    reps = {}
    for c, members in labels.groupby(labels).groups.items():
        members = list(members)
        if liquidity is not None and len(members) > 2:
            liq = liquidity.reindex(members)["zero_return_frac"].sort_values()
            members = list(liq.index[: max(2, len(members) // 2)])
        d = distance.loc[members, members]
        reps[c] = d.sum(axis=1).idxmin()
    return pd.Series(reps, name="representative")


def tradable(close: pd.DataFrame, max_zero_frac: float = 0.5, min_coverage: float = 0.8) -> list[str]:
    """Pass *local-currency* prices: zero-return days measure trading, not FX."""
    rets = features.log_returns(close)
    cov = rets.notna().mean()
    zero = (rets == 0).sum() / rets.notna().sum().clip(lower=1)
    return list(close.columns[(cov >= min_coverage) & (zero <= max_zero_frac)])


def target_weights(
    close: pd.DataFrame,
    volume: pd.DataFrame | None,
    view_name: str,
    k: int,
    fundamentals: pd.DataFrame | None = None,
    max_zero_frac: float = 0.5,
    local_close: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Weights for every strategy from one estimation window, plus that window's cluster labels.

    ``close`` is in the portfolio's currency; ``local_close`` (traded prices) drives liquidity.
    """
    local = close if local_close is None else local_close.reindex(index=close.index, columns=close.columns)
    keys = tradable(local, max_zero_frac)
    if len(keys) <= k:
        raise ValueError(f"Only {len(keys)} tradable stocks in the window; need more than k={k}.")
    win = close[keys]
    simple = np.expm1(features.log_returns(win)).fillna(0.0)
    cov = simple.cov()
    vol = None if volume is None else volume.reindex(index=win.index, columns=keys)
    view = views.build_view(view_name, win, vol, fundamentals, local[keys])
    labels = views.cluster(view, k)
    liq = features.return_features(win, local_close=local[keys])[["zero_return_frac"]]
    reps = representatives(labels, view.distance, liq)

    w = pd.DataFrame(index=keys)
    w["equal_weight"] = 1.0 / len(keys)
    w["inverse_vol"] = _ivp(cov)
    w["hrp"] = hrp_weights(cov)
    w["cluster_parity"] = cluster_parity_weights(cov, labels)
    w["cluster_reps"] = 0.0
    w.loc[reps.values, "cluster_reps"] = 1.0 / len(reps)
    return w.fillna(0.0), labels


# ----------------------------------------------------------------- backtest
def walk_forward(
    close: pd.DataFrame,
    volume: pd.DataFrame | None,
    view_name: str,
    k: int,
    fundamentals: pd.DataFrame | None = None,
    train: int = 252,
    step: int = 63,
    cost_bps: float = 50.0,
    max_zero_frac: float = 0.5,
    progress: Callable[[str], None] | None = None,
    local_close: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-sample daily returns per strategy, plus turnover per rebalance."""
    simple_all = np.expm1(features.log_returns(close))
    rets, turns = [], []
    prev: pd.DataFrame | None = None
    for start in range(train, len(simple_all), step):
        # simple_all.iloc[start] is the return from close[start] to close[start+1], so
        # estimate on prices up to and including close[start]: no look-ahead.
        est = close.iloc[start - train: start + 1]
        w, _ = target_weights(est, volume, view_name, k, fundamentals, max_zero_frac, local_close)
        test = simple_all.iloc[start: start + step]
        if test.empty:
            break
        period = test.reindex(columns=w.index).fillna(0.0) @ w
        all_keys = w.index if prev is None else w.index.union(prev.index)
        w_full = w.reindex(all_keys, fill_value=0.0)
        prev_full = (pd.DataFrame(0.0, index=all_keys, columns=w.columns) if prev is None
                     else prev.reindex(all_keys, fill_value=0.0))
        turnover = (w_full - prev_full).abs().sum()  # initial build counts as 100% turnover
        period.iloc[0] -= turnover * cost_bps / 1e4
        rets.append(period)
        turns.append(turnover.rename(test.index[0]))
        prev = w
        if progress:
            progress(f"rebalance {test.index[0].date()}: {len(w)} tradable stocks")
    if not rets:
        raise ValueError(f"Not enough history: need more than train={train} days.")
    return pd.concat(rets), pd.DataFrame(turns)


def performance(rets: pd.DataFrame, turnover: pd.DataFrame | None = None) -> pd.DataFrame:
    eq = (1 + rets).cumprod()
    years = len(rets) / TRADING_DAYS
    out = pd.DataFrame({
        "ann_return": eq.iloc[-1] ** (1 / years) - 1,
        "ann_vol": rets.std() * np.sqrt(TRADING_DAYS),
        "max_drawdown": (eq / eq.cummax() - 1).min(),
        "worst_day": rets.min(),
    })
    out["sharpe"] = out["ann_return"] / out["ann_vol"]
    out["return_per_drawdown"] = out["ann_return"] / out["max_drawdown"].abs()
    if turnover is not None and len(turnover) > 1:
        out["avg_turnover"] = turnover.iloc[1:].mean()  # skip the initial build
    return out


# ----------------------------------------------------------------- exposure
def load_holdings(path, universe_keys) -> pd.Series:
    """CSV with ``ticker`` and ``weight`` columns; bare tickers are matched to EXCHANGE:TICKER."""
    h = pd.read_csv(path)
    h.columns = [c.lower() for c in h.columns]
    by_ticker = {}
    for key in universe_keys:
        by_ticker.setdefault(key.split(":", 1)[1], []).append(key)
    out = {}
    for tk, wt in zip(h["ticker"].astype(str), h["weight"].astype(float)):
        if ":" in tk:
            key = tk
        elif len(by_ticker.get(tk, [])) == 1:
            key = by_ticker[tk][0]
        else:
            raise ValueError(f"Ticker '{tk}' is missing or ambiguous; write it as EXCHANGE:{tk}.")
        out[key] = out.get(key, 0.0) + wt
    w = pd.Series(out)
    return w / w.sum()


def cluster_exposure(weights: pd.Series, labels: pd.Series, cov: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Capital and risk share per cluster, plus effective number of clusters."""
    keys = weights.index.intersection(cov.index).intersection(labels.index)
    missing = weights.index.difference(keys)
    w = weights[keys] / weights[keys].sum()
    S = cov.loc[keys, keys]
    port_var = float(w @ S @ w)
    rc = w * (S @ w) / port_var  # each holding's share of portfolio variance
    lab = labels[keys]
    table = pd.DataFrame({"weight": w.groupby(lab).sum(), "risk_share": rc.groupby(lab).sum(),
                          "n_holdings": lab.value_counts()})
    table = table.reindex(sorted(labels.unique()), fill_value=0.0)
    table["universe_size"] = labels.value_counts().reindex(table.index)
    rs = table["risk_share"].clip(lower=0)
    summary = {
        "port_vol_ann": float(np.sqrt(port_var * TRADING_DAYS)),
        "effective_clusters_capital": float(1 / (table["weight"] ** 2).sum()),
        "effective_clusters_risk": float(1 / ((rs / rs.sum()) ** 2).sum()),
        "clusters_held": int((table["n_holdings"] > 0).sum()),
        "clusters_total": len(table),
        "unmatched_holdings": list(missing),
    }
    return table, summary
