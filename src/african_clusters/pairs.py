"""Mean-reversion pair candidates found within clusters.

Clusters narrow the search: testing every pair across about 900 stocks invites
false discoveries, so only pairs in the same cluster are tested. For each pair:

1. Engle-Granger cointegration on the **train** part of the log prices, with a
   Benjamini-Hochberg q-value across every test run.
2. Hedge ratio and spread half-life estimated on train.
3. **Out-of-sample check**: ADF test on the *test* part of the spread, using the
   train hedge ratio (does the relationship survive?).
4. Current spread z-score (relative to train mean/std) as a *candidate* entry
   signal. This screens for ideas; it is not a trading recommendation.

Illiquid stocks are excluded: a price that rarely changes can look
cointegrated with anything.
"""

from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint

from . import features


def half_life(spread: pd.Series) -> float:
    lagged = spread.shift(1).dropna()
    delta = spread.diff().dropna()
    lam = np.polyfit(lagged.to_numpy(), delta.loc[lagged.index].to_numpy(), 1)[0]
    return float(-np.log(2) / lam) if lam < 0 else np.inf


def bh_qvalues(p: pd.Series) -> pd.Series:
    order = p.sort_values()
    n = len(order)
    q = (order * n / np.arange(1, n + 1))[::-1].cummin()[::-1].clip(upper=1)
    return q.reindex(p.index)


def pair_candidates(
    close: pd.DataFrame,
    labels: pd.Series,
    train_frac: float = 0.7,
    min_corr: float = 0.3,
    max_zero_frac: float = 0.3,
    min_obs: int = 150,
    local_close: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """``close`` in a common currency (needed for cross-exchange pairs); liquidity from ``local_close``."""
    rets = features.log_returns(close if local_close is None else local_close)
    zero = (rets == 0).sum() / rets.notna().sum().clip(lower=1)
    liquid = set(zero.index[zero <= max_zero_frac])
    logp = np.log(close)
    rows = []
    for c, members in labels.groupby(labels).groups.items():
        members = [m for m in members if m in liquid]
        for a, b in combinations(members, 2):
            x = logp[[a, b]].dropna()
            if len(x) < min_obs:
                continue
            cut = int(len(x) * train_frac)
            tr, te = x.iloc[:cut], x.iloc[cut:]
            corr = tr.diff().corr().iloc[0, 1]
            if not np.isfinite(corr) or corr < min_corr:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p_train = coint(tr[a], tr[b])[1]
                beta, alpha = np.polyfit(tr[b], tr[a], 1)
                spread_tr = tr[a] - beta * tr[b] - alpha
                spread_te = te[a] - beta * te[b] - alpha
                p_test = adfuller(spread_te, maxlag=5)[1] if len(te) > 30 else np.nan
            mu, sd = spread_tr.mean(), spread_tr.std()
            z_now = (spread_te.iloc[-1] - mu) / sd if sd > 0 else np.nan
            rows.append({
                "cluster": c, "a": a, "b": b, "hedge_ratio": beta, "train_corr": corr,
                "p_train": p_train, "p_test": p_test, "half_life_days": half_life(spread_tr),
                "z_now": z_now,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["q_train"] = bh_qvalues(df["p_train"])
    df["holds_oos"] = df["p_test"] < 0.10
    df["candidate"] = (df["q_train"] < 0.10) & df["holds_oos"] & df["half_life_days"].between(2, 60)
    df["signal"] = np.select(
        [df["candidate"] & (df["z_now"] >= 2), df["candidate"] & (df["z_now"] <= -2)],
        ["spread rich: short a / long b", "spread cheap: long a / short b"],
        default="",
    )
    return df.sort_values(["candidate", "q_train"], ascending=[False, True]).reset_index(drop=True)
