"""Feature engineering for clustering.

Two inputs are supported:

* **History-based** (``return_features``): built from a daily close panel.
  Includes an illiquidity measure (share of zero-return days), which matters a
  lot on thinly traded African exchanges.
* **Snapshot-based** (``snapshot_features``): built from the per-exchange
  stock list, which is available on the free tier. Much weaker signal, but
  useful to get the pipeline running.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def clean_panel(
    close: pd.DataFrame, min_obs: int = 120, max_ffill: int = 5
) -> pd.DataFrame:
    """Forward-fill short gaps (holidays/non-trading days) and drop sparse series."""
    close = close.where(close > 0).ffill(limit=max_ffill)
    return close.loc[:, close.notna().sum() >= min_obs]


def log_returns(close: pd.DataFrame) -> pd.DataFrame:
    return np.log(close).diff().iloc[1:]


def prices_from_returns(rets: pd.DataFrame) -> pd.DataFrame:
    """Rebuild a price index (starting at 1) from log returns, keeping NaN gaps."""
    return np.exp(rets.cumsum(skipna=True)).where(rets.notna())


# Column groups used by the similarity views (see ``views.py``).
RISK_COLS = ["ann_vol", "downside_vol", "beta", "max_drawdown", "skew", "kurtosis"]
LIQUIDITY_COLS = ["zero_return_frac", "autocorr_1d", "log_avg_volume"]


def return_features(
    close: pd.DataFrame,
    volume: pd.DataFrame | None = None,
    local_close: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-stock features. ``close`` may be converted to USD.

    Pass the local-currency prices as ``local_close``: the liquidity measures
    (zero-return days, lag-1 autocorrelation) must come from the traded price,
    because after FX conversion a stock that didn't trade still "moves" with the currency.
    """
    rets = log_returns(close)
    liq_rets = rets if local_close is None else log_returns(local_close.reindex(columns=close.columns))
    market = rets.mean(axis=1)  # equal-weighted cross-exchange proxy

    feats = pd.DataFrame(index=close.columns)
    feats["ann_return"] = rets.mean() * TRADING_DAYS
    feats["ann_vol"] = rets.std() * np.sqrt(TRADING_DAYS)
    feats["downside_vol"] = rets.clip(upper=0).pow(2).mean().pow(0.5) * np.sqrt(TRADING_DAYS)
    feats["skew"] = rets.skew()
    feats["kurtosis"] = rets.kurt()
    feats["max_drawdown"] = (close / close.cummax() - 1).min()
    feats["zero_return_frac"] = (liq_rets == 0).sum() / liq_rets.notna().sum()
    feats["beta"] = rets.apply(lambda s: s.cov(market) / market.var())
    feats["autocorr_1d"] = liq_rets.apply(lambda s: s.autocorr(1))

    if volume is not None and not volume.empty:
        vol = volume.reindex(columns=close.columns)
        feats["log_avg_volume"] = np.log1p(vol.mean())

    return feats.replace([np.inf, -np.inf], np.nan)


def correlation_distance(close: pd.DataFrame, min_periods: int = 60) -> pd.DataFrame:
    """sqrt(2 * (1 - rho)) distance matrix between return series."""
    corr = log_returns(close).corr(min_periods=min_periods)
    rho = corr.fillna(0.0).clip(-1, 1).to_numpy(copy=True)
    np.fill_diagonal(rho, 1.0)
    return pd.DataFrame(np.sqrt(2.0 * (1.0 - rho)), index=corr.index, columns=corr.columns)


_FUNDAMENTAL_ALIASES = {
    "market_cap": ["market_cap", "marketcap", "mcap"],
    "pe_ratio": ["pe_ratio", "pe"],
    "pb_ratio": ["pb_ratio", "pb", "price_to_book"],
    "dividend_yield": ["dividend_yield_ttm", "dividend_yield"],
    "roe": ["roe", "return_on_equity"],
    "debt_to_equity": ["debt_to_equity", "de_ratio"],
    "eps_growth": ["eps_growth", "earnings_growth"],
}


def fundamental_features(fund: pd.DataFrame, min_coverage: float = 0.5) -> pd.DataFrame:
    """Numeric fundamentals indexed by ``EXCHANGE:TICKER``.

    Only fields with at least ``min_coverage`` non-missing values are kept, so
    the view grows automatically as the API adds fields (P/E is not yet
    ingested at the time of writing, leaving mostly market cap).
    """
    df = fund.copy()
    df.index = df["exchange"].astype(str) + ":" + df["ticker"].astype(str)
    out = pd.DataFrame(index=df.index)
    for name, opts in _FUNDAMENTAL_ALIASES.items():
        col = next((c for c in opts if c in df.columns), None)
        if col:
            out[name] = pd.to_numeric(df[col], errors="coerce")
    if "market_cap" in out:
        out["log_market_cap"] = np.log(out.pop("market_cap").where(lambda s: s > 0))
    keep = out.columns[out.notna().mean() >= min_coverage]
    return out[keep]


def snapshot_features(stocks: pd.DataFrame) -> pd.DataFrame:
    """Numeric features from the free-tier stock list."""
    df = stocks.copy()
    df.index = df["exchange"].astype(str) + ":" + df["ticker"].astype(str)
    candidates = {
        "change_pct": ["change_pct", "change_percentage", "percent_change"],
        "market_cap": ["market_cap", "marketcap", "mcap"],
        "volume": ["volume"],
        "price": ["price", "close", "last"],
        "pe_ratio": ["pe", "pe_ratio"],
        "dividend_yield": ["dividend_yield"],
    }
    out = pd.DataFrame(index=df.index)
    for name, opts in candidates.items():
        col = next((c for c in opts if c in df.columns), None)
        if col:
            out[name] = pd.to_numeric(df[col], errors="coerce")
    for c in ("market_cap", "volume", "price"):
        if c in out:
            out[f"log_{c}"] = np.log1p(out.pop(c).clip(lower=0))
    return out
