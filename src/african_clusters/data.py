"""Turn raw Mansa responses into tidy DataFrames."""

from __future__ import annotations

import logging

import pandas as pd

from .mansa_client import MansaAPIError, MansaClient

log = logging.getLogger(__name__)

# Map the many names the API might use onto canonical columns.
_COLUMN_ALIASES = {
    "date": ["date", "trade_date", "timestamp", "time", "day"],
    "open": ["open", "open_price"],
    "high": ["high", "high_price"],
    "low": ["low", "low_price"],
    # Prefer dividend/split-adjusted prices for return calculations.
    "close": ["adj_close", "adjusted_close", "close", "close_price", "price", "last", "value"],
    "volume": ["volume", "vol", "shares_traded"],
}


def _first_present(cols: list[str], options: list[str]) -> str | None:
    lower = {c.lower(): c for c in cols}
    for o in options:
        if o in lower:
            return lower[o]
    return None


def normalize_history(rows: list[dict]) -> pd.DataFrame:
    """Rows from ``MansaClient.history`` -> DataFrame indexed by date."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Pick exactly one source column per canonical field (e.g. adj_close over
    # close) so we never end up with duplicate column names.
    picked = {}
    for canon, options in _COLUMN_ALIASES.items():
        src = _first_present(list(df.columns), options)
        if src:
            picked[canon] = df[src]
    if "date" not in picked or "close" not in picked:
        raise MansaAPIError(f"History rows lack date/close columns: {list(df.columns)}")
    df = pd.DataFrame(picked)
    keep = list(df.columns)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    for c in keep[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.drop_duplicates("date").set_index("date").sort_index()


def universe(client: MansaClient, exchanges: list[str], refresh: bool = False) -> pd.DataFrame:
    """Listed stocks across exchanges (one API call per exchange)."""
    frames = []
    for ex in exchanges:
        df = pd.DataFrame(client.stocks(ex, refresh=refresh))
        if df.empty:
            continue
        if "ticker" not in df.columns:
            alt = _first_present(list(df.columns), ["symbol", "code"])
            if alt:
                df = df.rename(columns={alt: "ticker"})
        df["exchange"] = ex
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["exchange", "ticker"])
    return pd.concat(frames, ignore_index=True)


def fundamentals(
    client: MansaClient, keys: list[str], refresh: bool = False
) -> pd.DataFrame:
    """One row of fundamentals per ``EXCHANGE:TICKER`` key (one call each)."""
    rows = []
    for key in keys:
        ex, tk = key.split(":", 1)
        try:
            rec = client.fundamentals(ex, tk, refresh=refresh)
        except MansaAPIError as e:
            log.warning("No fundamentals for %s: %s", key, e)
            if "429" in str(e) or "tier" in str(e):
                raise
            continue
        if isinstance(rec, dict):
            rows.append({**rec, "exchange": ex, "ticker": tk})
    return pd.DataFrame(rows)


def price_panel(
    client: MansaClient,
    stocks: pd.DataFrame,
    range_: str = "2Y",
    max_stocks: int | None = None,
    refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch history for each (exchange, ticker).

    Returns ``(close, volume)`` wide DataFrames with columns ``EXCHANGE:TICKER``.
    Stocks that fail to fetch are logged and skipped.
    """
    closes, volumes = {}, {}
    rows = stocks[["exchange", "ticker"]].drop_duplicates()
    if max_stocks:
        rows = rows.head(max_stocks)
    for ex, tk in rows.itertuples(index=False):
        key = f"{ex}:{tk}"
        try:
            hist = normalize_history(client.history(ex, tk, range_=range_, refresh=refresh))
        except MansaAPIError as e:
            log.warning("Skipping %s: %s", key, e)
            if "429" in str(e) or "tier" in str(e):
                raise
            continue
        except (ValueError, TypeError, KeyError) as e:  # malformed rows for one stock
            log.warning("Skipping %s (bad data): %s", key, e)
            continue
        if hist.empty:
            continue
        closes[key] = hist["close"]
        if "volume" in hist:
            volumes[key] = hist["volume"]
    return pd.DataFrame(closes).sort_index(), pd.DataFrame(volumes).sort_index()
