"""Daily FX rates and currency conversion.

Mansa's ``/markets/forex`` only returns today's spot rates, so historical rates
come from the free, keyless fawazahmed0/currency-api (daily snapshots since
2024-03, all African currencies). It is a community project, not an official
central-bank fixing. Each day is one small JSON file, cached under
``data/raw/fx/`` like the Mansa responses, so it is downloaded only once.

Rates are stored as **units of local currency per 1 USD**, so
``price_usd = price_local / rate``.

A price index's returns don't depend on its unit, so JSE prices quoted in
cents (ZAc) convert correctly with the ZAR rate for every return-based statistic.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

from . import config

log = logging.getLogger(__name__)

SOURCES = (
    "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@{date}/v1/currencies/usd.min.json",
    "https://{date}.currency-api.pages.dev/v1/currencies/usd.min.json",
)
EXCHANGE_CURRENCY = {
    "NGX": "NGN", "JSE": "ZAR", "NSE": "KES", "GSE": "GHS", "EGX": "EGP",
    "BRVM": "XOF", "CSE": "MAD", "USE": "UGX", "DSE": "TZS", "LUSE": "ZMW",
    "BSE": "BWP", "SEM": "MUR", "RSE": "RWF", "MSE": "MWK", "NSX": "NAD",
}
FX_DIR = config.RAW_DIR / "fx"


class FXError(RuntimeError):
    pass


def _fetch_day(date: str, cache_dir: Path, refresh: bool, session: requests.Session) -> dict | None:
    path = cache_dir / f"usd_{date}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    for url in SOURCES:
        try:
            resp = session.get(url.format(date=date), timeout=20)
        except requests.RequestException:
            continue
        if resp.ok:
            payload = resp.json()
            path.write_text(json.dumps(payload))
            return payload
        if resp.status_code == 404:  # date not published (e.g. before 2024-03)
            continue
    return None


def usd_rates(
    dates: pd.DatetimeIndex,
    currencies: list[str],
    cache_dir: Path | None = None,
    refresh: bool = False,
    workers: int = 8,
) -> pd.DataFrame:
    """Local-currency-per-USD rates for every calendar day between ``dates`` min and max."""
    cache_dir = Path(cache_dir or FX_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    days = pd.date_range(dates.min().normalize(), dates.max().normalize(), freq="D")
    session = requests.Session()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        payloads = list(pool.map(
            lambda d: _fetch_day(d.strftime("%Y-%m-%d"), cache_dir, refresh, session), days
        ))
    rows = {}
    for day, p in zip(days, payloads):
        if p and "usd" in p:
            rows[day] = {c: p["usd"].get(c.lower()) for c in currencies}
    if not rows:
        raise FXError("No FX history could be downloaded (check your connection).")
    missing = len(days) - len(rows)
    if missing:
        log.warning("FX: %d of %d days unavailable; forward-filled.", missing, len(days))
    rates = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    rates = rates.reindex(days).ffill().bfill()
    rates.index.name = "date"
    rates["USD"] = 1.0
    return rates


def stock_currencies(keys, universe: pd.DataFrame | None = None) -> pd.Series:
    """Currency per ``EXCHANGE:TICKER``: from the universe list, falling back to the exchange default."""
    keys = pd.Index(keys)
    out = pd.Series(keys.str.split(":").str[0].map(EXCHANGE_CURRENCY), index=keys)
    if universe is not None and "currency" in universe.columns:
        u = universe.assign(key=universe["exchange"] + ":" + universe["ticker"].astype(str))
        listed = u.dropna(subset=["currency"]).drop_duplicates("key").set_index("key")["currency"]
        out = listed.reindex(keys).str.upper().fillna(out)
    return out


def to_usd(close: pd.DataFrame, currencies: pd.Series, rates: pd.DataFrame) -> pd.DataFrame:
    """Divide each price series by its currency's daily rate (aligned on date)."""
    unknown = sorted(set(currencies.dropna()) - set(rates.columns))
    no_ccy = currencies.index[currencies.isna()].tolist()
    if unknown or no_ccy:
        raise FXError(f"No FX rate for currencies {unknown} / stocks with no currency {no_ccy[:5]}")
    r = rates.reindex(close.index, method="ffill")
    divisor = r[currencies.reindex(close.columns).to_numpy()]
    divisor.columns = close.columns
    return close / divisor
