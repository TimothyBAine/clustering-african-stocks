"""Thin REST client for the Mansa African markets API, with an on-disk cache.

Docs: https://mansamarkets.com/developers
Auth: ``Authorization: Bearer <MANSA_API_KEY>``

The free tier allows ~100 requests/day, so every response is cached as JSON
under ``data/raw/`` and reused unless ``refresh=True``.

Response envelopes vary between endpoints (some wrap the payload in
``{"success": true, "data": ...}``, some don't), so the helpers below unwrap
defensively.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import requests

from . import config


class MansaAPIError(RuntimeError):
    pass


class MansaClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        cache_dir: Path | None = None,
        timeout: float = 30.0,
        min_interval: float = 0.5,
        retries: int = 3,
    ) -> None:
        self.retries = retries
        self.api_key = api_key or config.api_key()
        self.base_url = (base_url or config.base_url()).rstrip("/")
        self.cache_dir = Path(cache_dir or config.RAW_DIR)
        self.timeout = timeout
        self.min_interval = min_interval
        self._last_call = 0.0
        self.session = requests.Session()
        self.session.headers.update(
            {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
        )

    # ------------------------------------------------------------------ core
    def _cache_path(self, path: str, params: dict[str, Any] | None) -> Path:
        slug = path.strip("/").replace("/", "__")
        if params:
            blob = json.dumps(params, sort_keys=True)
            slug += "__" + hashlib.md5(blob.encode()).hexdigest()[:10]
        return self.cache_dir / f"{slug}.json"

    def get(
        self, path: str, params: dict[str, Any] | None = None, refresh: bool = False
    ) -> Any:
        params = {k: v for k, v in (params or {}).items() if v is not None}
        cache_file = self._cache_path(path, params)
        if cache_file.exists() and not refresh:
            return json.loads(cache_file.read_text())

        url = f"{self.base_url}/{path.lstrip('/')}"
        resp = None
        for attempt in range(self.retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except (requests.Timeout, requests.ConnectionError) as e:
                resp, err = None, e
            finally:
                self._last_call = time.monotonic()
            if resp is not None and resp.status_code < 500:
                break
            if attempt < self.retries:
                time.sleep(2 ** attempt)  # 1s, 2s, 4s
        if resp is None:
            raise MansaAPIError(f"Network error for {path} after {self.retries + 1} tries: {err}")

        if resp.status_code == 429:
            raise MansaAPIError("Rate limit hit (429). Free tier is ~100 requests/day.")
        if resp.status_code in (401, 403):
            raise MansaAPIError(
                f"{resp.status_code} for {path}: check your API key / plan tier. "
                "Stock history requires the Professional tier."
            )
        if not resp.ok:
            raise MansaAPIError(f"{resp.status_code} for {path}: {resp.text[:300]}")

        payload = resp.json()
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(payload))
        return payload

    # ------------------------------------------------------------- endpoints
    def exchanges(self, refresh: bool = False) -> list[dict]:
        return _extract_list(self.get("markets/exchanges", refresh=refresh), "exchanges")

    def stocks(
        self, exchange: str, sector: str | None = None, refresh: bool = False
    ) -> list[dict]:
        out: list[dict] = []
        offset, limit = 0, 200
        while True:
            data = self.get(
                f"markets/exchanges/{exchange}/stocks",
                {"sector": sector, "limit": limit, "offset": offset or None},
                refresh=refresh,
            )
            page = _extract_list(data, "stocks")
            out.extend(page)
            pag = data.get("pagination", {}) if isinstance(data, dict) else {}
            if not pag.get("has_more") or not page:
                return out
            offset += len(page)

    def stock(self, exchange: str, ticker: str, refresh: bool = False) -> dict:
        return _unwrap(self.get(f"markets/exchanges/{exchange}/stocks/{ticker}", refresh=refresh))

    def fundamentals(self, exchange: str, ticker: str, refresh: bool = False) -> dict:
        return _unwrap(
            self.get(f"markets/exchanges/{exchange}/stocks/{ticker}/fundamentals", refresh=refresh)
        )

    def history(
        self,
        exchange: str,
        ticker: str,
        range_: str | None = "1Y",
        start: str | None = None,
        end: str | None = None,
        refresh: bool = False,
    ) -> list[dict]:
        """Daily OHLCV rows. Requires the Professional tier.

        ``range_`` is one of 1M..ALL; ``start``/``end`` are YYYY-MM-DD and take
        precedence when given.
        """
        params = {"range": None if start else range_, "from": start, "to": end, "order": "asc"}
        data = self.get(
            f"markets/exchanges/{exchange}/stocks/{ticker}/history", params, refresh=refresh
        )
        return _extract_list(data, "history", "prices", "data")


# ---------------------------------------------------------------- helpers
def _unwrap(payload: Any) -> Any:
    if isinstance(payload, dict) and "data" in payload and set(payload) <= {
        "success", "data", "meta", "message", "count", "pagination",
    }:
        return payload["data"]
    return payload


def _extract_list(payload: Any, *keys: str) -> list[dict]:
    payload = _unwrap(payload)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for k in keys:
            if isinstance(payload.get(k), list):
                return payload[k]
        for v in payload.values():  # fall back to the first list-valued field
            if isinstance(v, list):
                return v
    raise MansaAPIError(f"Could not find a list in response (looked for {keys})")
