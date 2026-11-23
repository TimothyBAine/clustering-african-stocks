import numpy as np
import pandas as pd
import pytest

from african_clusters import clustering, data, features
from african_clusters.mansa_client import MansaAPIError, MansaClient, _extract_list


def synthetic_panel(n_days=300, seed=0):
    """Two latent factors -> two obvious groups of stocks."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    f1, f2 = rng.normal(0, 0.01, (2, n_days))
    cols = {}
    for i in range(6):
        cols[f"NGX:A{i}"] = f1 + rng.normal(0, 0.003, n_days)
        cols[f"JSE:B{i}"] = f2 + rng.normal(0, 0.003, n_days)
    rets = pd.DataFrame(cols, index=dates)
    return 100 * np.exp(rets.cumsum())


def test_extract_list_handles_envelopes():
    assert _extract_list({"stocks": [1], "count": 1}, "stocks") == [1]
    assert _extract_list({"success": True, "data": [{"a": 1}]}, "stocks") == [{"a": 1}]
    assert _extract_list({"success": True, "data": {"history": [2]}}, "history") == [2]
    with pytest.raises(MansaAPIError):
        _extract_list({"x": 1}, "stocks")


def test_normalize_history_aliases():
    rows = [
        {"trade_date": "2024-01-03", "Close_Price": "10.5", "vol": 100},
        {"trade_date": "2024-01-02", "Close_Price": "10.0", "vol": 50},
    ]
    df = data.normalize_history(rows)
    assert list(df.columns) == ["close", "volume"]
    assert df.index.is_monotonic_increasing
    assert df["close"].iloc[-1] == 10.5


def test_normalize_history_prefers_adj_close():
    rows = [{"date": "2024-01-02", "open": 1, "high": 1, "low": 1, "close": 10,
             "adj_close": 9, "volume": 5}]
    df = data.normalize_history(rows)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df["close"].iloc[0] == 9


def test_correlation_clusters_recover_groups():
    close = synthetic_panel()
    labels, _ = clustering.hierarchical(features.correlation_distance(close), k=2)
    a = labels[labels.index.str.startswith("NGX")].nunique()
    b = labels[labels.index.str.startswith("JSE")].nunique()
    assert a == 1 and b == 1
    assert labels.nunique() == 2


def test_return_features_and_kmeans():
    close = synthetic_panel()
    feats = features.return_features(close)
    assert {"ann_vol", "beta", "zero_return_frac"} <= set(feats.columns)
    labels = clustering.kmeans(feats, k=2)
    assert len(labels) == close.shape[1]


def test_client_retries_timeouts(tmp_path, monkeypatch):
    import requests
    client = MansaClient(api_key="t", base_url="https://example.invalid", cache_dir=tmp_path,
                         min_interval=0)
    monkeypatch.setattr("african_clusters.mansa_client.time.sleep", lambda s: None)
    calls = []

    class Resp:
        status_code, ok = 200, True
        def json(self):
            return {"data": [1]}

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) < 3:
            raise requests.ReadTimeout("slow")
        return Resp()

    monkeypatch.setattr(client.session, "get", flaky)
    assert client.get("x") == {"data": [1]}
    assert len(calls) == 3

    monkeypatch.setattr(client.session, "get",
                        lambda *a, **k: (_ for _ in ()).throw(requests.ReadTimeout("slow")))
    with pytest.raises(MansaAPIError, match="Network error"):
        client.get("y")


def test_client_uses_cache(tmp_path, monkeypatch):
    client = MansaClient(api_key="test", base_url="https://example.invalid", cache_dir=tmp_path)
    calls = []

    class Resp:
        status_code, ok = 200, True
        def json(self):
            return {"stocks": [{"ticker": "MTNN"}], "count": 1}

    monkeypatch.setattr(client.session, "get", lambda *a, **k: calls.append(1) or Resp())
    assert client.stocks("NGX") == [{"ticker": "MTNN"}]
    assert client.stocks("NGX") == [{"ticker": "MTNN"}]
    assert len(calls) == 1
    assert client.session.headers["Authorization"] == "Bearer test"
