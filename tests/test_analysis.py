"""Offline tests for views, comparison, stability, portfolio and pairs modules."""

import numpy as np
import pandas as pd
import pytest

from african_clusters import compare, features, pairs, portfolio, stability, views
from african_clusters.pairs import bh_qvalues


def panel(n_days=400, seed=0):
    """Two co-movement groups; group A is volatile, group B is calm."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    f1, f2 = rng.normal(0, 0.015, n_days), rng.normal(0, 0.005, n_days)
    cols = {}
    for i in range(6):
        cols[f"NGX:A{i}"] = f1 + rng.normal(0, 0.004, n_days)
        cols[f"JSE:B{i}"] = f2 + rng.normal(0, 0.002, n_days)
    rets = pd.DataFrame(cols, index=dates)
    close = 100 * np.exp(rets.cumsum())
    volume = pd.DataFrame(rng.integers(1_000, 100_000, close.shape), index=dates, columns=close.columns)
    return close, volume


def group(labels):
    return labels.index.str.split(":").str[1].str[0]


@pytest.mark.parametrize("name", ["returns", "risk", "liquidity", "profile"])
def test_history_views_build_and_cluster(name):
    close, volume = panel()
    v = views.build_view(name, close, volume)
    assert v.distance.shape == (12, 12)
    assert np.allclose(np.diag(v.distance), 0)
    assert views.cluster(v, 2).nunique() == 2


def test_returns_and_risk_views_recover_groups():
    close, volume = panel()
    for name in ("returns", "risk"):
        labels = views.cluster(views.build_view(name, close, volume), 2)
        assert pd.crosstab(labels, group(labels)).gt(0).sum().eq(1).all(), name


def test_fundamental_features_coverage_and_log():
    fund = pd.DataFrame({
        "exchange": ["NGX"] * 4, "ticker": list("ABCD"),
        "market_cap": [1e9, 1e10, 1e11, 1e12], "pe_ratio": [None, None, None, 5.0],
        "sector": ["Banks", "Banks", "ICT", "ICT"],
    })
    f = features.fundamental_features(fund)
    assert list(f.columns) == ["log_market_cap"]  # P/E coverage 25% < 50%
    assert f.index[0] == "NGX:A"
    v = views.build_view("fundamentals", fundamentals=fund)
    assert views.cluster(v, 2).nunique() == 2


def test_partition_agreement_and_mantel():
    close, _ = panel()
    d = features.correlation_distance(close)
    r, p = compare.mantel(d, d, n_perm=50)
    assert r == pytest.approx(1.0) and p < 0.05
    labs = views.cluster(views.build_view("returns", close), 2)
    ari = compare.partition_agreement({"x": labs, "ref": pd.Series(group(labs), index=labs.index)})
    assert ari.loc["x", "ref"] == pytest.approx(1.0)


def test_out_of_sample_scores_favour_real_structure():
    close, volume = panel()
    train, test = compare.split_by_time(close)
    labs = views.cluster(views.build_view("returns", train), 2)
    rng = np.random.default_rng(1)
    rand = pd.Series(rng.integers(0, 2, len(labs)), index=labs.index)
    scores = compare.evaluate_partitions({"returns": labs, "random": rand}, test, n_perm=100)
    assert scores.loc["returns", "comovement_gap_z"] > 2
    assert scores.loc["returns", "risk_eta2"] > 0.8  # groups also differ in vol


def test_bootstrap_stability_on_clear_structure():
    close, volume = panel()
    res = stability.bootstrap("returns", 2, close, volume, n_boot=15, block=20)
    assert res["ari"].mean() > 0.9
    assert (res["clusters"]["verdict"] == "stable").all()
    assert res["stocks"]["confidence"].min() > 0.9
    assert res["coassign"].shape == (12, 12)


def test_bootstrap_stability_flags_noise():
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2024-01-01", periods=300)
    close = pd.DataFrame(100 * np.exp(rng.normal(0, 0.01, (300, 12)).cumsum(0)), index=dates,
                         columns=[f"NGX:N{i}" for i in range(12)])
    res = stability.bootstrap("returns", 3, close, n_boot=15)
    assert res["ari"].mean() < 0.5


def test_bootstrap_cross_sectional_view():
    fund = pd.DataFrame({"exchange": "NGX", "ticker": [f"T{i}" for i in range(12)],
                         "market_cap": [1e9] * 6 + [1e12] * 6})
    fund["market_cap"] *= np.linspace(1, 2, 12)
    res = stability.bootstrap("fundamentals", 2, fundamentals=fund, n_boot=10)
    assert res["ari"].mean() > 0.9


def test_temporal_stability():
    close, volume = panel(n_days=500)
    trans, lab = stability.temporal("returns", 2, close, volume, window=200, step=100)
    assert len(trans) >= 2
    assert (trans["ari"] > 0.9).all()


def test_hrp_and_cluster_parity_weights():
    close, _ = panel()
    cov = np.expm1(features.log_returns(close)).cov()
    w = portfolio.hrp_weights(cov)
    assert w.sum() == pytest.approx(1) and (w > 0).all()
    # Calm group B should get more weight than volatile group A.
    assert w[w.index.str.contains(":B")].sum() > w[w.index.str.contains(":A")].sum()
    labels = views.cluster(views.build_view("returns", close), 2)
    cp = portfolio.cluster_parity_weights(cov, labels)
    assert cp.groupby(labels).sum().to_numpy() == pytest.approx([0.5, 0.5])


def test_walk_forward_and_performance():
    close, volume = panel(n_days=450)
    rets, turnover = portfolio.walk_forward(close, volume, "returns", 2, train=200, step=60)
    assert set(rets.columns) == {"equal_weight", "inverse_vol", "hrp", "cluster_parity", "cluster_reps"}
    assert len(rets) == len(close) - 1 - 200
    perf = portfolio.performance(rets, turnover)
    assert {"ann_return", "ann_vol", "sharpe", "max_drawdown", "avg_turnover"} <= set(perf.columns)
    assert perf.loc["cluster_reps", "ann_vol"] > 0


def test_cluster_exposure(tmp_path):
    close, _ = panel()
    labels = views.cluster(views.build_view("returns", close), 2)
    cov = np.expm1(features.log_returns(close)).cov()
    path = tmp_path / "h.csv"
    pd.DataFrame({"ticker": ["A0", "A1", "NGX:A2"], "weight": [1, 1, 2]}).to_csv(path, index=False)
    w = portfolio.load_holdings(path, list(close.columns))
    assert w["NGX:A2"] == pytest.approx(0.5)
    table, summary = portfolio.cluster_exposure(w, labels, cov)
    assert summary["clusters_held"] == 1
    assert summary["effective_clusters_risk"] == pytest.approx(1.0)


def test_pairs_find_cointegrated_pair():
    rng = np.random.default_rng(0)
    n = 500
    dates = pd.bdate_range("2024-01-01", periods=n)
    x = np.cumsum(rng.normal(0, 0.01, n))
    spread = np.zeros(n)
    for t in range(1, n):  # AR(1) with half-life ~7 days
        spread[t] = 0.9 * spread[t - 1] + rng.normal(0, 0.01)
    close = pd.DataFrame({
        "NGX:X": np.exp(x + 4), "NGX:Y": np.exp(x + spread + 4),
        "NGX:Z": np.exp(np.cumsum(rng.normal(0, 0.01, n)) + 4),
    }, index=dates)
    labels = pd.Series(0, index=close.columns)
    df = pairs.pair_candidates(close, labels, min_corr=0.0)
    top = df.iloc[0]
    assert {top["a"], top["b"]} == {"NGX:X", "NGX:Y"}
    assert top["candidate"] and 2 < top["half_life_days"] < 20


def test_bh_qvalues():
    q = bh_qvalues(pd.Series([0.01, 0.04, 0.03, 0.5]))
    assert (q >= pd.Series([0.01, 0.04, 0.03, 0.5])).all()
    assert q.max() <= 1


# ---------------------------------------------------------------------- FX
from african_clusters import fx


def test_to_usd_and_liquidity_uses_local_prices():
    dates = pd.bdate_range("2024-01-01", periods=5)
    local = pd.DataFrame({"NGX:A": [100.0] * 5, "JSE:B": [10.0, 11, 12, 13, 14]}, index=dates)
    rates = pd.DataFrame({"NGN": [1000.0, 1000, 2000, 2000, 2000, 2000, 2000], "ZAR": 20.0, "USD": 1.0},
                         index=pd.date_range("2024-01-01", periods=7))
    ccy = fx.stock_currencies(local.columns)
    usd = fx.to_usd(local, ccy, rates)
    assert usd["NGX:A"].tolist() == [0.1, 0.1, 0.05, 0.05, 0.05]  # devaluation halves USD price
    assert usd["JSE:B"].iloc[0] == pytest.approx(0.5)
    f_usd = features.return_features(usd)
    f_mix = features.return_features(usd, local_close=local)
    assert f_usd.loc["NGX:A", "zero_return_frac"] < 1   # FX makes a flat stock "move"
    assert f_mix.loc["NGX:A", "zero_return_frac"] == 1  # but it never traded at a new price
    assert f_mix.loc["NGX:A", "max_drawdown"] == pytest.approx(-0.5)


def test_stock_currencies_prefers_universe():
    uni = pd.DataFrame({"exchange": ["EGX", "EGX"], "ticker": ["X", "Y"], "currency": ["usd", "EGP"]})
    ccy = fx.stock_currencies(["EGX:X", "EGX:Y", "NSE:Z"], uni)
    assert ccy.tolist() == ["USD", "EGP", "KES"]


def test_usd_rates_caches_and_fills(tmp_path, monkeypatch):
    calls = []

    class Resp:
        ok, status_code = True, 200

        def __init__(self, url):
            self.url = url

        def json(self):
            return {"date": "x", "usd": {"ngn": 1500.0, "zar": 18.0}}

    def fake_get(self, url, timeout):
        calls.append(url)
        if "2024-01-02" in url:  # a missing day on both mirrors
            r = Resp(url); r.ok, r.status_code = False, 404
            return r
        return Resp(url)

    monkeypatch.setattr(fx.requests.Session, "get", fake_get)
    dates = pd.DatetimeIndex(["2024-01-01", "2024-01-03"])
    rates = fx.usd_rates(dates, ["NGN", "ZAR"], cache_dir=tmp_path, workers=1)
    assert len(rates) == 3 and rates["NGN"].notna().all() and (rates["USD"] == 1).all()
    n = len(calls)
    fx.usd_rates(dates, ["NGN", "ZAR"], cache_dir=tmp_path, workers=1)
    assert len(calls) - n == 2  # only the missing day is retried (both mirrors)
