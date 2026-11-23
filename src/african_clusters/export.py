"""Bundle one consistent run of every analysis into JSON files for the dashboard.

``acs export`` writes ``reports/dashboard/*.json``. Each file is
``{"rows": [...], "meta": [...]}``: the rows a chart draws, plus labels saying
how the run was made (command, sample, currency, dates).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import compare, features, pairs, portfolio, stability, views

log = logging.getLogger(__name__)


def _mds_2d(distance: pd.DataFrame) -> pd.DataFrame:
    """Classical MDS: 2-D coordinates whose distances approximate ``distance``."""
    d2 = distance.to_numpy() ** 2
    n = len(d2)
    j = np.eye(n) - 1.0 / n
    b = -0.5 * j @ d2 @ j
    vals, vecs = np.linalg.eigh(b)
    top = np.argsort(vals)[::-1][:2]
    xy = vecs[:, top] * np.sqrt(np.clip(vals[top], 0, None))
    return pd.DataFrame(xy, index=distance.index, columns=["x", "y"])


def _records(df: pd.DataFrame) -> list[dict]:
    """JSON-safe rows: NaN/inf → null, numpy scalars → Python, dates → ISO strings."""
    out = []
    for rec in df.to_dict(orient="records"):
        clean = {}
        for k, v in rec.items():
            if isinstance(v, (pd.Timestamp, datetime)):
                v = v.strftime("%Y-%m-%d")
            elif hasattr(v, "isoformat") and not isinstance(v, str):
                v = v.isoformat()
            elif isinstance(v, (np.floating, float)):
                v = None if not np.isfinite(v) else round(float(v), 6)
            elif isinstance(v, np.integer):
                v = int(v)
            elif isinstance(v, np.bool_):
                v = bool(v)
            clean[str(k)] = v
        out.append(clean)
    return out


def run(
    close: pd.DataFrame,
    local: pd.DataFrame,
    volume: pd.DataFrame | None,
    fund: pd.DataFrame | None,
    currency: str,
    out_dir: Path,
    view_name: str = "returns",
    k: int = 6,
    n_boot: int = 50,
    k_range=(3, 4, 5, 6, 8),
    compare_views=("returns", "risk", "liquidity"),
    train: int = 252,
    step: int = 63,
    cost_bps: float = 50.0,
    command: str = "acs export",
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sample = (f"{close.shape[1]} stocks on {', '.join(sorted(close.columns.str.split(':').str[0].unique()))}, "
              f"{close.index.min().date()} to {close.index.max().date()}")
    base_meta = [
        {"label": "Command", "value": command, "type": "code", "lang": "bash"},
        {"label": "Sample", "value": sample},
        {"label": "Currency", "value": currency},
        {"label": "Generated", "value": generated},
    ]
    written: list[Path] = []

    def write(name: str, df: pd.DataFrame, extra: list[dict] | None = None) -> None:
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps({"rows": _records(df), "meta": base_meta + (extra or [])},
                                   allow_nan=False))
        written.append(path)
        log.info("wrote %s (%d rows)", path.name, len(df))

    # --- run summary ---------------------------------------------------------
    write("run", pd.DataFrame([{
        "n_stocks": close.shape[1], "exchanges": ", ".join(sorted(close.columns.str.split(":").str[0].unique())),
        "start": close.index.min(), "end": close.index.max(), "currency": currency,
        "view": view_name, "k": k, "n_boot": n_boot, "generated": generated,
    }]))

    # --- clusters + map + stability -----------------------------------------
    log.info("clustering + bootstrap stability (%d resamples)", n_boot)
    view = views.build_view(view_name, close, volume, fund, local)
    boot = stability.bootstrap(view_name, k, close, volume, fund, n_boot=n_boot, local_close=local)
    labels = boot["labels"]
    feats = features.return_features(close, volume, local)
    reps = portfolio.representatives(labels, view.distance, feats[["zero_return_frac"]])
    xy = _mds_2d(view.distance)
    members = pd.DataFrame({
        "key": labels.index, "ticker": labels.index.str.split(":").str[1],
        "exchange": labels.index.str.split(":").str[0], "cluster": labels.values,
    }).set_index("key")
    members = members.join(boot["stocks"][["confidence"]]).join(xy).join(
        feats[["ann_return", "ann_vol", "beta", "max_drawdown", "zero_return_frac"]])
    members["representative"] = members.index.isin(reps.values)
    members["cluster_name"] = "C" + members["cluster"].astype(str)
    write("clusters", members.reset_index(),
          [{"label": "View", "value": f"{view_name}, k={k}, Ward linkage"},
           {"label": "Map", "value": "Classical MDS of the view's distance matrix"}])

    cl = boot["clusters"].copy()
    cl["verdict"] = cl["verdict"].astype(str)
    cl["representative"] = reps.reindex(cl.index)
    cl["exchange_mix"] = [
        ", ".join(f"{ex} {n}" for ex, n in members[members["cluster"] == c]["exchange"].value_counts().items())
        for c in cl.index
    ]
    cl.index.name = "cluster"
    cl = cl.reset_index()
    cl["cluster_name"] = "C" + cl["cluster"].astype(str)
    write("cluster_stability", cl, [{"label": "Resamples", "value": f"{n_boot} block bootstraps (20-day blocks)"}])

    log.info("stability by k")
    by_k = stability.stability_by_k(view_name, list(k_range), close, volume, fund,
                                    n_boot=max(10, n_boot // 3), local_close=local)
    write("stability_k", by_k.reset_index())

    if view_name in views.HISTORY_VIEWS:
        trans, _ = stability.temporal(view_name, k, close, volume, local_close=local)
        write("temporal", trans.rename(columns={"from": "window_from", "to": "window_to"}),
              [{"label": "Windows", "value": "250 trading days, stepped every 63"}])

    # --- what similarity means ----------------------------------------------
    log.info("comparing views")
    built = {n: views.build_view(n, close, volume, fund, local) for n in compare_views}
    if fund is not None:
        try:
            built["fundamentals"] = views.build_view("fundamentals", close, volume, fund, local)
        except ValueError as e:
            log.info("skipping fundamentals: %s", e)
    if currency == "USD":
        for n in ("returns", "risk"):
            if n in built:
                built[f"{n}_local"] = views.build_view(n, local, volume, fund, local)
    common = sorted(set.intersection(*(set(v.distance.index) for v in built.values())))
    parts = {n: views.cluster(v.subset(common), k) for n, v in built.items()}
    refs = {"exchange": pd.Series(pd.Index(common).str.split(":").str[0], index=common)}
    if fund is not None and "sector" in fund.columns:
        s = fund.assign(key=fund["exchange"] + ":" + fund["ticker"]).set_index("key")["sector"]
        s = s.reindex(common).dropna()
        if s.nunique() > 1:
            refs["sector"] = s
    ari = compare.partition_agreement({**parts, **refs})
    ari_long = ari.stack().rename("ari").rename_axis(["a", "b"]).reset_index()
    write("agreement", ari_long)

    train_c, test_c = compare.split_by_time(close)
    train_c = features.clean_panel(train_c, min_obs=min(120, len(train_c) // 2))
    test_c = features.clean_panel(test_c, min_obs=min(120, len(test_c) // 2))
    keys = sorted(set(common) & set(train_c.columns) & set(test_c.columns))
    train_l = local.loc[train_c.index, keys]
    oos_parts = {}
    for n in built:
        base = n.removesuffix("_local")
        src = train_l if n.endswith("_local") else train_c[keys]
        oos_parts[n] = views.cluster(views.build_view(base, src, volume, fund, train_l).subset(keys), k)
    oos_parts.update({r: s.reindex(keys).dropna() for r, s in refs.items()})
    scores = compare.evaluate_partitions(oos_parts, test_c[keys], n_perm=200,
                                         test_local=local.loc[test_c.index, keys])
    rows = []
    for part, r in scores.iterrows():
        kind = "reference" if part in refs else ("local currency" if part.endswith("_local") else "view")
        for outcome in compare.SCORE_COLS:
            rows.append({"partition": part, "kind": kind, "outcome": outcome.replace("_eta2", "").replace("_gap", ""),
                         "z": r[f"{outcome}_z"], "raw": r[outcome], "n_clusters": r["n_clusters"]})
    write("oos_scores", pd.DataFrame(rows), [
        {"label": "Train window", "value": f"{train_c.index[0].date()} to {train_c.index[-1].date()}"},
        {"label": "Test window", "value": f"{test_c.index[0].date()} to {test_c.index[-1].date()}"},
        {"label": "Null", "value": "200 random partitions with the same cluster sizes"},
    ])

    # --- portfolio -------------------------------------------------------------
    log.info("walk-forward backtest")
    rets, turnover = portfolio.walk_forward(close, volume, view_name, k, fund, train=train, step=step,
                                            cost_bps=cost_bps, local_close=local)
    perf = portfolio.performance(rets, turnover)
    perf.index.name = "strategy"
    write("strategy_perf", perf.reset_index(), [
        {"label": "Out-of-sample", "value": f"{rets.index[0].date()} to {rets.index[-1].date()} ({len(rets)} days)"},
        {"label": "Rules", "value": f"{train}-day estimation, rebalance every {step} days, {cost_bps:g} bps per unit turnover"},
    ])
    eq = (1 + rets).cumprod()
    eq_long = eq.stack().rename("value").rename_axis(["date", "strategy"]).reset_index()
    write("equity", eq_long)

    w, wl = portfolio.target_weights(close.iloc[-train - 1:], volume, view_name, k, fund, local_close=local)
    w.insert(0, "cluster", wl.reindex(w.index))
    alloc = w.groupby("cluster")[["hrp", "cluster_parity", "equal_weight", "inverse_vol"]].sum()
    alloc.insert(0, "n", w["cluster"].value_counts().sort_index())
    rep_now = w.index[w["cluster_reps"] > 0]
    alloc["representative"] = [next((r for r in rep_now if w.at[r, "cluster"] == c), None) for c in alloc.index]
    alloc = alloc.reset_index()
    alloc["cluster_name"] = "W" + alloc["cluster"].astype(str)
    write("allocation", alloc, [{"label": "Note", "value": "Clusters re-estimated on the latest window; "
                                 "numbering can differ from the full-sample clusters"}])

    # --- pairs ---------------------------------------------------------------
    log.info("pair screen")
    pr = pairs.pair_candidates(close, labels, local_close=local)
    if not pr.empty:
        pr = pr.sort_values(["candidate", "q_train"], ascending=[False, True]).head(30)
    write("pairs", pr)
    return written
