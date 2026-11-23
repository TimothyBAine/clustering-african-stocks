"""Command-line entry point: ``acs <command>``.

    acs exchanges
    acs fetch --exchanges NGX JSE NSE --range 2Y --fundamentals
    acs cluster   --view returns --k 6            # one view, dendrogram + CSV
    acs compare   --k 6                           # what does "similar" mean?
    acs stability --view returns --k 6            # are the clusters real?
    acs portfolio --view returns --k 6            # backtest cluster-aware weights
    acs exposure  holdings.csv --view returns --k 6
    acs pairs     --view returns --k 6            # mean-reversion candidates
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import clustering, compare, config, data, features, fx, pairs, plots, portfolio, stability, views
from .mansa_client import MansaClient

P = config.PROCESSED_DIR
R = config.REPORTS_DIR
log = logging.getLogger("acs")


# ------------------------------------------------------------------ loading
@dataclass
class Inputs:
    close: pd.DataFrame | None     # prices in the analysis currency (USD or local)
    local: pd.DataFrame | None     # traded local-currency prices (for liquidity)
    volume: pd.DataFrame | None
    fund: pd.DataFrame | None
    currency: str


def _fx_rates(close: pd.DataFrame, universe: pd.DataFrame | None, refresh: bool = False) -> pd.DataFrame:
    ccys = sorted(set(fx.stock_currencies(close.columns, universe).dropna()) - {"USD"})
    rates = fx.usd_rates(close.index, ccys, refresh=refresh)
    rates.to_csv(P / "fx_usd.csv")
    return rates


def _load(min_obs: int, currency: str = "usd", need_history: bool = True) -> Inputs:
    close = local = volume = fund = None
    universe = pd.read_csv(P / "universe.csv") if (P / "universe.csv").exists() else None
    if (P / "close.csv").exists():
        # Fill holidays in local currency first, then convert at each day's rate.
        local = features.clean_panel(pd.read_csv(P / "close.csv", index_col=0, parse_dates=True),
                                     min_obs=min_obs)
        close = local
        if currency == "usd":
            rates_path = P / "fx_usd.csv"
            if rates_path.exists():
                rates = pd.read_csv(rates_path, index_col=0, parse_dates=True)
            else:
                log.info("No data/processed/fx_usd.csv yet; downloading FX history (cached).")
                rates = _fx_rates(local, universe)
            ccy = fx.stock_currencies(local.columns, universe)
            if not set(ccy.dropna()) <= set(rates.columns) or rates.index.max() < local.index.max():
                rates = _fx_rates(local, universe)  # new currencies or dates since last fetch
            close = fx.to_usd(local, ccy, rates)
        if (P / "volume.csv").exists():
            volume = pd.read_csv(P / "volume.csv", index_col=0, parse_dates=True)
    elif need_history:
        raise SystemExit("No price history at data/processed/close.csv. Run `acs fetch` first.")
    if (P / "fundamentals.csv").exists():
        fund = pd.read_csv(P / "fundamentals.csv")
    return Inputs(close, local, volume, fund, currency.upper())


def _reference_labels(keys, fund: pd.DataFrame | None) -> dict[str, pd.Series]:
    idx = pd.Index(keys)
    refs = {"exchange": pd.Series(idx.str.split(":").str[0], index=idx)}
    if fund is not None and "sector" in fund.columns:
        s = fund.assign(key=fund["exchange"] + ":" + fund["ticker"]).set_index("key")["sector"]
        s = s.reindex(idx).dropna()
        if s.nunique() > 1:
            refs["sector"] = s
    return refs


def _print(title: str, df: pd.DataFrame | pd.Series, digits: int = 3) -> None:
    print(f"\n{title}\n" + df.round(digits).to_string())


# ----------------------------------------------------------------- commands
def cmd_exchanges(args: argparse.Namespace) -> None:
    df = pd.DataFrame(MansaClient().exchanges(refresh=args.refresh))
    print(df.to_string(index=False))


def cmd_fetch(args: argparse.Namespace) -> None:
    client = MansaClient()
    P.mkdir(parents=True, exist_ok=True)
    if not (args.fundamentals_only or args.fx_only):
        uni = data.universe(client, args.exchanges, refresh=args.refresh)
        uni.to_csv(P / "universe.csv", index=False)
        print(f"Universe: {len(uni)} stocks across {uni['exchange'].nunique()} exchanges")
        if not args.snapshot_only:
            close, volume = data.price_panel(
                client, uni, range_=args.range, max_stocks=args.max_stocks, refresh=args.refresh
            )
            close.to_csv(P / "close.csv")
            volume.to_csv(P / "volume.csv")
            print(f"Prices: {close.shape[1]} series, {close.index.min()} → {close.index.max()}")
            args.fx = True
    if args.fx or args.fx_only:
        close = pd.read_csv(P / "close.csv", index_col=0, parse_dates=True)
        uni = pd.read_csv(P / "universe.csv") if (P / "universe.csv").exists() else None
        rates = _fx_rates(close, uni, refresh=args.refresh)
        print(f"FX: daily USD rates for {[c for c in rates.columns if c != 'USD']}, "
              f"{rates.index.min().date()} → {rates.index.max().date()}")
    if args.fundamentals or args.fundamentals_only:
        if (P / "close.csv").exists():  # only stocks we have prices for
            keys = list(pd.read_csv(P / "close.csv", index_col=0, nrows=1).columns)
        else:
            uni = pd.read_csv(P / "universe.csv")
            keys = list(uni["exchange"] + ":" + uni["ticker"])
        fund = data.fundamentals(client, keys[: args.max_stocks], refresh=args.refresh)
        fund.to_csv(P / "fundamentals.csv", index=False)
        usable = features.fundamental_features(fund).columns.tolist() if not fund.empty else []
        print(f"Fundamentals: {len(fund)} stocks; usable numeric fields: {usable or 'none'}")


def cmd_cluster(args: argparse.Namespace) -> None:
    R.mkdir(parents=True, exist_ok=True)
    if args.view == "snapshot":
        feats = features.snapshot_features(pd.read_csv(P / "universe.csv")).dropna(axis=1, how="all")
        if args.choose_k:
            _print("k selection (KMeans)", clustering.choose_k(feats))
            return
        labels = clustering.kmeans(feats, args.k)
        plots.pca_plot(feats, labels, R / "pca_snapshot.png", "snapshot features")
    else:
        d = _load(args.min_obs, args.currency, need_history=args.view != "fundamentals")
        view = views.build_view(args.view, d.close, d.volume, d.fund, d.local)
        if args.choose_k:
            _print(f"k selection for view '{args.view}'",
                   clustering.choose_k_distance(view.distance, view.method))
            print("Tip: also run `acs stability --k-range ...` — pick k that is stable, not just separated.")
            return
        labels = views.cluster(view, args.k)
        plots.dendrogram_plot(views.linkage_matrix(view), view.distance.index,
                              f"Hierarchical clustering — {args.view} view ({d.currency})", R / f"dendrogram_{args.view}.png")
        feats = view.features
        if feats is not None and feats.shape[1] >= 2:
            plots.pca_plot(feats, labels, R / f"pca_{args.view}.png", f"{args.view} view")

    out = labels.to_frame()
    out["exchange"] = out.index.str.split(":").str[0]
    if feats is not None:
        out = out.join(feats)
        summary = feats.groupby(labels).median()
        summary.insert(0, "n", labels.value_counts().sort_index())
        _print("Cluster medians:", summary)
    out.sort_values("cluster").to_csv(R / f"clusters_{args.view}.csv")
    print("\nExchange mix per cluster:")
    print(pd.crosstab(out["cluster"], out["exchange"]).to_string())
    print(f"\nSaved results to {R}")


def cmd_compare(args: argparse.Namespace) -> None:
    R.mkdir(parents=True, exist_ok=True)
    d = _load(args.min_obs, args.currency)
    close, volume, fund, local = d.close, d.volume, d.fund, d.local
    names = [v for v in args.views if v != "fundamentals" or fund is not None]
    if "fundamentals" in args.views and fund is None:
        print("(skipping fundamentals view: run `acs fetch --fundamentals --fundamentals-only`)")

    # 1–2. Full-sample agreement between views and with exchange/sector.
    built = {}
    for n in names:
        try:
            built[n] = views.build_view(n, close, volume, fund, local)
        except ValueError as e:
            print(f"(skipping {n}: {e})")
    # The same co-movement and risk views in local currency show how much is FX.
    if d.currency == "USD" and not args.no_local:
        for n in ("returns", "risk"):
            if n in built:
                built[f"{n}_local"] = views.build_view(n, local, volume, fund, local)
    common = sorted(set.intersection(*(set(v.distance.index) for v in built.values())))
    built = {n: v.subset(common) for n, v in built.items()}
    partitions = {n: views.cluster(v, args.k) for n, v in built.items()}
    refs = _reference_labels(common, fund)
    ari = compare.partition_agreement({**partitions, **refs}, "ari")
    mantel = compare.distance_agreement({n: v.distance for n, v in built.items()}, n_perm=args.n_perm)
    print(f"Comparing {len(built)} views on {len(common)} common stocks, k={args.k}, prices in {d.currency}"
          + (" (*_local = same view in local currency)" if any(n.endswith("_local") for n in built) else ""))
    _print("Partition agreement (ARI; 0 = unrelated, 1 = identical):", ari)
    _print("Distance agreement (Mantel Spearman r above diagonal, p-value below):", mantel)
    ari.to_csv(R / "compare_ari.csv")
    mantel.to_csv(R / "compare_mantel.csv")
    plots.heatmap(ari, R / "compare_ari.png", f"Partition agreement (ARI), k={args.k}", vmin=0, vmax=1)

    # 3. Out-of-sample: what does each notion of similarity predict?
    # Outcomes are always scored in the analysis currency (what the investor experiences).
    train, test = compare.split_by_time(close, args.train_frac)
    train = features.clean_panel(train, min_obs=min(args.min_obs, len(train) // 2))
    test = features.clean_panel(test, min_obs=min(args.min_obs, len(test) // 2))
    keys = sorted(set(common) & set(train.columns) & set(test.columns))
    train_local = local.loc[train.index, keys]
    oos_parts = {}
    for n in built:
        base = n.removesuffix("_local")
        src = train_local if n.endswith("_local") else train[keys]
        v = views.build_view(base, src, volume, fund, train_local).subset(keys)
        oos_parts[n] = views.cluster(v, args.k)
    oos_parts.update(_reference_labels(keys, fund))
    scores = compare.evaluate_partitions(oos_parts, test[keys], n_perm=args.n_perm,
                                         test_local=local.loc[test.index, keys])
    print(f"\nOut-of-sample test ({d.currency} outcomes): clusters built on {train.index[0].date()}–{train.index[-1].date()}, "
          f"scored on {test.index[0].date()}–{test.index[-1].date()} ({len(keys)} stocks)")
    zcols = [f"{c}_z" for c in compare.SCORE_COLS]
    _print("Raw scores (comovement_gap = within − between cluster correlation; *_eta2 = share "
           "of test-period variance explained by cluster):", scores[["n_clusters", "within_corr", "between_corr"] + compare.SCORE_COLS])
    _print("z-scores against random partitions of the same sizes (>2 means a real effect):", scores[zcols], 2)
    scores.to_csv(R / "compare_oos.csv")
    plots.heatmap(scores[zcols].rename(columns=lambda c: c.replace("_z", "")),
                  R / "compare_oos_z.png", "What each partition predicts out of sample (z vs random)",
                  fmt="{:.1f}", cmap="RdBu_r",
                  vmin=-max(3, np.nanmax(np.abs(scores[zcols].values))),
                  vmax=max(3, np.nanmax(np.abs(scores[zcols].values))))
    best = scores[zcols].idxmax()
    print("\nBest partition for each outcome: " +
          ", ".join(f"{c.replace('_z', '')} → {p}" for c, p in best.items()))
    print(f"Saved results to {R}")


def cmd_stability(args: argparse.Namespace) -> None:
    R.mkdir(parents=True, exist_ok=True)
    d = _load(args.min_obs, args.currency, need_history=args.view != "fundamentals")
    close, volume, fund, local = d.close, d.volume, d.fund, d.local
    if args.k_range:
        by_k = stability.stability_by_k(args.view, args.k_range, close, volume, fund,
                                        n_boot=max(10, args.n_boot // 3), block=args.block,
                                        local_close=local)
        _print(f"Stability by k ({args.view} view): higher ARI = more reproducible", by_k)
        by_k.to_csv(R / f"stability_by_k_{args.view}.csv")

    res = stability.bootstrap(args.view, args.k, close, volume, fund,
                              n_boot=args.n_boot, block=args.block, local_close=local)
    ari = res["ari"]
    print(f"\nBootstrap ({args.n_boot} resamples, {args.view} view, k={args.k}): "
          f"ARI vs full sample mean {ari.mean():.2f}, 10th pct {ari.quantile(0.1):.2f}")
    _print("Per-cluster recovery (Jaccard: >0.75 stable, 0.6–0.75 pattern, <0.5 dissolves):", res["clusters"])
    shaky = res["stocks"][res["stocks"]["confidence"] < 0.5]
    print(f"\n{len(shaky)} of {len(res['stocks'])} stocks stay with their cluster-mates in <50% of resamples"
          + (" (see stability_stocks CSV)." if len(shaky) else "."))
    res["clusters"].to_csv(R / f"stability_clusters_{args.view}.csv")
    res["stocks"].to_csv(R / f"stability_stocks_{args.view}.csv")
    base_view = views.build_view(args.view, close, volume, fund, local)
    plots.coassignment_plot(res["coassign"], views.linkage_matrix(base_view),
                            R / f"stability_coassign_{args.view}.png",
                            f"Bootstrap co-assignment — {args.view} view, k={args.k}")

    if args.view in views.HISTORY_VIEWS:
        trans, lab = stability.temporal(args.view, args.k, close, volume,
                                        window=args.window, step=args.step, min_obs=args.min_obs,
                                        local_close=local)
        if len(trans):
            _print(f"Temporal stability ({args.window}-day windows every {args.step} days):",
                   trans.set_index(["from", "to"]))
            trans.to_csv(R / f"stability_temporal_{args.view}.csv", index=False)
            lab.to_csv(R / f"stability_temporal_labels_{args.view}.csv")
        else:
            print("\n(Not enough history for temporal windows; lower --window.)")
    print(f"Saved results to {R}")


def cmd_portfolio(args: argparse.Namespace) -> None:
    R.mkdir(parents=True, exist_ok=True)
    d = _load(args.min_obs, args.currency)
    close, volume, fund, local = d.close, d.volume, d.fund, d.local
    rets, turnover = portfolio.walk_forward(
        close, volume, args.view, args.k, fund, train=args.train, step=args.step,
        cost_bps=args.cost_bps, max_zero_frac=args.max_zero_frac, progress=log.info,
        local_close=local,
    )
    perf = portfolio.performance(rets, turnover)
    print(f"\nWalk-forward backtest, {rets.index[0].date()}–{rets.index[-1].date()} "
          f"({len(rets)} out-of-sample days), view={args.view}, k={args.k}, costs={args.cost_bps:g} bps, "
          f"returns in {d.currency}")
    _print("Performance (compare strategies, not absolute levels):", perf)
    perf.to_csv(R / "portfolio_performance.csv")
    rets.to_csv(R / "portfolio_returns.csv")
    plots.equity_plot(rets, R / "portfolio_equity.png",
                      f"Walk-forward — clusters from {args.view} view ({d.currency})", d.currency)

    # What to hold now, estimated on the most recent window.
    w, labels = portfolio.target_weights(close.iloc[-args.train - 1:], volume, args.view, args.k,
                                         fund, args.max_zero_frac, local)
    w.insert(0, "cluster", labels.reindex(w.index))
    w.sort_values(["cluster", "hrp"], ascending=[True, False]).to_csv(R / "portfolio_weights_now.csv")
    reps = w.index[w["cluster_reps"] > 0]
    by_cluster = w.groupby("cluster")[["hrp", "cluster_parity", "equal_weight"]].sum()
    by_cluster.insert(0, "n", w["cluster"].value_counts().sort_index())
    by_cluster["representative"] = [next((r for r in reps if w.at[r, "cluster"] == c), "") for c in by_cluster.index]
    _print("Current allocation by cluster:", by_cluster)
    if len(rets) < 250:
        print("\nNote: under a year of out-of-sample data; differences may be noise. Fetch a longer --range.")
    if d.currency != "USD":
        print("Note: --currency local mixes currencies, so cross-exchange portfolio returns are not meaningful.")
    print(f"Saved results to {R}")


def cmd_exposure(args: argparse.Namespace) -> None:
    d = _load(args.min_obs, args.currency)
    close, volume, fund, local = d.close, d.volume, d.fund, d.local
    win = close.iloc[-args.train - 1:]
    win_local = local.loc[win.index]
    view = views.build_view(args.view, win, volume, fund, win_local)
    labels = views.cluster(view, args.k)
    weights = portfolio.load_holdings(args.holdings, list(close.columns))
    cov = np.expm1(features.log_returns(win[labels.index])).fillna(0.0).cov()
    table, summary = portfolio.cluster_exposure(weights, labels, cov)
    liq = features.return_features(win[labels.index], local_close=win_local)[["zero_return_frac"]]
    table["representative"] = portfolio.representatives(labels, view.distance, liq)
    _print(f"Holdings by cluster ({args.view} view, k={args.k}):", table)
    print(f"\nPortfolio vol (ann., {d.currency}): {summary['port_vol_ann']:.1%}")
    print(f"Clusters held: {summary['clusters_held']} of {summary['clusters_total']}; "
          f"effective clusters by capital {summary['effective_clusters_capital']:.1f}, "
          f"by risk {summary['effective_clusters_risk']:.1f}")
    if summary["unmatched_holdings"]:
        print(f"Not in the clustered universe (ignored): {summary['unmatched_holdings']}")
    over = table[table["risk_share"] > 1.5 * table["weight"]]
    for c, row in over.iterrows():
        print(f"- Cluster {c}: {row['weight']:.0%} of capital but {row['risk_share']:.0%} of risk.")
    empty = table[table["n_holdings"] == 0]
    if len(empty):
        print("Unheld clusters (diversification candidates): " +
              ", ".join(f"{c} (e.g. {r})" for c, r in empty["representative"].items()))
    R.mkdir(parents=True, exist_ok=True)
    table.to_csv(R / "exposure.csv")


def cmd_pairs(args: argparse.Namespace) -> None:
    R.mkdir(parents=True, exist_ok=True)
    d = _load(args.min_obs, args.currency)
    labels = views.cluster(views.build_view(args.view, d.close, d.volume, d.fund, d.local), args.k)
    df = pairs.pair_candidates(d.close, labels, train_frac=args.train_frac, min_corr=args.min_corr,
                               max_zero_frac=args.max_zero_frac, local_close=d.local)
    if df.empty:
        print("No pairs passed the liquidity/correlation filters.")
        return
    df.to_csv(R / "pairs.csv", index=False)
    cand = df[df["candidate"]]
    print(f"Tested {len(df)} within-cluster pairs; {int((df['q_train'] < 0.1).sum())} cointegrated "
          f"in-sample (q<0.10), {len(cand)} also hold out of sample with half-life 2–60 days.")
    if len(cand):
        _print("Candidates:", cand[["cluster", "a", "b", "hedge_ratio", "q_train", "p_test",
                                    "half_life_days", "z_now", "signal"]].head(args.top))
    print("\nScreen only: check borrow/short availability and liquidity (shorting is restricted on "
          "most African exchanges, so 'short' legs may be impossible to trade).")
    print(f"Saved results to {R / 'pairs.csv'}")


# --------------------------------------------------------------------- main
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(prog="acs", description="Cluster African stocks (Mansa data)")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_currency(s):
        s.add_argument("--currency", choices=["usd", "local"], default="usd",
                       help="Convert prices to USD (default) or keep local currency")

    def add_common(s, view_default="returns", k=True):
        add_currency(s)
        s.add_argument("--view", choices=views.VIEW_NAMES, default=view_default)
        if k:
            s.add_argument("--k", type=int, default=6)
        s.add_argument("--min-obs", type=int, default=120, help="Min trading days per stock")

    s = sub.add_parser("exchanges", help="List exchanges covered by Mansa")
    s.add_argument("--refresh", action="store_true")
    s.set_defaults(func=cmd_exchanges)

    s = sub.add_parser("fetch", help="Download universe, price history and fundamentals")
    s.add_argument("--exchanges", nargs="+", default=["NGX", "JSE", "NSE", "GSE", "EGX"])
    s.add_argument("--range", default="2Y", help="1M..ALL")
    s.add_argument("--max-stocks", type=int, default=None, help="Cap per-stock calls (rate limits)")
    s.add_argument("--snapshot-only", action="store_true", help="Skip history (free tier)")
    s.add_argument("--fundamentals", action="store_true", help="Also fetch /fundamentals per stock")
    s.add_argument("--fundamentals-only", action="store_true", help="Only fetch fundamentals")
    s.add_argument("--fx", action="store_true", help="(Re)build daily USD FX rates (automatic after prices)")
    s.add_argument("--fx-only", action="store_true", help="Only fetch FX rates for the existing price panel")
    s.add_argument("--refresh", action="store_true", help="Ignore cache")
    s.set_defaults(func=cmd_fetch)

    s = sub.add_parser("cluster", help="Cluster stocks under one similarity view")
    s.add_argument("--view", choices=views.VIEW_NAMES + ("snapshot",), default="returns")
    s.add_argument("--k", type=int, default=6)
    s.add_argument("--choose-k", action="store_true", help="Print silhouette scores and exit")
    s.add_argument("--min-obs", type=int, default=120)
    add_currency(s)
    s.set_defaults(func=cmd_cluster)

    s = sub.add_parser("compare", help="Compare similarity views and test what each predicts")
    s.add_argument("--views", nargs="+", choices=views.VIEW_NAMES,
                   default=["returns", "risk", "liquidity", "fundamentals"])
    s.add_argument("--k", type=int, default=6)
    s.add_argument("--train-frac", type=float, default=0.5)
    s.add_argument("--n-perm", type=int, default=200)
    s.add_argument("--min-obs", type=int, default=120)
    s.add_argument("--no-local", action="store_true", help="Skip the local-currency comparison views")
    add_currency(s)
    s.set_defaults(func=cmd_compare)

    s = sub.add_parser("stability", help="Bootstrap and rolling-window cluster stability")
    add_common(s)
    s.add_argument("--n-boot", type=int, default=100)
    s.add_argument("--block", type=int, default=20, help="Bootstrap block length in days")
    s.add_argument("--k-range", type=int, nargs="+", help="Also compare stability across these k")
    s.add_argument("--window", type=int, default=250)
    s.add_argument("--step", type=int, default=63)
    s.set_defaults(func=cmd_stability)

    s = sub.add_parser("portfolio", help="Walk-forward backtest of cluster-aware portfolios")
    add_common(s)
    s.add_argument("--train", type=int, default=252, help="Estimation window (days)")
    s.add_argument("--step", type=int, default=63, help="Rebalance every N days")
    s.add_argument("--cost-bps", type=float, default=50.0, help="One-way cost per unit turnover")
    s.add_argument("--max-zero-frac", type=float, default=0.5, help="Liquidity filter")
    s.set_defaults(func=cmd_portfolio)

    s = sub.add_parser("exposure", help="Cluster concentration of an existing portfolio")
    s.add_argument("holdings", help="CSV with columns ticker, weight")
    add_common(s)
    s.add_argument("--train", type=int, default=252)
    s.set_defaults(func=cmd_exposure)

    s = sub.add_parser("pairs", help="Within-cluster cointegrated pair candidates")
    add_common(s)
    s.add_argument("--train-frac", type=float, default=0.7)
    s.add_argument("--min-corr", type=float, default=0.3)
    s.add_argument("--max-zero-frac", type=float, default=0.3)
    s.add_argument("--top", type=int, default=20)
    s.set_defaults(func=cmd_pairs)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
