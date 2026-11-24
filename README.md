# Clustering African Stocks

Group equities across African exchanges (NGX, JSE, NSE, GSE, EGX, BRVM, …) by
how they actually behave, using market data from the
[Mansa API](https://mansamarkets.com/developers).

**[View the interactive dashboard →](https://timothybaine.github.io/clustering-african-stocks/)**
852 stocks on five exchanges, two years of daily data. The page is built from `acs export` output
(`docs/data/`, refreshed by copying `reports/dashboard/*.json` there).

## Questions this project explores

- Do stocks cluster by **exchange/country** (currency and macro factors), or by **sector** across borders?
- How much of the structure comes from **liquidity**? Many African stocks trade rarely, so stale prices make their returns look less correlated than they really are.
- Which stocks look like each other for the purposes of **diversifying** a portfolio?

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env          # then paste your MANSA_API_KEY
```

On macOS with Python 3.13+, if `acs` can't import `african_clusters` after
installing, the editable-install `.pth` file has been marked hidden and Python
skips hidden `.pth` files. Fix it with
`chflags nohidden .venv/lib/python3.*/site-packages/__editable__*.pth`.

### Mansa API tiers

| Tier | Requests/day | What works here |
|---|---|---|
| Standard (free) | 100 | Exchange and stock lists → `acs cluster --view snapshot` |
| Professional ($50/mo) | 10,000 | Daily `/history` and `/fundamentals` → everything else |

Every response is cached in `data/raw/`, so running a command again costs no
requests. Pass `--refresh` to re-download.

## Workflow

```bash
acs exchanges                                          # see the exchange codes
acs fetch --exchanges NGX JSE NSE GSE EGX --range 2Y --fundamentals

# 1. Cluster under one notion of similarity
acs cluster --view returns --choose-k                  # silhouette by k
acs cluster --view returns --k 6                       # dendrogram + CSV

# 2. Test what "similar" means
acs compare --k 6

# 3. Check the clusters are real
acs stability --view returns --k 6 --k-range 3 4 6 8

# 4. Turn clusters into decisions
acs portfolio --view returns --k 6                     # walk-forward backtest + current weights
acs exposure my_holdings.csv --view returns --k 6      # cluster concentration of a portfolio
acs pairs --view returns --k 6                         # mean-reversion pair candidates

# 5. Everything above as dashboard data
acs export --view returns --k 6                        # reports/dashboard/*.json
```

Every command writes CSVs and PNGs to `reports/`. Analysis commands take
`--currency usd` (the default) or `--currency local`.

## Currency

Returns are measured in **USD by default**. Adding naira returns to rand returns
produces a portfolio return nobody earns, and after a devaluation a stock that's
calm in local terms can lose heavily in dollars.

- **FX source.** Mansa's `/markets/forex` only has today's spot rates, so daily history comes
  from the free, keyless [fawazahmed0/currency-api](https://github.com/fawazahmed0/exchange-api)
  (one cached JSON per day in `data/raw/fx/`, saved to `data/processed/fx_usd.csv`).
  It is a community dataset, not an official central-bank fixing.
- **Order of operations.** Holidays are forward-filled in local currency, then
  each day's price is divided by that day's rate. Each stock's currency comes from
  the stock list (one EGX stock is quoted in USD), falling back to the exchange's currency.
- **Liquidity stays local.** Zero-return days, lag-1 autocorrelation and the
  tradability filters use *local* prices. Once converted, a stock that never
  trades still "moves" with the currency, which would hide how illiquid it is.
- **`acs compare` runs both.** In USD mode it adds `returns_local` and
  `risk_local` views, so you can see how much of the structure is currency
  and how much is the equity market itself.

## Similarity views

What counts as "similar" is a modelling choice, so the project makes it explicit.
Each view produces a distance matrix, and all of them are clustered the same way
(Ward hierarchical linkage), which makes the results comparable.

| View | Two stocks are close if they… | Built from |
|---|---|---|
| `returns` | move together day to day | `d = sqrt(2(1 − ρ))` on daily log returns |
| `risk` | have a similar risk profile | ann. vol, downside vol, beta, max drawdown, skew, kurtosis |
| `liquidity` | trade similarly | share of zero-return days, lag-1 autocorrelation, log volume |
| `fundamentals` | look alike on paper | `/fundamentals` fields with ≥50% coverage (currently mostly log market cap; P/E is not yet in the API) |
| `profile` | all history features together | everything above except fundamentals |
| `snapshot` | (free tier) look alike in today's quote | price, % change, volume from the stock list |

Feature views are median-imputed, robust-scaled and capped at ±4 so a handful
of jumpy, illiquid stocks can't claim their own clusters. Ward is valid for
the returns view too, because `sqrt(2(1 − ρ))` is the Euclidean distance between
standardised return series. Average linkage chained badly on illiquid stocks.

## `acs compare`: what does similarity mean?

1. **Agreement between views.** ARI between the partitions, plus a Mantel test
   between the raw distance matrices (needs no choice of k).
2. **Exchange or sector in disguise?** Each partition is compared with the
   `exchange` and (when fundamentals are fetched) `sector` labels.
3. **What does each view predict?** Views are built on the first half of the
   history and scored on the second half:
   - `comovement_gap`: average correlation within clusters minus between clusters
   - `risk_eta2`, `drawdown_eta2`, `liquidity_eta2`: share of the test period's
     cross-sectional variance in vol, drawdown and zero-return frequency that
     the clusters explain

   Each score is reported as a z-score against random partitions with the same
   cluster sizes, so `exchange` (2 groups) and a 6-cluster view can be compared fairly.

## `acs stability`: are the clusters real?

- **Bootstrap**: history views resample days in blocks (default 20 days,
  which keeps autocorrelation and stale-price effects). The fundamentals view
  resamples stocks. Reported:
  ARI against the full-sample clustering, per-cluster **Jaccard recovery**
  (>0.75 stable, 0.6–0.75 a real pattern whose exact members are uncertain,
  <0.5 the cluster falls apart), each stock's confidence, and a co-assignment heatmap.
- `--k-range` repeats the bootstrap for several k. Prefer a k whose clusters are
  reproducible over one that only scores a slightly higher silhouette.
- **Temporal**: rolling windows (default 250 days, stepped every 63) are
  clustered separately, and ARI between consecutive windows shows how much
  membership changes over time.

## `acs portfolio`, `exposure`, `pairs`: from clusters to decisions

**Portfolio.** A walk-forward backtest that re-estimates on a trailing window
(default 252 days), rebalances every 63 days, and charges 50 bps per unit of
turnover. Only stocks with ≤50% zero-return days are eligible. Strategies:

| Strategy | Idea |
|---|---|
| `equal_weight`, `inverse_vol` | baselines |
| `hrp` | Hierarchical Risk Parity (López de Prado) |
| `cluster_parity` | equal risk budget per cluster, inverse-vol inside |
| `cluster_reps` | one liquid representative (medoid) per cluster: a compact portfolio for thin markets |

It also writes the current target weights (`portfolio_weights_now.csv`).

**Exposure.** Takes a CSV with `ticker,weight` (bare tickers or `EXCHANGE:TICKER`) and
reports capital vs **risk share** per cluster, the effective number of clusters,
clusters that carry far more risk than capital, and representatives of clusters
you don't hold.

**Pairs.** Engle–Granger cointegration tests run only within clusters (fewer
tests, fewer false discoveries), with Benjamini–Hochberg q-values. The hedge ratio
and half-life come from the train period, then an **out-of-sample ADF check**
on the test spread and the current z-score. The output is a screen to research
further, not a set of trading signals.

## Caveats

- In USD, the **country/currency factor dominates co-movement**: `exchange`
  alone beats every clustering at predicting next year's correlations. The
  2024–26 sample had no major naira devaluation (NGN strengthened), so results
  over a devaluation period may differ sharply.
- Exchanges have different **trading calendars**. Short gaps are forward-filled, up to 5 days.
- Thin trading lowers measured correlations and makes flat prices look
  cointegrated. Watch `zero_return_frac` and the liquidity filters.
- The JSE list includes **ETFs**, which can end up as cluster representatives or pair legs.
- Shorting is restricted on most African exchanges. Pair "short" legs may be untradable.
- The backtest resets to target weights daily and ignores the bid-ask spread
  beyond the flat cost. With about 2 years of data the out-of-sample period is
  short, so use it to compare strategies, not to forecast returns.
- The fundamentals view uses today's fundamentals throughout the backtest (look-ahead).

## Layout

```
src/african_clusters/
  mansa_client.py   REST client + JSON cache
  data.py           universe, price panel, fundamentals builders
  features.py       return/risk/liquidity/fundamental features
  fx.py             daily USD FX rates + currency conversion
  views.py          similarity views → distance matrices
  clustering.py     hierarchical / KMeans, k selection
  compare.py        view agreement + out-of-sample tests
  stability.py      bootstrap and rolling-window stability
  portfolio.py      HRP, cluster parity, representatives, backtest, exposure
  pairs.py          within-cluster cointegration screen
  plots.py          figures
  cli.py            `acs` command
tests/              offline tests (synthetic data, mocked HTTP)
```

## Roadmap

- [x] Convert prices to USD (community FX history; Mansa has no FX history endpoint)
- [ ] Hedged view: local returns minus the interest-rate differential
- [x] Sector labels from `/fundamentals` compared with clusters (ARI)
- [x] Rolling-window clustering to track how cluster membership changes over time
- [ ] Flag/exclude ETFs from the universe
- [ ] Exploration notebook in `notebooks/`

## License

The code is released under the [MIT License](LICENSE). Market data fetched
from the Mansa API, including the derived figures in `docs/data/`, is subject
to [Mansa's terms](https://mansamarkets.com/developers), not this license.
