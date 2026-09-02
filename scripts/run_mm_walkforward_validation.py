"""Walk-forward, out-of-sample validation of the MM market-selection filters
(pace range, volume floor, pre-resolution trading-history length) discovered
in run_mm_proxy_backtest.py's pace segmentation and the Q3 deep dive's volume
and resolution-proximity analyses -- and, critically, run over the UNBIASED
population from build_mm_unbiased_population.py, not the Final-1% strategy's
own extreme-probability-skewed one.

Why this exists: every filter threshold used so far (the Q3 pace bucket, the
V5 volume quintile, the 24h resolution-history cutoff) was DERIVED BY LOOKING
AT the same population it was then reported as performing well on. That's
in-sample curve-fitting / data snooping -- exactly the failure mode a real
fund's risk desk would flag first, ahead of anything about the model itself.
Trying several cuts (pace, then volume, then resolution proximity) in
sequence and keeping whichever looked good compounds the problem.

The fix is standard walk-forward validation:
  1. Split the population chronologically by resolution_time into an EARLIER
     TRAIN period and a LATER TEST period. Any rule derived from train could,
     in real deployment, only have used information available before the
     test period began.
  2. On TRAIN ONLY, grid-search over candidate (pace range, volume floor,
     min pre-resolution trading history) combinations and select whichever
     maximizes the TRAIN per-market MEDIAN markout PnL (median, not total --
     the total is exactly what let a handful of tail markets dominate the
     unvalidated version of this analysis) subject to a minimum surviving-
     market count, so a lucky 2-market combo can't win.
  3. Apply that EXACT combination, unchanged, to TEST. This is the real,
     out-of-sample answer.
  4. Bootstrap the TEST result (resample its markets with replacement many
     times) to see how much of the total is real edge vs. sampling noise
     from a small number of markets, and report max drawdown on the TEST
     equity curve as a basic risk metric.

market_pnl is computed exactly ONCE per market at the base config (spread/
fill assumptions, 15s markout window) for the whole population -- filter
combinations only change which ALREADY-COMPUTED markets are included, so the
grid search itself is pure in-memory aggregation, not repeated backtesting.
"""
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_mm_proxy_backtest as base

REPO = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO / "results" / "polymarket_final_pct"
POPULATION_PATH = RESULTS_DIR / "mm_unbiased_population.csv"

TRAIN_FRACTION = 0.70
MIN_MARKETS_FOR_CANDIDATE = 30  # a filter combo must retain at least this many TRAIN markets to be considered
BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 7

PACE_QUANTILE_EDGES_PCT = [0, 20, 40, 60, 80, 100]
VOLUME_CANDIDATE_PCT = [0, 25, 50, 75]
HISTORY_CANDIDATES_SECONDS = [0, 6 * 3600, 24 * 3600, 48 * 3600]


def load_population_meta(path: Path = POPULATION_PATH) -> dict:
    """cid -> {resolution_time, question, report_bucket} from
    mm_unbiased_population.csv. Every row in that file is already a cleanly-
    resolved market (build_mm_unbiased_population.py's _slim_market drops
    anything resolved_outcome_index can't be determined for), so unlike
    run_mm_proxy_backtest.load_market_meta() there's no resolved_outcome_index
    column to carry -- market_pnl never uses it anyway (see that module's
    docstring on why the model deliberately has no mark-to-resolution term)."""
    meta = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            meta[r["condition_id"]] = {
                "resolution_time": r["resolution_time"],
                "question": r["question"],
                "report_bucket": r["report_bucket"],
            }
    return meta


def chronological_split(per_market: list[dict], train_frac: float = TRAIN_FRACTION) -> tuple[list[dict], list[dict]]:
    """(train, test) split by resolution_time -- train is the chronologically
    EARLIER train_frac of markets, test is the later remainder. Genuine
    walk-forward: a rule fit on train could only have used information that,
    in real deployment, would already have happened before test began."""
    ordered = sorted(per_market, key=lambda r: r["resolution_time"])
    k = round(len(ordered) * train_frac)
    return ordered[:k], ordered[k:]


def evaluate_filter(subset: list[dict], pace_range: tuple[float, float], volume_min: float, history_min_s: float) -> dict:
    """Applies one candidate filter combo to `subset` and reports aggregate +
    per-market stats. Shared by the train-side grid search and the final
    test-side report, so "how a combo is scored" is identical in both
    places."""
    filtered = [
        r for r in subset
        if r["median_inter_trade_s"] is not None
        and pace_range[0] <= r["median_inter_trade_s"] < pace_range[1]
        and r["total_market_volume"] >= volume_min
        and r["pre_resolution_history_s"] is not None
        and r["pre_resolution_history_s"] >= history_min_s
    ]
    if not filtered:
        return {
            "n_markets": 0, "total_pnl_best_case": 0.0, "total_pnl_with_markout_time": 0.0,
            "median_pnl_with_markout_time": None, "mean_pnl_with_markout_time": None, "pct_positive": None,
        }
    markouts = sorted(r["pnl_with_markout_time"] for r in filtered)
    return {
        "n_markets": len(filtered),
        "total_pnl_best_case": round(sum(r["pnl"] for r in filtered), 2),
        "total_pnl_with_markout_time": round(sum(markouts), 2),
        "median_pnl_with_markout_time": round(base._percentile(markouts, 50), 4),
        "mean_pnl_with_markout_time": round(sum(markouts) / len(markouts), 4),
        "pct_positive": round(sum(1 for x in markouts if x > 0) / len(markouts) * 100, 1),
    }


def select_best_filter(train: list[dict], min_markets: int = MIN_MARKETS_FOR_CANDIDATE) -> dict:
    """Grid search over candidate (pace_range, volume_min, history_min_s)
    combinations using ONLY `train`. Selects whichever combo maximizes
    TRAIN per-market MEDIAN markout PnL, subject to `min_markets` surviving.
    Returns None if no candidate combo meets the minimum count."""
    paces = sorted(r["median_inter_trade_s"] for r in train if r["median_inter_trade_s"] is not None)
    if not paces:
        return None
    pace_edges = [base._percentile(paces, p) for p in PACE_QUANTILE_EDGES_PCT]
    pace_candidates = [(pace_edges[i], pace_edges[i + 1]) for i in range(len(pace_edges) - 1)]
    if pace_candidates:
        lo, hi = pace_candidates[-1]
        pace_candidates[-1] = (lo, hi + 1)  # make the top edge inclusive

    volumes = sorted(r["total_market_volume"] for r in train)
    volume_candidates = [base._percentile(volumes, p) for p in VOLUME_CANDIDATE_PCT]

    best = None
    for pace_range in pace_candidates:
        for volume_min in volume_candidates:
            for history_min_s in HISTORY_CANDIDATES_SECONDS:
                result = evaluate_filter(train, pace_range, volume_min, history_min_s)
                if result["n_markets"] < min_markets or result["median_pnl_with_markout_time"] is None:
                    continue
                if best is None or result["median_pnl_with_markout_time"] > best["result"]["median_pnl_with_markout_time"]:
                    best = {"pace_range": pace_range, "volume_min": volume_min,
                             "history_min_s": history_min_s, "result": result}
    return best


def bootstrap_ci(filtered: list[dict], n_iterations: int = BOOTSTRAP_ITERATIONS, seed: int = BOOTSTRAP_SEED) -> dict:
    """Resamples `filtered` markets WITH replacement `n_iterations` times,
    summing pnl_with_markout_time each time, to build an empirical
    distribution of the total PnL a same-sized, same-composition population
    would show -- how much of the headline number is real edge vs. sampling
    noise from a small number of markets."""
    if not filtered:
        return {"n_markets": 0, "n_iterations": n_iterations}
    rng = random.Random(seed)
    values = [r["pnl_with_markout_time"] for r in filtered]
    n = len(values)
    totals = sorted(sum(rng.choice(values) for _ in range(n)) for _ in range(n_iterations))
    return {
        "n_markets": n,
        "n_iterations": n_iterations,
        "p5": round(base._percentile(totals, 5), 2),
        "p25": round(base._percentile(totals, 25), 2),
        "median": round(base._percentile(totals, 50), 2),
        "p75": round(base._percentile(totals, 75), 2),
        "p95": round(base._percentile(totals, 95), 2),
        "pct_iterations_positive": round(sum(1 for t in totals if t > 0) / n_iterations * 100, 1),
    }


def compute_drawdown(filtered: list[dict]) -> dict:
    """Chronological (by resolution_time) equity curve on the filtered
    markout PnL, plus max drawdown -- a basic risk metric no prior version
    of this analysis reported."""
    ordered = sorted(filtered, key=lambda r: r["resolution_time"])
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    curve = []
    for r in ordered:
        equity += r["pnl_with_markout_time"]
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
        curve.append((r["resolution_time"][:10], round(equity, 2)))
    return {"final_equity": round(equity, 2), "max_drawdown": round(max_dd, 2), "equity_curve": curve}


def main():
    meta = load_population_meta()
    print(f"[walkforward] {len(meta)} markets in the unbiased population")

    per_market = []
    n_no_trades = 0
    for i, (cid, m) in enumerate(meta.items()):
        raw_trades = base.pmf.fetch_market_trades(cid)
        if not raw_trades:
            n_no_trades += 1
            continue
        sorted_trades, total_market_volume = base.parse_and_sort_trades(raw_trades)
        if not sorted_trades:
            n_no_trades += 1
            continue
        r = base.market_pnl(sorted_trades, total_market_volume, base.BASE_HALF_SPREAD, base.BASE_FILL_SHARE)
        if r["n_captured"] == 0:
            continue
        resolution_epoch = base.resolution_epoch_seconds(m["resolution_time"])
        pre_resolution_history_s = (
            resolution_epoch - sorted_trades[0]["timestamp"] if resolution_epoch is not None else None
        )
        per_market.append({
            "condition_id": cid, "question": m["question"][:80] if m["question"] else "",
            "resolution_time": m["resolution_time"], "report_bucket": m["report_bucket"],
            "pnl": round(r["pnl_best_case"], 4),
            "pnl_with_markout_time": round(r["pnl_with_markout_time"], 4),
            "n_captured_trades": r["n_captured"],
            "total_market_volume": round(total_market_volume, 2),
            "median_inter_trade_s": base.market_pace_seconds(sorted_trades),
            "pre_resolution_history_s": pre_resolution_history_s,
        })
        if (i + 1) % 250 == 0:
            print(f"  [walkforward] {i + 1}/{len(meta)} markets processed ...", flush=True)

    print(f"[walkforward] {len(per_market)} markets with captured flow ({n_no_trades} had no usable trades)")

    train, test = chronological_split(per_market)
    print(f"[walkforward] TRAIN: {len(train)} markets ({train[0]['resolution_time'][:10]} to {train[-1]['resolution_time'][:10]})")
    print(f"[walkforward] TEST:  {len(test)} markets ({test[0]['resolution_time'][:10]} to {test[-1]['resolution_time'][:10]})")

    unfiltered_test = evaluate_filter(test, (0.0, float("inf")), 0.0, 0.0)
    print(f"\nUnfiltered TEST baseline (no market selection at all): "
          f"n={unfiltered_test['n_markets']}  total_markout=${unfiltered_test['total_pnl_with_markout_time']:,.2f}  "
          f"median=${unfiltered_test['median_pnl_with_markout_time']}  %positive={unfiltered_test['pct_positive']}%")

    winner = select_best_filter(train)
    if winner is None:
        print("\n[walkforward] No filter combination met the minimum TRAIN market count -- population too small. Stopping.")
        return

    lo, hi = winner["pace_range"]
    print(f"\nBest TRAIN-derived filter (selected by TRAIN median markout PnL, "
          f">= {MIN_MARKETS_FOR_CANDIDATE} markets required):")
    print(f"  pace range: [{lo:.1f}s, {hi:.1f}s)   volume floor: ${winner['volume_min']:,.2f}   "
          f"min pre-resolution history: {winner['history_min_s'] / 3600:.1f}h")
    print(f"  TRAIN result: n={winner['result']['n_markets']}  "
          f"total_markout=${winner['result']['total_pnl_with_markout_time']:,.2f}  "
          f"median=${winner['result']['median_pnl_with_markout_time']}  %positive={winner['result']['pct_positive']}%")

    test_result = evaluate_filter(test, winner["pace_range"], winner["volume_min"], winner["history_min_s"])
    print(f"\n=== OUT-OF-SAMPLE TEST result (same filter, held-out period, never seen during selection) ===")
    print(f"  n={test_result['n_markets']}  total_best_case=${test_result['total_pnl_best_case']:,.2f}  "
          f"total_markout=${test_result['total_pnl_with_markout_time']:,.2f}  "
          f"median=${test_result['median_pnl_with_markout_time']}  mean=${test_result['mean_pnl_with_markout_time']}  "
          f"%positive={test_result['pct_positive']}%")

    filtered_test = [
        r for r in test
        if r["median_inter_trade_s"] is not None and winner["pace_range"][0] <= r["median_inter_trade_s"] < winner["pace_range"][1]
        and r["total_market_volume"] >= winner["volume_min"]
        and r["pre_resolution_history_s"] is not None and r["pre_resolution_history_s"] >= winner["history_min_s"]
    ]

    ci = bootstrap_ci(filtered_test)
    print(f"\nBootstrap CI on the TEST result ({ci['n_iterations']} resamples of its {ci['n_markets']} markets):")
    print(f"  p5=${ci.get('p5')}  p25=${ci.get('p25')}  median=${ci.get('median')}  p75=${ci.get('p75')}  p95=${ci.get('p95')}")
    print(f"  P(total PnL > 0) = {ci.get('pct_iterations_positive')}%")

    dd = compute_drawdown(filtered_test)
    print(f"\nTEST equity curve: final=${dd['final_equity']:,.2f}  max_drawdown=${dd['max_drawdown']:,.2f}")

    category_mix = {}
    for r in filtered_test:
        category_mix[r["report_bucket"]] = category_mix.get(r["report_bucket"], 0) + 1
    print(f"TEST filtered population category mix: {category_mix}")

    import json
    out = {
        "train_fraction": TRAIN_FRACTION,
        "n_train": len(train), "n_test": len(test),
        "unfiltered_test_baseline": unfiltered_test,
        "selected_filter": {
            "pace_range_seconds": list(winner["pace_range"]),
            "volume_min": winner["volume_min"],
            "history_min_seconds": winner["history_min_s"],
            "train_result": winner["result"],
        },
        "test_result": test_result,
        "bootstrap_ci": ci,
        "drawdown": dd,
        "test_category_mix": category_mix,
    }
    out_path = RESULTS_DIR / "mm_walkforward_validation.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
