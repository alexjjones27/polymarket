"""Deep dive on Q3 -- the one pace bucket that run_mm_proxy_backtest.py's
market-selection analysis found to be unambiguously good: markets with a
median 56-280s gap between real trades are the only segment where
adverse-selection-adjusted PnL is both meaningfully positive AND backed by
real trading volume (Q5, the slowest bucket, "survives" markout even better
in percentage terms but has almost no volume to capture -- see that
script's own pace_breakdown output).

This script does NOT re-derive Q3 membership independently. It reads the
condition_ids already labeled pace_bucket=="Q3" out of the saved
mm_proxy_results.json, so the two scripts can never silently disagree about
which markets are in scope -- if you change the pace-bucketing logic, re-run
run_mm_proxy_backtest.py first and this script picks up the new membership
automatically. Everything below re-slices the SAME reused, disk-cached trade
tapes (via fetch_market_trades) -- no new network calls.

Four questions this asks that the whole-population report doesn't answer:
  1. Does Q3's positive result survive the full half_spread x fill_share
     sensitivity grid, or was it only positive at the one base config?
  2. At exactly what reaction speed does Q3 stop being profitable? (The
     coarse 4-point grid in run_mm_proxy_backtest.py showed positive at 60s,
     negative at 300s -- this finds the actual crossover in between.)
  3. Is the profit broad-based across the 573 Q3 markets, or dominated by a
     handful of them (same concentration check already applied to the whole
     population)?
  4. Which specific markets are the best and worst performers within Q3, and
     what do they have in common?
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_mm_proxy_backtest as base

REPO = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO / "results" / "polymarket_final_pct"
RESULTS_PATH = RESULTS_DIR / "mm_proxy_results.json"

TARGET_BUCKET = "Q3"
# Finer than run_mm_proxy_backtest.py's MARKOUT_TIME_GRID_SECONDS ([5,15,60,300])
# specifically to locate the crossover it only bracketed (positive at 60s,
# negative at 300s).
TIME_WINDOW_FINE_GRID = [5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 300]
# Percentile cutoffs on total_market_volume, for the "does a volume floor on
# top of the pace filter turn a few-winners-carry-it bucket into something
# broad-based" sweep.
VOLUME_PERCENTILE_THRESHOLDS = [0, 10, 25, 50, 75, 90]


def load_bucket_condition_ids(results_path: Path = RESULTS_PATH, bucket: str = TARGET_BUCKET) -> list[str]:
    """Reads the condition_ids already labeled pace_bucket==`bucket` out of a
    prior run_mm_proxy_backtest.py run, rather than re-deriving quantile
    edges here -- guarantees the two scripts can't silently disagree about
    which markets are in scope."""
    with open(results_path) as f:
        prior = json.load(f)
    if prior.get("best_pace_bucket") != bucket:
        print(f"WARNING: {results_path.name}'s best_pace_bucket is "
              f"{prior.get('best_pace_bucket')!r}, not {bucket!r} -- the data may "
              f"have changed since this deep dive was written. Re-run "
              f"run_mm_proxy_backtest.py and re-check before trusting this output.")
    cids = [r["condition_id"] for r in prior["per_market_base_case"] if r.get("pace_bucket") == bucket]
    if not cids:
        raise SystemExit(f"No markets found with pace_bucket == {bucket!r} in {results_path} -- "
                          f"run run_mm_proxy_backtest.py first.")
    return cids


def find_crossover(time_grid: list[dict]):
    """First adjacent (window_seconds, total_pnl) pair in `time_grid` where
    total_pnl goes from >=0 to <0, i.e. the reaction speed beyond which the
    segment stops being profitable. Returns None if it's positive throughout
    (or negative throughout, or the grid is too short to tell)."""
    for a, b in zip(time_grid, time_grid[1:]):
        if a["total_pnl"] >= 0 and b["total_pnl"] < 0:
            return a["window_seconds"], b["window_seconds"]
    return None


def build_volume_threshold_sweep(per_market: list[dict], percentiles: list[float] = VOLUME_PERCENTILE_THRESHOLDS) -> list[dict]:
    """For each percentile P, keep only markets whose total_market_volume is
    at or above that percentile of the population, and report the PER-MARKET
    median/mean of pnl_with_markout_time for what's left, not just the sum.
    The sum alone can hide one big winner carrying an otherwise-flat group
    (exactly what the deep dive found for Q3 as a whole); the median is the
    actual answer to "does the typical market above this volume bar work,"
    which is the real question a volume filter is supposed to answer."""
    volumes = sorted(r["total_market_volume"] for r in per_market)
    out = []
    for p in percentiles:
        cutoff = base._percentile(volumes, p)
        subset = [r for r in per_market if r["total_market_volume"] >= cutoff] if cutoff is not None else []
        markouts = sorted(r["pnl_with_markout_time"] for r in subset)
        out.append({
            "percentile": p,
            "volume_cutoff": round(cutoff, 2) if cutoff is not None else None,
            "n_markets": len(subset),
            "total_pnl_best_case": round(sum(r["pnl"] for r in subset), 2),
            "total_pnl_with_markout_time": round(sum(markouts), 2),
            "median_pnl_with_markout_time": round(base._percentile(markouts, 50), 2) if markouts else None,
            "mean_pnl_with_markout_time": round(sum(markouts) / len(markouts), 2) if markouts else None,
            "pct_markets_positive_markout": round(sum(1 for x in markouts if x > 0) / len(markouts) * 100, 1) if markouts else None,
        })
    return out


def main():
    cids = load_bucket_condition_ids()
    meta = base.load_market_meta()
    print(f"[q3-deep-dive] {len(cids)} markets in pace bucket {TARGET_BUCKET} "
          f"(median 56-280s between real trades), reusing cached trade tapes -- no new fetches")

    market_data = {}  # cid -> (sorted_trades, total_market_volume)
    for cid in cids:
        raw_trades = base.pmf.fetch_market_trades(cid)
        sorted_trades, total_market_volume = base.parse_and_sort_trades(raw_trades)
        if sorted_trades:
            market_data[cid] = (sorted_trades, total_market_volume)
    print(f"[q3-deep-dive] {len(market_data)} of {len(cids)} still have parseable cached trades")

    # (1) Full half_spread x fill_share sensitivity grid, restricted to this bucket.
    sensitivity = {}
    for hs in base.HALF_SPREADS:
        for fs in base.FILL_SHARES:
            sensitivity[f"hs{hs}_fs{fs}"] = {
                "half_spread": hs, "fill_share": fs,
                "total_pnl_best_case": 0.0, "total_pnl_with_markout_time": 0.0,
                "n_markets_active": 0,
            }
    for sorted_trades, total_market_volume in market_data.values():
        for cfg in sensitivity.values():
            r = base.market_pnl(sorted_trades, total_market_volume, cfg["half_spread"], cfg["fill_share"])
            cfg["total_pnl_best_case"] += r["pnl_best_case"]
            cfg["total_pnl_with_markout_time"] += r["pnl_with_markout_time"]
            if r["n_captured"] > 0:
                cfg["n_markets_active"] += 1

    # (2) Fine-grained reaction-speed grid at the base config -- find the crossover.
    time_grid = []
    for w in TIME_WINDOW_FINE_GRID:
        total = sum(
            base.market_pnl(sorted_trades, total_market_volume, base.BASE_HALF_SPREAD, base.BASE_FILL_SHARE,
                             markout_window_seconds=w)["pnl_with_markout_time"]
            for sorted_trades, total_market_volume in market_data.values()
        )
        time_grid.append({"window_seconds": w, "total_pnl": round(total, 2)})
    crossover = find_crossover(time_grid)

    # (3)/(4) Per-market detail at the base config -- concentration, category mix,
    # equity curve, and the actual best/worst performers.
    per_market = []
    for cid, (sorted_trades, total_market_volume) in market_data.items():
        m = meta[cid]
        r = base.market_pnl(sorted_trades, total_market_volume, base.BASE_HALF_SPREAD, base.BASE_FILL_SHARE)
        if r["n_captured"] == 0:
            continue
        per_market.append({
            "condition_id": cid, "question": m["question"][:80],
            "resolution_time": m["resolution_time"], "report_bucket": m["report_bucket"],
            "pnl": round(r["pnl_best_case"], 4),
            "pnl_with_markout_trades": round(r["pnl_with_markout_trades"], 4),
            "pnl_with_markout_time": round(r["pnl_with_markout_time"], 4),
            "n_captured_trades": r["n_captured"],
            "captured_notional": round(r["captured_notional"], 2),
            "total_market_volume": round(total_market_volume, 2),
            "volume_share_captured": round(r["volume_share_captured"], 4) if r["volume_share_captured"] is not None else None,
            "median_inter_trade_s": base.market_pace_seconds(sorted_trades),
        })

    concentration_best = base.concentration_by_top_n(per_market, "pnl")
    concentration_markout = base.concentration_by_top_n(per_market, "pnl_with_markout_time", worst=True)
    category_report = base.category_breakdown(per_market)

    # Does a volume filter turn "a few winners carry a breakeven bucket" into
    # something broad-based? Segment by total_market_volume (an inherent
    # property of the market, same kind of characteristic pace was) and check
    # both the top volume bucket's own concentration AND a threshold sweep's
    # PER-MARKET outcome -- a bigger total can still hide one big winner;
    # only the per-market median/mean says whether the TYPICAL high-volume
    # market actually works.
    base.assign_quantile_buckets(per_market, "total_market_volume", "volume_bucket", n_quantiles=5, prefix="V")
    volume_report = base.quantile_breakdown(per_market, "volume_bucket", "total_market_volume")
    best_volume_bucket = next(iter(volume_report), None)
    top_volume_subset = [r for r in per_market if r["volume_bucket"] == best_volume_bucket]
    concentration_top_volume = base.concentration_by_top_n(top_volume_subset, "pnl_with_markout_time")
    volume_threshold_sweep = build_volume_threshold_sweep(per_market)

    per_market.sort(key=lambda r: r["resolution_time"])
    equity = base.START_BANKROLL
    equity_markout = base.START_BANKROLL
    curve = [(per_market[0]["resolution_time"][:10] if per_market else None, equity, equity_markout)]
    for r in per_market:
        equity += r["pnl"]
        equity_markout += r["pnl_with_markout_time"]
        curve.append((r["resolution_time"][:10], round(equity, 2), round(equity_markout, 2)))

    by_markout = sorted(per_market, key=lambda r: -r["pnl_with_markout_time"])
    top10 = by_markout[:10]
    bottom10 = list(reversed(by_markout[-10:]))

    vol_shares = [r["volume_share_captured"] for r in per_market if r["volume_share_captured"] is not None]

    base_key = f"hs{base.BASE_HALF_SPREAD}_fs{base.BASE_FILL_SHARE}"
    base_cfg = sensitivity[base_key]

    summary = {
        "target_bucket": TARGET_BUCKET,
        "n_markets_in_bucket": len(cids),
        "n_markets_with_captured_flow": len(per_market),
        "base_case": {"half_spread": base.BASE_HALF_SPREAD, "fill_share": base.BASE_FILL_SHARE},
        "total_pnl_best_case": round(base_cfg["total_pnl_best_case"], 2),
        "total_pnl_with_markout_time_15s": round(base_cfg["total_pnl_with_markout_time"], 2),
        "sensitivity_grid": list(sensitivity.values()),
        "time_window_fine_grid": time_grid,
        "profitability_crossover_seconds": crossover,
        "concentration_best_case": concentration_best,
        "concentration_markout_worst": concentration_markout,
        "category_breakdown": category_report,
        "volume_breakdown": volume_report,
        "best_volume_bucket": best_volume_bucket,
        "concentration_within_best_volume_bucket": concentration_top_volume,
        "volume_threshold_sweep": volume_threshold_sweep,
        "volume_share_captured": {
            "mean": round(sum(vol_shares) / len(vol_shares), 4) if vol_shares else None,
            "max": round(max(vol_shares), 4) if vol_shares else None,
            "n_samples": len(vol_shares),
        },
        "equity_curve": curve,
        "top10_by_markout_time": top10,
        "bottom10_by_markout_time": bottom10,
        "per_market": per_market,
    }

    print(f"\n=== Q3 deep dive: {len(per_market)} markets with captured flow "
          f"(half_spread=${base.BASE_HALF_SPREAD}, fill_share={base.BASE_FILL_SHARE:.0%}) ===")
    print(f"Best case:                  ${base_cfg['total_pnl_best_case']:>10,.2f}")
    print(f"15s markout (base config):  ${base_cfg['total_pnl_with_markout_time']:>10,.2f}")

    print("\n(1) Sensitivity grid -- does the positive result survive other spread/fill assumptions?")
    for cfg in sensitivity.values():
        flag = "OK" if cfg["total_pnl_with_markout_time"] > 0 else "NEGATIVE"
        print(f"  half_spread=${cfg['half_spread']:<6} fill_share={cfg['fill_share']:<6.0%}  "
              f"best_case=${cfg['total_pnl_best_case']:>9,.2f}  markout_time=${cfg['total_pnl_with_markout_time']:>9,.2f}  [{flag}]")

    print("\n(2) Reaction-speed crossover -- how fast do you actually need to react?")
    for row in time_grid:
        print(f"  window={row['window_seconds']:>4}s   total_pnl=${row['total_pnl']:>10,.2f}")
    if crossover:
        print(f"  Crossover: profitable through {crossover[0]}s, negative by {crossover[1]}s.")
    else:
        print("  No sign change across the grid tested.")

    print("\n(3) Concentration -- is this broad-based or a few markets carrying it?")
    for row in concentration_best["by_top_n"]:
        print(f"  best-case top {row['n']:<3}: ${row['pnl']:>9,.2f}  ({row['pct_of_total']}% of total)")
    for row in concentration_markout["by_top_n"]:
        print(f"  markout worst {row['n']:<3}: ${row['pnl']:>9,.2f}  ({row['pct_of_total']}% of total loss)")

    print("\nCategory mix within Q3 (15s markout):")
    for name, b in category_report.items():
        pct = f"{b['markout_time_pct_of_best_case']}%" if b['markout_time_pct_of_best_case'] is not None else "n/a"
        print(f"  {name:<15} n_markets={b['n_markets']:<5} best_case=${b['pnl_best_case']:>9,.2f}  "
              f"markout_time=${b['pnl_with_markout_time']:>9,.2f}  ({pct} of best case)")

    print(f"\nLiquidity: mean volume share captured {summary['volume_share_captured']['mean']}, "
          f"max {summary['volume_share_captured']['max']} (cap is {base.MAX_MARKET_VOLUME_SHARE})")

    print(f"\n(5) Volume segmentation within Q3 -- does market size (V1=lowest .. V5=highest "
          f"total_market_volume) explain the concentration?")
    for name, b in volume_report.items():
        pct = f"{b['markout_time_pct_of_best_case']}%" if b['markout_time_pct_of_best_case'] is not None else "n/a"
        lo, hi = b["total_market_volume_range"]
        print(f"  {name}  n_markets={b['n_markets']:<5} volume=${lo:>9,.0f}-${hi:>10,.0f}  "
              f"best_case=${b['pnl_best_case']:>9,.2f}  markout_time=${b['pnl_with_markout_time']:>9,.2f}  ({pct} of best case)")
    print(f"  Best volume bucket: {best_volume_bucket} -- concentration within it (top N as % of ITS OWN total):")
    for row in concentration_top_volume["by_top_n"]:
        print(f"    top {row['n']:<3}: ${row['pnl']:>9,.2f}  ({row['pct_of_total']}% of {best_volume_bucket}'s total)")

    print(f"\nVolume threshold sweep -- as the minimum market-volume bar rises, does the TYPICAL "
          f"(median) market start working, or is it still just the total being carried by a few?")
    for row in volume_threshold_sweep:
        print(f"  p{row['percentile']:>2} (volume>=${row['volume_cutoff']:>9,.0f})  n={row['n_markets']:<4}  "
              f"total_markout=${row['total_pnl_with_markout_time']:>9,.2f}  "
              f"median=${row['median_pnl_with_markout_time']:>8,.2f}  mean=${row['mean_pnl_with_markout_time']:>8,.2f}  "
              f"%positive={row['pct_markets_positive_markout']}%")

    print("\n(4) Top 5 markets by 15s markout PnL:")
    for r in top10[:5]:
        print(f"  ${r['pnl_with_markout_time']:>9,.2f}  [{r['report_bucket']:<12}]  {r['question']!r} "
              f"({r['n_captured_trades']} trades, gap={r['median_inter_trade_s']}s)")
    print("Bottom 5 markets by 15s markout PnL:")
    for r in bottom10[:5]:
        print(f"  ${r['pnl_with_markout_time']:>9,.2f}  [{r['report_bucket']:<12}]  {r['question']!r} "
              f"({r['n_captured_trades']} trades, gap={r['median_inter_trade_s']}s)")

    out_path = RESULTS_DIR / "mm_proxy_q3_deep_dive.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
