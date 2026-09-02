"""Tests the "free, no-new-data" improvements from the production-engineering
roadmap against the unbiased population: category-specific markout windows,
VPIN-driven dynamic spread, and inventory-aware quote skewing (see
market_pnl_advanced / compute_vpin_series in run_mm_proxy_backtest.py for the
mechanics). All three reuse data already on disk -- no new fetches, no paid
services, no credentials.

Four configurations are compared side by side on the SAME population, never
replacing the existing baseline number:
  1. FLAT BASELINE       -- market_pnl, flat 15s markout window for every
                             market (exactly what every prior result in this
                             project's history used).
  2. CATEGORY WINDOW      -- market_pnl, but the markout window is each
                             market's own report_bucket's typical trading
                             pace (median inter-trade gap across that
                             category), clamped to [5s, 120s] -- "react about
                             as fast as this category's own cadence," rather
                             than one universal number for every category.
  3. ADVANCED (flat)      -- market_pnl_advanced (VPIN spread + inventory
                             skew) at the flat 15s window.
  4. ADVANCED + CATEGORY  -- market_pnl_advanced at each market's own
                             category-tailored window.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_mm_proxy_backtest as base
import run_mm_walkforward_validation as wf

REPO = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO / "results" / "polymarket_final_pct"

REPORT_BUCKET_TO_FEE_CATEGORY = {
    "crypto_price": "crypto", "sports": "sports", "politics": "politics", "other": "other",
}
CATEGORY_WINDOW_MIN_S = 5.0
CATEGORY_WINDOW_MAX_S = 120.0


def compute_category_windows(per_market_pace: dict) -> dict:
    """per_market_pace: report_bucket -> list of per-market median_inter_trade_s
    (None entries already excluded). Returns report_bucket -> markout window
    seconds: the category's own median pace, clamped to
    [CATEGORY_WINDOW_MIN_S, CATEGORY_WINDOW_MAX_S] -- "react about as fast as
    this category's own typical cadence," bounded to sane limits, rather than
    one flat number applied to every category regardless of how it trades."""
    windows = {}
    for bucket, paces in per_market_pace.items():
        if not paces:
            continue
        s = sorted(paces)
        n = len(s)
        median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
        windows[bucket] = max(CATEGORY_WINDOW_MIN_S, min(CATEGORY_WINDOW_MAX_S, median))
    return windows


def _new_agg() -> dict:
    return {"pnl_best_case": 0.0, "pnl_with_markout_time": 0.0, "n_markets_active": 0}


def _accumulate(agg: dict, r: dict) -> None:
    agg["pnl_best_case"] += r["pnl_best_case"]
    agg["pnl_with_markout_time"] += r["pnl_with_markout_time"]
    if r["n_captured"] > 0:
        agg["n_markets_active"] += 1


def main():
    meta = wf.load_population_meta()
    print(f"[advanced] {len(meta)} markets in the unbiased population")

    # Pass 1: fetch/parse each market once, compute its own pace -- needed
    # before category windows can be derived.
    market_data = {}  # cid -> (sorted_trades, total_market_volume, pace, fee_category)
    per_market_pace = {b: [] for b in base.pmf.REPORT_BUCKETS}
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
        pace = base.market_pace_seconds(sorted_trades)
        fee_category = REPORT_BUCKET_TO_FEE_CATEGORY.get(m["report_bucket"], "other")
        market_data[cid] = (sorted_trades, total_market_volume, pace, m["report_bucket"], fee_category)
        if pace is not None:
            per_market_pace.setdefault(m["report_bucket"], []).append(pace)
        if (i + 1) % 250 == 0:
            print(f"  [advanced] {i + 1}/{len(meta)} markets scanned ...", flush=True)

    print(f"[advanced] {len(market_data)} markets with usable trades ({n_no_trades} had none)")

    category_windows = compute_category_windows(per_market_pace)
    print("\nCategory-specific markout windows (median pace, clamped to "
          f"[{CATEGORY_WINDOW_MIN_S}s, {CATEGORY_WINDOW_MAX_S}s]):")
    for bucket, w in category_windows.items():
        print(f"  {bucket:<15} {w:.1f}s  (n={len(per_market_pace.get(bucket, []))} markets with measurable pace)")

    aggs = {name: _new_agg() for name in
            ["flat_baseline", "category_window", "advanced_flat", "advanced_category"]}
    advanced_diag = {"avg_vpin_sum": 0.0, "avg_vpin_n": 0, "n_inventory_capped_total": 0}

    for i, (cid, (sorted_trades, total_market_volume, pace, report_bucket, fee_category)) in enumerate(market_data.items()):
        cat_window = category_windows.get(report_bucket, base.MARKOUT_WINDOW_SECONDS)

        r_flat = base.market_pnl(sorted_trades, total_market_volume, base.BASE_HALF_SPREAD, base.BASE_FILL_SHARE)
        _accumulate(aggs["flat_baseline"], r_flat)

        r_cat = base.market_pnl(sorted_trades, total_market_volume, base.BASE_HALF_SPREAD, base.BASE_FILL_SHARE,
                                 markout_window_seconds=cat_window)
        _accumulate(aggs["category_window"], r_cat)

        r_adv_flat = base.market_pnl_advanced(sorted_trades, total_market_volume, base.BASE_HALF_SPREAD, base.BASE_FILL_SHARE)
        _accumulate(aggs["advanced_flat"], r_adv_flat)

        r_adv_cat = base.market_pnl_advanced(sorted_trades, total_market_volume, base.BASE_HALF_SPREAD, base.BASE_FILL_SHARE,
                                              markout_window_seconds=cat_window)
        _accumulate(aggs["advanced_category"], r_adv_cat)
        if r_adv_cat["avg_vpin"] is not None:
            advanced_diag["avg_vpin_sum"] += r_adv_cat["avg_vpin"]
            advanced_diag["avg_vpin_n"] += 1
        advanced_diag["n_inventory_capped_total"] += r_adv_cat["n_inventory_capped"]

        if (i + 1) % 250 == 0:
            print(f"  [advanced] {i + 1}/{len(market_data)} markets backtested ...", flush=True)

    print(f"\n=== Results across {len(market_data)} markets (base_half_spread=${base.BASE_HALF_SPREAD}, "
          f"base_fill_share={base.BASE_FILL_SHARE:.0%}) ===")
    for name, agg in aggs.items():
        gap = (
            (1 - agg["pnl_with_markout_time"] / agg["pnl_best_case"]) * 100
            if agg["pnl_best_case"] else None
        )
        gap_str = f"{gap:.1f}%" if gap is not None else "n/a"
        print(f"  {name:<20} n_active={agg['n_markets_active']:<5} "
              f"best_case=${agg['pnl_best_case']:>12,.2f}  markout=${agg['pnl_with_markout_time']:>12,.2f}  gap={gap_str}")

    avg_vpin = advanced_diag["avg_vpin_sum"] / advanced_diag["avg_vpin_n"] if advanced_diag["avg_vpin_n"] else None
    print(f"\nAdvanced-model diagnostics: mean per-market avg_vpin={avg_vpin}, "
          f"total inventory-capped fill events={advanced_diag['n_inventory_capped_total']}")

    print("\nDoes each free improvement actually help, measured against the flat baseline?")
    base_markout = aggs["flat_baseline"]["pnl_with_markout_time"]
    for name in ["category_window", "advanced_flat", "advanced_category"]:
        delta = aggs[name]["pnl_with_markout_time"] - base_markout
        direction = "IMPROVES" if delta > 0 else "WORSENS" if delta < 0 else "NO CHANGE"
        print(f"  {name:<20} markout PnL delta vs. flat baseline: ${delta:>+12,.2f}  [{direction}]")

    out = {
        "category_windows_seconds": category_windows,
        "results": aggs,
        "advanced_diagnostics": {
            "mean_avg_vpin": avg_vpin,
            "n_inventory_capped_total": advanced_diag["n_inventory_capped_total"],
        },
    }
    out_path = RESULTS_DIR / "mm_proxy_advanced_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
