"""Follow-up to Section 9 (run_mm_proxy_advanced.py): tests whether the four
specific gaps that document flagged -- VPIN bucket miscalibration (mean VPIN
0.88), a linear/binary inventory skew, no volatility-scaling on position
size, and no reaction to a fill that goes on to look toxic -- can narrow the
gap between best-case spread capture and realistic markout loss any further,
using ONLY data already on disk (mm_risk_controls_v3.py; no real L2 data, no
paid services).

Same walk-forward discipline as run_mm_walkforward_validation.py throughout,
because exploratory whole-population tuning of these many new knobs is
exactly the kind of in-sample pattern-matching Section 3/4 of the methodology
doc exists to catch: every hyperparameter here is selected on TRAIN ONLY
(by per-market MEDIAN markout PnL, same criterion as select_best_filter) and
then applied UNCHANGED to TEST. Four sequential stages, each holding the
previous stage's winner fixed (coordinate-ascent, not a full cross product --
36 candidates total instead of a combinatorial explosion):

  1. Toxicity signal: which bucket size / mechanism narrows the VPIN
     calibration problem (Section 9's mean-VPIN-near-1.0 finding) and
     improves TRAIN median markout PnL?
  2. Inventory skew: linear vs. sigmoid, and what inventory_limit_notional?
  3. Volatility-scaled position cap: on or off, what sensitivity?
  4. Post-fill cooldown: on or off, what parameters?

The winning combined config is then (a) reported on the SAME whole
1,331-market unbiased population used throughout, in the same table format
as Section 9, for direct comparison against the existing -$112,335.07 /
-$94,667.06 numbers; and (b) run through the FULL walk-forward pipeline
(TRAIN-selected market-selection filter + bootstrap CI + drawdown on TEST)
to answer the actual strategic question: does combining better risk controls
with the existing market-selection filter search flip the out-of-sample
result to positive?
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import mm_risk_controls_v3 as v3
import run_mm_proxy_backtest as base
import run_mm_walkforward_validation as wf

REPO = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO / "results" / "polymarket_final_pct"


def load_tapes(meta: dict) -> dict:
    """Fetches (cached) + parses every market's trade tape exactly ONCE,
    reused across every candidate config below -- the expensive part
    (network fetch + parse) shouldn't be repeated per config, only the
    market_pnl_v3 call itself."""
    tapes = {}
    n_no_trades = 0
    for i, (cid, m) in enumerate(meta.items()):
        raw = base.pmf.fetch_market_trades(cid)
        if not raw:
            n_no_trades += 1
            continue
        sorted_trades, total_market_volume = base.parse_and_sort_trades(raw)
        if not sorted_trades:
            n_no_trades += 1
            continue
        tapes[cid] = (sorted_trades, total_market_volume, m)
        if (i + 1) % 500 == 0:
            print(f"  [v3] {i + 1}/{len(meta)} tapes loaded ...", flush=True)
    print(f"[v3] {len(tapes)} markets with usable trades ({n_no_trades} had none)")
    return tapes


def per_market_stats_for_config(tapes: dict, config: dict) -> list[dict]:
    """market_pnl_v3(**config) on every tape, in the same row shape
    run_mm_walkforward_validation's evaluate_filter/select_best_filter/
    bootstrap_ci/compute_drawdown already expect -- lets this script reuse
    every one of those functions unchanged instead of duplicating them."""
    per_market = []
    for cid, (sorted_trades, total_market_volume, m) in tapes.items():
        r = v3.market_pnl_v3(sorted_trades, total_market_volume, base.BASE_HALF_SPREAD, base.BASE_FILL_SHARE,
                              **config)
        if r["n_captured"] == 0:
            continue
        resolution_epoch = base.resolution_epoch_seconds(m["resolution_time"])
        pre_resolution_history_s = (
            resolution_epoch - sorted_trades[0]["timestamp"] if resolution_epoch is not None else None
        )
        per_market.append({
            "condition_id": cid, "question": (m["question"] or "")[:80],
            "resolution_time": m["resolution_time"], "report_bucket": m["report_bucket"],
            "pnl": round(r["pnl_best_case"], 4),
            "pnl_with_markout_time": round(r["pnl_with_markout_time"], 4),
            "n_captured_trades": r["n_captured"],
            "total_market_volume": round(total_market_volume, 2),
            "median_inter_trade_s": base.market_pace_seconds(sorted_trades),
            "pre_resolution_history_s": pre_resolution_history_s,
            "avg_toxicity": r["avg_vpin"],
        })
    return per_market


NO_FILTER = ((0.0, float("inf")), 0.0, 0.0)  # pace_range, volume_min, history_min_s that matches every market


def unfiltered_summary(per_market: list[dict]) -> dict:
    """evaluate_filter with a pass-everything filter -- the whole-set
    aggregate, reusing wf's own scoring function so config selection and
    market-selection-filter selection are scored identically."""
    pace_range, volume_min, history_min_s = NO_FILTER
    return wf.evaluate_filter(per_market, pace_range, volume_min, history_min_s)


def select_best_config(tapes: dict, train_cids: set, candidates: dict) -> tuple[str, dict, dict]:
    """Runs every named candidate config, scores it on TRAIN median markout
    PnL (matching select_best_filter's own criterion), and returns
    (winning_name, winning_config, {name: {"train": ..., "avg_toxicity": ...}}
    for every candidate) so the full sweep is visible in the saved output,
    not just the winner."""
    results = {}
    best_name, best_median = None, None
    for name, config in candidates.items():
        t0 = time.time()
        per_market = per_market_stats_for_config(tapes, config)
        train_rows = [r for r in per_market if r["condition_id"] in train_cids]
        train_summary = unfiltered_summary(train_rows)
        tox_samples = [r["avg_toxicity"] for r in per_market if r["avg_toxicity"] is not None]
        mean_tox = sum(tox_samples) / len(tox_samples) if tox_samples else None
        results[name] = {"train": train_summary, "mean_avg_toxicity": mean_tox, "config": config}
        median = train_summary["median_pnl_with_markout_time"]
        print(f"  [{name:28s}] TRAIN n={train_summary['n_markets']:4d} median=${median!s:>10} "
              f"total=${train_summary['total_pnl_with_markout_time']:>12,.2f} "
              f"mean_tox={mean_tox} ({time.time() - t0:.1f}s)")
        if median is not None and (best_median is None or median > best_median):
            best_name, best_median = name, median
    return best_name, candidates[best_name], results


def main():
    meta = wf.load_population_meta()
    print(f"[v3] {len(meta)} markets in the unbiased population")
    tapes = load_tapes(meta)

    train_meta, test_meta = wf.chronological_split(
        [{"condition_id": cid, "resolution_time": m["resolution_time"]} for cid, (_, _, m) in tapes.items()])
    train_cids = {r["condition_id"] for r in train_meta}
    test_cids = {r["condition_id"] for r in test_meta}
    print(f"[v3] TRAIN: {len(train_cids)} markets   TEST: {len(test_cids)} markets (chronological split, same as walk-forward)")

    sweeps = {}

    # -----------------------------------------------------------------
    # Stage 1: toxicity signal (VPIN bucket size, fixed vs. dynamic, or
    # order-imbalance) -- holding skew at the existing linear/limit=100
    # reference so the comparison isolates this one mechanism.
    # -----------------------------------------------------------------
    print("\n=== Stage 1: toxicity signal (bucket size / mechanism) ===")
    stage1_candidates = {
        "vpin_fixed_250": dict(toxicity_mode="vpin_fixed", vpin_bucket_notional=250.0),
        "vpin_fixed_500_existing": dict(toxicity_mode="vpin_fixed", vpin_bucket_notional=500.0),
        "vpin_fixed_1000": dict(toxicity_mode="vpin_fixed", vpin_bucket_notional=1000.0),
        "vpin_fixed_2500": dict(toxicity_mode="vpin_fixed", vpin_bucket_notional=2500.0),
        "vpin_fixed_5000": dict(toxicity_mode="vpin_fixed", vpin_bucket_notional=5000.0),
        "vpin_dynamic_5": dict(toxicity_mode="vpin_dynamic", vpin_bucket_trade_target=5.0),
        "vpin_dynamic_10": dict(toxicity_mode="vpin_dynamic", vpin_bucket_trade_target=10.0),
        "vpin_dynamic_20": dict(toxicity_mode="vpin_dynamic", vpin_bucket_trade_target=20.0),
        "order_imbalance_10": dict(toxicity_mode="order_imbalance", order_imbalance_window_trades=10),
        "order_imbalance_20": dict(toxicity_mode="order_imbalance", order_imbalance_window_trades=20),
        "order_imbalance_40": dict(toxicity_mode="order_imbalance", order_imbalance_window_trades=40),
        "toxicity_none": dict(toxicity_mode="none"),
    }
    winner1, config1, sweeps["stage1_toxicity"] = select_best_config(tapes, train_cids, stage1_candidates)
    print(f"  -> Stage 1 winner: {winner1}")

    # -----------------------------------------------------------------
    # Stage 2: inventory skew shape + limit (sensitivity sweep over
    # max_inventory, skew_strength, and skew_mode; target_inventory_notional
    # left at 0 -- flat is the natural target for a two-sided quoting desk).
    # -----------------------------------------------------------------
    print("\n=== Stage 2: inventory skew (mode, limit, strength, half-life) ===")
    stage2_candidates = {}
    for limit in [10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0]:
        stage2_candidates[f"linear_limit{int(limit)}"] = dict(config1, skew_mode="linear", inventory_limit_notional=limit)
    for limit in [10.0, 20.0, 30.0, 50.0, 75.0, 100.0]:
        stage2_candidates[f"sigmoid_s6_limit{int(limit)}"] = dict(
            config1, skew_mode="sigmoid", inventory_limit_notional=limit, skew_strength=6.0)
    winner2, config2, sweeps["stage2_skew"] = select_best_config(tapes, train_cids, stage2_candidates)
    print(f"  -> Stage 2 winner: {winner2}")

    # Half-life decay, tested at the Stage-2-winning limit/mode on top.
    print("\n=== Stage 2b: inventory half-life decay (does mean-reversion between fills help?) ===")
    stage2b_candidates = {
        "no_decay_existing": dict(config2, inventory_half_life_seconds=None),
        "half_life_60": dict(config2, inventory_half_life_seconds=60.0),
        "half_life_120": dict(config2, inventory_half_life_seconds=120.0),
        "half_life_600": dict(config2, inventory_half_life_seconds=600.0),
    }
    winner2b, config2b, sweeps["stage2b_half_life"] = select_best_config(tapes, train_cids, stage2b_candidates)
    print(f"  -> Stage 2b winner: {winner2b}")

    # -----------------------------------------------------------------
    # Stage 3: volatility-scaled position cap.
    # -----------------------------------------------------------------
    print("\n=== Stage 3: volatility-scaled position cap ===")
    stage3_candidates = {
        "vol_cap_off_existing": dict(config2b, use_volatility_cap=False),
        "vol_cap_sens0.5": dict(config2b, use_volatility_cap=True, vol_cap_sensitivity=0.5),
        "vol_cap_sens1.0": dict(config2b, use_volatility_cap=True, vol_cap_sensitivity=1.0),
        "vol_cap_sens2.0": dict(config2b, use_volatility_cap=True, vol_cap_sensitivity=2.0),
    }
    winner3, config3, sweeps["stage3_vol_cap"] = select_best_config(tapes, train_cids, stage3_candidates)
    print(f"  -> Stage 3 winner: {winner3}")

    # -----------------------------------------------------------------
    # Stage 4: post-fill cooldown.
    # -----------------------------------------------------------------
    print("\n=== Stage 4: post-fill cooldown ===")
    stage4_candidates = {
        "cooldown_off_existing": dict(config3, enable_cooldown=False),
        "cooldown_default": dict(config3, enable_cooldown=True),
        "cooldown_strict": dict(config3, enable_cooldown=True, toxic_adverse_spread_multiple=1.0,
                                 cooldown_max_spread_boost=2.0, cooldown_max_size_cut=0.8),
    }
    winner4, config_final, sweeps["stage4_cooldown"] = select_best_config(tapes, train_cids, stage4_candidates)
    print(f"  -> Stage 4 winner: {winner4}")

    print(f"\n=== Final TRAIN-selected config ===\n{json.dumps(config_final, indent=2)}")

    # -----------------------------------------------------------------
    # Whole-population comparison table, Section-9 format, for direct
    # comparison against the existing -$112,335.07 / -$94,667.06 numbers.
    # -----------------------------------------------------------------
    print("\n=== Whole-population comparison (Section 9 format) ===")
    whole_pop_table = {}
    reference_configs = {
        "flat_baseline": None,  # handled specially: base.market_pnl, not market_pnl_v3
        "advanced_existing": dict(toxicity_mode="vpin_fixed", vpin_bucket_notional=500.0, skew_mode="linear",
                                   inventory_limit_notional=100.0, use_volatility_cap=False, enable_cooldown=False),
        "v3_stage1_toxicity_only": config1,
        "v3_stage2_plus_skew": config2b,
        "v3_stage3_plus_volcap": config3,
        "v3_final_combined": config_final,
    }
    for name, config in reference_configs.items():
        if name == "flat_baseline":
            best = markout = 0.0
            n_active = 0
            for cid, (st, tv, m) in tapes.items():
                r = base.market_pnl(st, tv, base.BASE_HALF_SPREAD, base.BASE_FILL_SHARE)
                best += r["pnl_best_case"]
                markout += r["pnl_with_markout_time"]
                if r["n_captured"] > 0:
                    n_active += 1
        else:
            per_market = per_market_stats_for_config(tapes, config)
            best = sum(r["pnl"] for r in per_market)
            markout = sum(r["pnl_with_markout_time"] for r in per_market)
            n_active = len(per_market)
        gap = (1 - markout / best) * 100 if best else None
        whole_pop_table[name] = {"n_active": n_active, "best_case": round(best, 2), "markout": round(markout, 2),
                                  "gap_pct": round(gap, 1) if gap is not None else None}
        gap_str = f"{gap:.1f}%" if gap is not None else "n/a"
        print(f"  {name:28s} n_active={n_active:<5} best_case=${best:>12,.2f}  markout=${markout:>12,.2f}  gap={gap_str}")

    # -----------------------------------------------------------------
    # Full walk-forward pipeline with the final v3 config: does layering
    # the existing market-selection filter search on top flip TEST positive?
    # -----------------------------------------------------------------
    print("\n=== Walk-forward validation of the final v3 config (TRAIN-select filter, evaluate on TEST) ===")
    per_market_final = per_market_stats_for_config(tapes, config_final)
    train_final = [r for r in per_market_final if r["condition_id"] in train_cids]
    test_final = [r for r in per_market_final if r["condition_id"] in test_cids]

    unfiltered_test = unfiltered_summary(test_final)
    print(f"Unfiltered TEST (v3 config, no market-selection filter): n={unfiltered_test['n_markets']}  "
          f"total_markout=${unfiltered_test['total_pnl_with_markout_time']:,.2f}  "
          f"median=${unfiltered_test['median_pnl_with_markout_time']}  %positive={unfiltered_test['pct_positive']}%")

    filter_sweep = []
    for min_markets in wf.MIN_MARKETS_SWEEP:
        res = wf.run_at_min_markets(train_final, test_final, min_markets)
        filter_sweep.append(res)
        if res["winner"] is None:
            print(f"\nmin_markets={min_markets}: no combo met the threshold on TRAIN.")
            continue
        w, tr = res["winner"], res["test_result"]
        lo, hi = w["pace_range"]
        print(f"\nmin_markets={min_markets}:")
        print(f"  TRAIN-selected filter: pace=[{lo:.1f}s,{hi:.1f}s)  volume>=${w['volume_min']:,.0f}  "
              f"history>={w['history_min_s']/3600:.1f}h  (TRAIN n={w['result']['n_markets']}, "
              f"TRAIN median=${w['result']['median_pnl_with_markout_time']})")
        print(f"  OUT-OF-SAMPLE TEST: n={tr['n_markets']}  total_markout=${tr['total_pnl_with_markout_time']:,.2f}  "
              f"median=${tr['median_pnl_with_markout_time']}  %positive={tr['pct_positive']}  "
              f"P(profitable)={res['bootstrap_ci'].get('pct_iterations_positive')}%")

    # -----------------------------------------------------------------
    # Does the optimal (half_spread, fill_share) grid point change under
    # the final v3 config, evaluated on TRAIN only?
    # -----------------------------------------------------------------
    print("\n=== 3x3 half-spread x fill-share grid under the final v3 config (TRAIN only) ===")
    grid_results = []
    for hs in base.HALF_SPREADS:
        for fs in base.FILL_SHARES:
            total = 0.0
            n_active = 0
            for cid, (st, tv, m) in tapes.items():
                if cid not in train_cids:
                    continue
                r = v3.market_pnl_v3(st, tv, hs, fs, **config_final)
                total += r["pnl_with_markout_time"]
                if r["n_captured"] > 0:
                    n_active += 1
            grid_results.append({"half_spread": hs, "fill_share": fs, "train_total_markout": round(total, 2), "n_active": n_active})
            print(f"  half_spread=${hs:<6} fill_share={fs:<6.0%}  TRAIN total_markout=${total:>12,.2f}  n_active={n_active}")
    best_grid = max(grid_results, key=lambda g: g["train_total_markout"])
    print(f"  -> best grid point under v3: half_spread=${best_grid['half_spread']} fill_share={best_grid['fill_share']:.0%} "
          f"(existing base config: half_spread=${base.BASE_HALF_SPREAD} fill_share={base.BASE_FILL_SHARE:.0%})")

    out = {
        "n_train": len(train_cids), "n_test": len(test_cids),
        "stage_sweeps": sweeps,
        "final_config": config_final,
        "whole_population_comparison": whole_pop_table,
        "walkforward_unfiltered_test": unfiltered_test,
        "walkforward_min_markets_sweep": [
            {
                "min_markets": res["min_markets"],
                "selected_filter": None if res["winner"] is None else {
                    "pace_range_seconds": list(res["winner"]["pace_range"]),
                    "volume_min": res["winner"]["volume_min"],
                    "history_min_seconds": res["winner"]["history_min_s"],
                    "train_result": res["winner"]["result"],
                },
                "test_result": res.get("test_result"),
                "bootstrap_ci": res.get("bootstrap_ci"),
                "drawdown": res.get("drawdown"),
            }
            for res in filter_sweep
        ],
        "half_spread_fill_share_grid_train": grid_results,
        "best_grid_point": best_grid,
    }
    out_path = RESULTS_DIR / "mm_proxy_v3_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
