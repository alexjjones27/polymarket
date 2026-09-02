"""Two follow-up checks raised by external review, run against the SAME
unbiased population as run_mm_walkforward_validation.py:

1. REGIME CHANGE: Polymarket removed the 500-millisecond taker-order delay
   on 2026-02-18 without warning (confirmed via multiple independent live
   sources -- see docs/mm_strategy_client_report.md's revision notes). Before
   that date, market makers could use the delay window to cancel stale
   quotes before an adverse fill landed -- effectively free insurance. After,
   fills are immediate. The walk-forward validation's TEST period
   (2026-04-18 to 2026-09-02) is entirely POST-regime-change, while its
   TRAIN period (2023-09-10 to 2026-04-17) is a BLEND of mostly pre-change
   data with a post-change tail -- so "the TRAIN-selected filter failed on
   TEST" is confounded with "the market microstructure TRAIN was mostly
   fit on no longer exists." This script isolates that: same population,
   split by trade timestamp (not market resolution date) into pre- and
   post-regime-change trades, and reports best-case / 15s-markout for each
   independently.

2. MAKER REBATES: also confirmed live against docs.polymarket.com/programs/
   maker-rebates -- makers pay 0% fees and the exchange redistributes a
   category-specific share of TAKER fees back to makers whose resting
   orders get filled (crypto 20%, sports 15%, everything else 25%,
   geopolitics fee-free). The real payout is a competitive pool split among
   ALL makers active in a market that day, which we cannot observe from
   trade-tape data alone -- so REBATE_RATE_BY_FEE_CATEGORY in
   run_mm_proxy_backtest.py computes an explicit UPPER BOUND (100% of the
   pool, i.e. sole liquidity provider), reported as a ceiling next to the
   existing best-case and markout numbers, never merged into them.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_mm_proxy_backtest as base
import run_mm_walkforward_validation as wf

REPO = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO / "results" / "polymarket_final_pct"

# 2026-02-18 00:00:00 UTC, the confirmed date Polymarket removed the 500ms
# taker delay.
REGIME_CHANGE_EPOCH = 1771372800.0

REPORT_BUCKET_TO_FEE_CATEGORY = {
    "crypto_price": "crypto", "sports": "sports", "politics": "politics", "other": "other",
}


def split_trades_by_regime(sorted_trades: list[dict], cutoff_epoch: float = REGIME_CHANGE_EPOCH) -> tuple[list[dict], list[dict]]:
    """Splits one market's chronologically-sorted trade tape into (pre, post)
    relative to `cutoff_epoch`, by each trade's own timestamp -- not by the
    market's resolution date, since a single market can straddle the regime
    change (trading both before and after it)."""
    pre = [t for t in sorted_trades if t["timestamp"] < cutoff_epoch]
    post = [t for t in sorted_trades if t["timestamp"] >= cutoff_epoch]
    return pre, post


def _accumulate(agg: dict, r: dict) -> None:
    agg["pnl_best_case"] += r["pnl_best_case"]
    agg["pnl_with_markout_time"] += r["pnl_with_markout_time"]
    agg["rebate_upper_bound"] += r["rebate_upper_bound"] or 0.0
    if r["n_captured"] > 0:
        agg["n_markets_active"] += 1


def _new_agg() -> dict:
    return {"pnl_best_case": 0.0, "pnl_with_markout_time": 0.0, "rebate_upper_bound": 0.0, "n_markets_active": 0}


def main():
    meta = wf.load_population_meta()
    print(f"[regime-rebate] {len(meta)} markets in the unbiased population")

    full_agg, pre_agg, post_agg = _new_agg(), _new_agg(), _new_agg()
    n_no_trades = 0
    n_markets_with_pre = 0
    n_markets_with_post = 0
    n_markets_straddling = 0

    for i, (cid, m) in enumerate(meta.items()):
        raw_trades = base.pmf.fetch_market_trades(cid)
        if not raw_trades:
            n_no_trades += 1
            continue
        sorted_trades, total_market_volume = base.parse_and_sort_trades(raw_trades)
        if not sorted_trades:
            n_no_trades += 1
            continue

        fee_category = REPORT_BUCKET_TO_FEE_CATEGORY.get(m["report_bucket"], "other")

        r_full = base.market_pnl(sorted_trades, total_market_volume, base.BASE_HALF_SPREAD, base.BASE_FILL_SHARE,
                                  fee_category=fee_category)
        _accumulate(full_agg, r_full)

        pre, post = split_trades_by_regime(sorted_trades)
        has_pre, has_post = bool(pre), bool(post)
        n_markets_with_pre += has_pre
        n_markets_with_post += has_post
        n_markets_straddling += has_pre and has_post

        if pre:
            r_pre = base.market_pnl(pre, total_market_volume, base.BASE_HALF_SPREAD, base.BASE_FILL_SHARE,
                                     fee_category=fee_category)
            _accumulate(pre_agg, r_pre)
        if post:
            r_post = base.market_pnl(post, total_market_volume, base.BASE_HALF_SPREAD, base.BASE_FILL_SHARE,
                                      fee_category=fee_category)
            _accumulate(post_agg, r_post)

        if (i + 1) % 250 == 0:
            print(f"  [regime-rebate] {i + 1}/{len(meta)} markets processed ...", flush=True)

    print(f"\n{len(meta) - n_no_trades} markets with usable trades ({n_no_trades} had none)")
    print(f"{n_markets_with_pre} markets have trades before the regime change, "
          f"{n_markets_with_post} have trades after, {n_markets_straddling} straddle it")

    def _line(label, agg):
        gap_pct = (
            (1 - agg["pnl_with_markout_time"] / agg["pnl_best_case"]) * 100
            if agg["pnl_best_case"] else None
        )
        gap_str = f"{gap_pct:.1f}%" if gap_pct is not None else "n/a"
        print(f"  {label:<22} n_active={agg['n_markets_active']:<5} "
              f"best_case=${agg['pnl_best_case']:>12,.2f}  15s_markout=${agg['pnl_with_markout_time']:>12,.2f}  "
              f"rebate_upper_bound=${agg['rebate_upper_bound']:>9,.2f}  markout_gap={gap_str}")

    print("\n=== Best case / 15s markout / rebate upper bound, by regime ===")
    _line("ALL TRADES (blended)", full_agg)
    _line("PRE-regime-change", pre_agg)
    _line("POST-regime-change", post_agg)

    print(f"\nRebate upper bound as % of the |markout gap| (best_case - markout), each regime:")
    for label, agg in [("ALL", full_agg), ("PRE", pre_agg), ("POST", post_agg)]:
        gap = agg["pnl_best_case"] - agg["pnl_with_markout_time"]
        if gap:
            pct = agg["rebate_upper_bound"] / abs(gap) * 100
            print(f"  {label}: rebate=${agg['rebate_upper_bound']:,.2f}  |gap|=${abs(gap):,.2f}  rebate/|gap|={pct:.2f}%")
        else:
            print(f"  {label}: gap is zero, ratio undefined")

    out = {
        "regime_change_epoch": REGIME_CHANGE_EPOCH,
        "n_markets_with_pre_regime_trades": n_markets_with_pre,
        "n_markets_with_post_regime_trades": n_markets_with_post,
        "n_markets_straddling": n_markets_straddling,
        "all_trades": full_agg,
        "pre_regime": pre_agg,
        "post_regime": post_agg,
    }
    out_path = RESULTS_DIR / "mm_regime_and_rebate_check.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
