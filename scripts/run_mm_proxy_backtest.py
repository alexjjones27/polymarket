"""Stylized market-making PnL estimate -- NOT a real backtest.

Real market-making can't be rigorously backtested here: Polymarket's `/book`
endpoint 404s on resolved markets (confirmed live, see
polymarket_final_pct.py), so there is no historical order-book depth to
determine what price a resting quote would actually have been filled at, or
how often. This script instead builds an explicitly toy, parameterized
estimate from the one thing that *is* recoverable historically -- the
public trade-print tape (data-api /trades) -- and states its assumptions as
assumptions, not measurements.

Model: pure spread capture. For a fixed assumed half-spread and an assumed
"fill share" (the fraction of each real trade's size a resting maker quote
is assumed to capture), every captured unit earns exactly the half-spread,
full stop -- no inventory-direction term.

Two earlier, more elaborate versions of this model were tried and rejected
before landing here, and it's worth recording why, since both looked
plausible before the numbers gave them away:
  1. Mark inventory to the market's RESOLVED payout (0 or 1). This let a
     handful of "decays to zero" longshot markets dominate the total with
     pure directional P&L that has nothing to do with market-making --
     $133k of a $428k total from one Florida-senator-appointment longshot.
  2. Mark inventory to the NEXT observed trade print instead. This still
     inflated results ($282k), because consecutive raw trade prints
     naturally alternate between hitting the real bid and the real ask
     (bid-ask bounce) even with a constant fair value -- using that bounce
     as a "mark" double-counts a spread that already exists in the prints,
     on top of the model's own assumed half-spread.
Removing the inventory term entirely avoids both failure modes. It also
means this model is a deliberate BEST CASE: it assumes perfect, instant,
costless flattening of every position, i.e. zero adverse selection. Real
market makers lose money precisely when informed flow moves the market
against inventory they haven't unwound yet; that risk is real and this
number does not include it, which is disclosed prominently in the report
rather than papered over with an unreliable price-based correction.

Reuses the SAME market population and (already disk-cached, no new network
calls) trade tapes as the Final-1% backtest, via fetch_market_trades.
"""
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import polymarket_final_pct as pmf

REPO = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO / "results" / "polymarket_final_pct"

START_BANKROLL = 10000.0
MAX_NOTIONAL_PER_TRADE = 25.0  # caps a single captured trade's size, so one whale print can't dominate

HALF_SPREADS = [0.005, 0.01, 0.02]
FILL_SHARES = [0.05, 0.15, 0.30]
BASE_HALF_SPREAD = 0.01
BASE_FILL_SHARE = 0.15


def load_market_meta():
    """cid -> {resolved_outcome_index, resolution_time}, from the existing
    Final-1% trade sample (same population, zero extra fetches)."""
    meta = {}
    with open(RESULTS_DIR / "trades_maker.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cid = r["condition_id"]
            if cid in meta:
                continue
            meta[cid] = {
                "resolved_outcome_index": int(r["resolved_outcome_index"]) if r["resolved_outcome_index"] not in ("", None) else None,
                "resolution_time": r["resolution_time"],
                "question": r["question"],
            }
    return meta


MAX_RELATIVE_SPREAD = 0.3  # a quoted half-spread can't exceed 30% of the distance to 0 or 1


def market_pnl(trades, resolved_idx, half_spread, fill_share):
    """Pure spread capture: every captured unit earns the half_spread, full
    stop. See the module docstring for why the two inventory-marking
    variants tried before this were rejected (both leaked spurious
    directional profit -- one from resolution windfalls on longshots, one
    from double-counting real trade-print bid-ask bounce as a price move).
    resolved_idx is accepted but unused -- kept in the signature so the
    call site doesn't need to change if a sounder inventory model is added
    later.

    A flat, price-independent half_spread is itself unrealistic at the
    extremes this dataset is full of (it's built from markets that crossed
    $0.99+, so the complementary side sits near $0.001-0.01 for most of the
    market's life): a $0.01 absolute spread on a $0.001 token is a 1,000%+
    relative spread, and combined with a fixed-dollar notional cap this
    let captured share counts explode at tiny prices (25/0.001 = 25,000
    shares from one $25 trade) and turned a handful of longshot markets
    into an implied 2,600%+ total return. The effective half-spread used is
    therefore capped at MAX_RELATIVE_SPREAD of the distance to whichever
    boundary (0 or 1) is closer -- still an assumption, but one that keeps
    quoted spreads sane in dollar terms near the extremes instead of
    silently blowing up."""
    total = 0.0
    n_captured = 0
    captured_notional = 0.0
    for t in trades:
        try:
            price = float(t["price"])
            size = float(t["size"])
            side = t["side"]
        except (KeyError, ValueError, TypeError):
            continue
        if price <= 0 or price >= 1 or size <= 0 or side not in ("BUY", "SELL"):
            continue
        eff_half_spread = min(half_spread, MAX_RELATIVE_SPREAD * price, MAX_RELATIVE_SPREAD * (1 - price))
        if eff_half_spread <= 0:
            continue
        shares = min(size * fill_share, MAX_NOTIONAL_PER_TRADE / price)
        if shares <= 0:
            continue
        total += shares * eff_half_spread
        n_captured += 1
        captured_notional += shares * price
    return total, n_captured, captured_notional


CONCENTRATION_TOP_NS = [1, 5, 10, 20]


def concentration_by_top_n(per_market: list[dict]) -> dict:
    """What fraction of total base-case PnL comes from the top N markets by
    PnL, for N in CONCENTRATION_TOP_NS. This model's PnL is not diversified
    spread capture spread evenly across thousands of markets -- a handful of
    high-trade-count, near-extreme-price markets can dominate the total (see
    module docstring's own worked example, "$133k of a $428k total from one
    Florida-senator-appointment longshot," for the earlier rejected version
    of this model -- the current version's own top contributor is smaller in
    relative terms but the same shape of concentration). Previously only
    discoverable by manually sorting per_market_base_case; surfaced here as
    a first-class number instead."""
    ranked = sorted(per_market, key=lambda r: -r["pnl"])
    total_pnl = sum(r["pnl"] for r in ranked)
    out = {"total_pnl": round(total_pnl, 2), "by_top_n": []}
    for n in CONCENTRATION_TOP_NS:
        top_pnl = sum(r["pnl"] for r in ranked[:n])
        out["by_top_n"].append({
            "n": n,
            "pnl": round(top_pnl, 2),
            "pct_of_total": round(top_pnl / total_pnl * 100, 2) if total_pnl else None,
            "top_markets": [
                {"question": r["question"], "pnl": r["pnl"], "n_captured_trades": r["n_captured_trades"]}
                for r in ranked[:n]
            ] if n <= 5 else None,  # only list markets by name for the smaller cuts
        })
    return out


def main():
    meta = load_market_meta()
    print(f"[mm-proxy] {len(meta)} markets in the reused Final-1% population, "
          f"all with cached trade tapes on disk already")

    sensitivity = {}
    for hs in HALF_SPREADS:
        for fs in FILL_SHARES:
            sensitivity[f"hs{hs}_fs{fs}"] = {"half_spread": hs, "fill_share": fs, "total_pnl": 0.0, "n_markets_active": 0}

    per_market_base = []  # for the base-case equity curve
    n_no_resolved_idx = 0
    n_no_trades = 0
    for i, (cid, m) in enumerate(meta.items()):
        if m["resolved_outcome_index"] is None:
            n_no_resolved_idx += 1
            continue
        trades = pmf.fetch_market_trades(cid)
        if not trades:
            n_no_trades += 1
            continue

        for key, cfg in sensitivity.items():
            pnl, n_cap, notional = market_pnl(trades, m["resolved_outcome_index"], cfg["half_spread"], cfg["fill_share"])
            cfg["total_pnl"] += pnl
            if n_cap > 0:
                cfg["n_markets_active"] += 1

        pnl_base, n_cap_base, notional_base = market_pnl(trades, m["resolved_outcome_index"], BASE_HALF_SPREAD, BASE_FILL_SHARE)
        if n_cap_base > 0:
            per_market_base.append({
                "condition_id": cid, "question": m["question"][:80],
                "resolution_time": m["resolution_time"], "pnl": round(pnl_base, 4),
                "n_captured_trades": n_cap_base, "captured_notional": round(notional_base, 2),
            })
        if (i + 1) % 500 == 0:
            print(f"  [mm-proxy] {i+1}/{len(meta)} markets processed ...", flush=True)

    concentration_report = concentration_by_top_n(per_market_base)

    per_market_base.sort(key=lambda r: r["resolution_time"])
    equity = START_BANKROLL
    curve = [(per_market_base[0]["resolution_time"][:10] if per_market_base else None, equity)]
    for r in per_market_base:
        equity += r["pnl"]
        curve.append((r["resolution_time"][:10], round(equity, 2)))

    base_key = f"hs{BASE_HALF_SPREAD}_fs{BASE_FILL_SHARE}"
    base = sensitivity[base_key]

    summary = {
        "n_markets_total": len(meta),
        "n_markets_no_resolution": n_no_resolved_idx,
        "n_markets_no_trades": n_no_trades,
        "n_markets_with_captured_flow_base_case": base["n_markets_active"],
        "start_bankroll": START_BANKROLL,
        "max_notional_per_trade": MAX_NOTIONAL_PER_TRADE,
        "base_case": {"half_spread": BASE_HALF_SPREAD, "fill_share": BASE_FILL_SHARE},
        "final_equity_base_case": round(equity, 2),
        "total_pnl_base_case": round(base["total_pnl"], 2),
        "total_return_pct_base_case": round(base["total_pnl"] / START_BANKROLL * 100, 2),
        "sensitivity_grid": list(sensitivity.values()),
        "concentration_base_case": concentration_report,
        "equity_curve_base_case": curve,
        "per_market_base_case": per_market_base,
    }

    print(f"\n=== Stylized MM proxy, base case (half_spread=${BASE_HALF_SPREAD}, fill_share={BASE_FILL_SHARE:.0%}) ===")
    print(f"{base['n_markets_active']} of {len(meta)} markets had any captured flow")
    print(f"Cumulative additive PnL: ${base['total_pnl']:,.2f}  ->  ${START_BANKROLL:,.0f} + PnL = ${equity:,.2f} "
          f"({summary['total_return_pct_base_case']:+.2f}%, NOT compounded)")
    print("\nSensitivity grid (total PnL, $):")
    for key, cfg in sensitivity.items():
        print(f"  half_spread=${cfg['half_spread']:<6} fill_share={cfg['fill_share']:<6.0%}  "
              f"total_pnl=${cfg['total_pnl']:>10,.2f}  active_markets={cfg['n_markets_active']}")
    print("\nPnL concentration (this model is NOT diversified spread capture -- check before trusting the headline number):")
    for row in concentration_report["by_top_n"]:
        print(f"  top {row['n']:<3} market(s): ${row['pnl']:>10,.2f}  ({row['pct_of_total']}% of total)")
    if concentration_report["by_top_n"][0]["top_markets"]:
        top1 = concentration_report["by_top_n"][0]["top_markets"][0]
        print(f"  #1 contributor: {top1['question']!r} (${top1['pnl']:,.2f}, {top1['n_captured_trades']} captured trades)")

    out_path = RESULTS_DIR / "mm_proxy_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
