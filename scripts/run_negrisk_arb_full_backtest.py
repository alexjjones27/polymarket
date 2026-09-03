"""Backtest: NegRisk basket arbitrage over the FULL qualifying population,
plus an execution-risk stress test and a floored-leg robustness check.

Extends run_negrisk_arb_backtest.py's methodology (see that file's
docstring for the core mechanism -- buying every leg of a complete,
mutually-exclusive-and-exhaustive NegRisk basket for less than $1 locks a
riskless $1 redemption -- and its data-reality caveats, all still true
here) in three ways, because a positive result on 20 hand-picked events is
not evidence of a durable edge until each of these has been checked:

1. Full population, not a volume-ranked sample. The Gamma /events endpoint
   caps pagination at offset 2000 (GAMMA_OFFSET_CAP, confirmed live), so
   "full population" here means "every complete NegRisk event among the
   2000 highest-volume closed events" -- a real, disclosed ceiling, not an
   unlimited census like the Final-1% market population. 168 events
   qualify (3-120 legs, an explicit `negRiskOther` catch-all leg present).
   The original backtest tested only the top 20 by volume (74% of this
   population's total volume, but 12% of its event count); this runs all
   168, so the headline rate is no longer "how often did the biggest
   baskets show a gap" but "how often does ANY complete basket".

2. Floored-leg robustness check. The base script prices any leg with zero
   recorded trades at a $0.001 floor rather than dropping the event --
   reasonable for one long-tail leg in a large field, but for a basket
   where MOST legs never traded, that floor is doing most of the work of
   manufacturing an apparent gap (60 untraded legs at $0.001 each already
   sums to $0.06 of "headroom" before any real price is even considered).
   This tracks each event's floored-leg fraction and reports the arb rate
   and profit split by whether >25% of an event's legs were floored --
   the same kind of artifact check that flipped the politics-Kelly
   backtest from apparently profitable to not.

3. Execution-risk stress test. The base backtest assumes all N legs fill
   simultaneously at the same historical snapshot price. A real N-leg
   basket buy crosses N separate order books one at a time, and each fill
   can move the next -- this is the single biggest unmodeled risk flagged
   in the original script. This re-evaluates every event's price grid
   (already fetched, no extra network calls) under added per-leg slippage
   of 0 / 50 / 150 bps of each leg's own price, to find whether the edge
   is a real structural gap or a thin one that a realistic multi-leg fill
   erases.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import polymarket_final_pct as pmf
from run_negrisk_arb_backtest import N_CONSECUTIVE, FIDELITY_MIN, extract_legs, fetch_leg_prices

REPO = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO / "results" / "polymarket_final_pct"
CACHE_DIR = REPO / "data" / "raw" / "polymarket" / "negrisk_events"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

START_BANKROLL = 10000.0
STAKE_FRAC = 0.15
MAX_LEGS = 120
SLIPPAGE_BPS_SCENARIOS = [0, 50, 150]
FLOORED_FRAC_ARTIFACT_THRESHOLD = 0.25


def fetch_all_qualifying_negrisk_events():
    """Every complete NegRisk event among the 2000 highest-volume closed
    events -- no top-N truncation, unlike fetch_top_negrisk_events."""
    cache_path = CACHE_DIR / "all_qualifying_complete_negrisk_events.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    print("[negrisk-full] paging /events by volume desc (capped at offset 2000) ...")
    all_events = []
    for offset in range(0, pmf.GAMMA_OFFSET_CAP, 100):
        page = pmf._get(pmf.GAMMA_BASE, "/events", {
            "closed": "true", "order": "volume", "ascending": "false", "limit": 100, "offset": offset,
        })
        if not page:
            break
        all_events.extend(page)

    qualifying = []
    for e in all_events:
        if not e.get("negRisk"):
            continue
        markets = e.get("markets", [])
        if not (3 <= len(markets) <= MAX_LEGS):
            continue
        if not any(m.get("negRiskOther") for m in markets):
            continue
        qualifying.append(e)

    qualifying.sort(key=lambda e: -(e.get("volume") or 0))
    print(f"[negrisk-full] {len(all_events)} events scanned, {len(qualifying)} complete NegRisk baskets qualify")
    cache_path.write_text(json.dumps(qualifying))
    return qualifying


def find_arb_entry_multi(event, legs):
    """Like run_negrisk_arb_backtest.find_arb_entry, but fetches the price
    grid once and evaluates every slippage scenario off the same data
    (no repeated network calls), and reports the floored-leg fraction so
    the artifact check in the module docstring can act on it."""
    scan_start = max(l["created_at"] for l in legs)
    scan_end = max(l["closed_time"] for l in legs)
    if scan_end <= scan_start:
        return None
    start_s, end_s = int(scan_start.timestamp()), int(scan_end.timestamp())

    from concurrent.futures import ThreadPoolExecutor, as_completed
    series = {}
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(fetch_leg_prices, l["token_id"], start_s, end_s): l["token_id"] for l in legs}
        for fut in as_completed(futures):
            tok = futures[fut]
            try:
                series[tok] = fut.result()
            except Exception:
                series[tok] = pmf.pd.DataFrame(columns=["t", "p"])

    frames = []
    n_floored = 0
    for l in legs:
        df = series.get(l["token_id"])
        if df is None or df.empty:
            s = pmf.pd.Series([0.001, 0.001], index=[start_s, end_s], name=l["token_id"])
            n_floored += 1
        else:
            s = df.set_index("t")["p"].rename(l["token_id"])
        frames.append(s)
    combined = pmf.pd.concat(frames, axis=1).sort_index().ffill()
    combined = combined.dropna(how="any")
    floored_frac = n_floored / len(legs)
    print(f"    [negrisk-full] {len(legs)} legs, {n_floored} floored ({floored_frac*100:.0f}%), "
          f"{len(combined)} usable grid points", flush=True)
    if combined.empty:
        return None

    fee_by_tok = {l["token_id"]: l["fee_rate"] for l in legs}

    scenario_results = {}
    for bps in SLIPPAGE_BPS_SCENARIOS:
        cost = combined.copy()
        for tok in cost.columns:
            fr = fee_by_tok[tok]
            slip = bps / 10000.0
            cost[tok] = combined[tok] * (1 + slip) + fr * combined[tok] * (1 - combined[tok])
        total_cost = cost.sum(axis=1)
        profit_frac = 1.0 - total_cost

        qualifies = profit_frac > 0
        run = 0
        hit_idx = None
        idx_list = qualifies.index.tolist()
        for i, ok in enumerate(qualifies.tolist()):
            run = run + 1 if ok else 0
            if run >= N_CONSECUTIVE:
                hit_idx = idx_list[i - N_CONSECUTIVE + 1]
                break

        if hit_idx is None:
            scenario_results[bps] = {"arb_found": False, "max_profit_frac": float(profit_frac.max())}
        else:
            scenario_results[bps] = {
                "arb_found": True,
                "entry_time": pmf.pd.Timestamp(hit_idx, unit="s", tz="UTC").isoformat(),
                "basket_sum": float(combined.loc[hit_idx].sum()),
                "profit_frac": float(profit_frac.loc[hit_idx]),
                "max_profit_frac": float(profit_frac.max()),
            }

    return {
        "event_id": event["id"], "title": event["title"], "n_legs": len(legs),
        "n_floored": n_floored, "floored_frac": round(floored_frac, 4),
        "scenarios": scenario_results,
    }


def simulate_equity(events_results, bps, exclude_floored_artifacts=False):
    trades = []
    for r in events_results:
        if exclude_floored_artifacts and r["floored_frac"] > FLOORED_FRAC_ARTIFACT_THRESHOLD:
            continue
        sc = r["scenarios"].get(bps) or r["scenarios"].get(str(bps))
        if sc and sc["arb_found"]:
            trades.append({"event_id": r["event_id"], "title": r["title"], "n_legs": r["n_legs"],
                            "floored_frac": r["floored_frac"], **sc})
    trades.sort(key=lambda t: t["entry_time"])

    gas_cfg = pmf.GasAssumptions(relayer_sponsored=False)
    gas_cost = gas_cfg.cost_usd()

    equity = START_BANKROLL
    curve = [(trades[0]["entry_time"][:10] if trades else None, equity)]
    for t in trades:
        stake = STAKE_FRAC * equity
        pnl = stake * t["profit_frac"] - gas_cost
        equity += pnl
        t["stake"] = round(stake, 2)
        t["pnl"] = round(pnl, 2)
        t["equity_after"] = round(equity, 2)
        curve.append((t["entry_time"][:10], round(equity, 2)))

    return {
        "n_trades": len(trades),
        "final_equity": round(equity, 2),
        "total_return_pct": round((equity / START_BANKROLL - 1) * 100, 2),
        "mean_profit_frac_pct": round(sum(t["profit_frac"] for t in trades) / len(trades) * 100, 3) if trades else None,
        "median_profit_frac_pct": round(sorted(t["profit_frac"] for t in trades)[len(trades)//2] * 100, 3) if trades else None,
        "gas_cost_per_trade": round(gas_cost, 4),
        "trades": trades,
        "equity_curve": curve,
    }


def main():
    events = fetch_all_qualifying_negrisk_events()
    print(f"[negrisk-full] backtesting all {len(events)} qualifying events ...")

    results = []
    for i, event in enumerate(events):
        legs = extract_legs(event)
        if len(legs) < 3:
            continue
        print(f"[negrisk-full] ({i+1}/{len(events)}) {event['title'][:55]!r} -- {len(legs)} legs ...", flush=True)
        r = find_arb_entry_multi(event, legs)
        if r is None:
            continue
        r["end_date"] = event.get("endDate")
        r["volume"] = event.get("volume")
        results.append(r)

    out = {"n_events_scanned": len(results), "n_legs_total": sum(r["n_legs"] for r in results)}

    for bps in SLIPPAGE_BPS_SCENARIOS:
        n_arb = sum(1 for r in results if r["scenarios"][bps]["arb_found"])
        sim_all = simulate_equity(results, bps, exclude_floored_artifacts=False)
        sim_clean = simulate_equity(results, bps, exclude_floored_artifacts=True)
        out[f"slippage_{bps}bps"] = {
            "n_events_with_arb": n_arb,
            "pct_events_with_arb": round(n_arb / len(results) * 100, 1) if results else None,
            "all_events": sim_all,
            "excl_floored_artifact_events": sim_clean,
        }
        print(f"\n=== slippage {bps}bps: {n_arb}/{len(results)} events had a qualifying gap "
              f"({out[f'slippage_{bps}bps']['pct_events_with_arb']}%) ===")
        print(f"  all events:      ${START_BANKROLL:,.0f} -> ${sim_all['final_equity']:,.2f} "
              f"({sim_all['total_return_pct']:+.2f}%, n={sim_all['n_trades']})")
        print(f"  excl >25% floored: ${START_BANKROLL:,.0f} -> ${sim_clean['final_equity']:,.2f} "
              f"({sim_clean['total_return_pct']:+.2f}%, n={sim_clean['n_trades']})")

    out["events_raw"] = results

    out_path = RESULTS_DIR / "negrisk_arb_full_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
