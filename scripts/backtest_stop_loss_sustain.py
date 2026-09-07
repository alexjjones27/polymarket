"""Tests requiring the stop level to be held for N seconds before exiting, rather
than exiting on first touch.

Rationale (user-proposed): the price sometimes wicks below a level and recovers
immediately, so a first-touch stop will sell into noise. This is the same reasoning
that made the ENTRY logic use a sustained crossing -- first-touch entry was tested
and found worse than doing nothing.

But the tradeoff runs the other way here. A sustain requirement:
  - cuts false stops (good)  -- winners that dip and recover no longer trigger
  - delays real exits (bad)  -- and these collapses run 10-43s, so several extra
                                seconds of confirmation can mean a much worse fill

So it is genuinely unclear a priori, and both effects are measured here on the
same two samples used by backtest_stop_loss.py: cost from 575 historical pre-close
entries, benefit from the 11 real live losses.
"""
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(r"D:\Finance\polymarket")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
import backtest_btc_5m_sustained as sust
from backtest_stop_loss import (load_live_loss, prepare, taker_fee, STAKE_USD,
                                LATENCY_S, SLIPPAGE, THRESHOLD, SUSTAIN_S)

EXIT_LEVEL = 0.70
SUSTAIN_OPTIONS = [0, 2, 3, 5, 8]


def simulate_sustained(series, entry_ts, entry_price, won, exit_level, stop_sustain_s):
    """Exit only once price has been continuously below exit_level for
    stop_sustain_s seconds (0 = first touch, the original rule)."""
    shares = STAKE_USD / entry_price
    cost = shares * entry_price + taker_fee(shares, entry_price)

    after = [(ts, p) for ts, p, _ in series if ts > entry_ts]
    breach_start = None
    confirmed_ts = None
    for ts, p in after:
        if p < exit_level:
            if breach_start is None:
                breach_start = ts
            if ts - breach_start >= stop_sustain_s:
                confirmed_ts = ts
                break
        else:
            breach_start = None  # recovered -- reset, this was a wick

    if confirmed_ts is not None:
        fill = next(((ts, p) for ts, p in after if ts >= confirmed_ts + LATENCY_S), None)
        if fill is None:
            fill = next(((ts, p) for ts, p in reversed(after) if ts >= confirmed_ts), None)
        if fill is not None:
            exit_price = max(0.001, fill[1] * (1.0 - SLIPPAGE))
            proceeds = shares * exit_price - taker_fee(shares, exit_price)
            return proceeds - cost, True

    return (shares if won else 0.0) - cost, False


def cost_side():
    print("=== COST: false stops on historical pre-close entries ===")
    for cache, label in [("btc_5m_events.json", "IN-SAMPLE"), ("btc_5m_events_oos.json", "OOS")]:
        prepared = prepare(cache)
        rows = []
        for p in prepared:
            r = sust.find_sustained_crossing(p, THRESHOLD, SUSTAIN_S)
            if r and -90 <= r["secs_after_close"] < 0:
                rows.append((p, r))
        print(f"\n  {label} (n={len(rows)})")
        print(f"  {'sustain':>8} {'stops':>6} {'false':>6} {'false_rate':>11} {'total_pnl':>10} {'$/trade':>9}")
        base = None
        for ss in SUSTAIN_OPTIONS:
            total, stops, false_stops = 0.0, 0, 0
            for p, r in rows:
                entry_ts = p["end_s"] + r["secs_after_close"]
                pnl, stopped = simulate_sustained(p["by_side"][r["side"]], entry_ts,
                                                  r["price"], r["won"], EXIT_LEVEL, ss)
                total += pnl
                if stopped:
                    stops += 1
                    if r["won"]:
                        false_stops += 1
            if base is None:
                base = total
            print(f"  {ss:>8} {stops:>6} {false_stops:>6} {false_stops/len(rows):>10.1%} "
                  f"{total:>+10.2f} {(total-base)/len(rows):>+9.4f}")


def benefit_side():
    rows = list(csv.DictReader(open(REPO / "results" / "btc_5m_live" / "trade_log.csv")))
    losses = [r for r in rows if r.get("resolved_won") == "False"]
    loaded = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(load_live_loss, r): r for r in losses}
        for fut in as_completed(futs):
            r = fut.result()
            if r:
                loaded.append(r)
    print(f"\n=== BENEFIT: the {len(loaded)} real live losses ===")
    print(f"{'sustain':>8} {'saved':>6} {'total_pnl':>10} {'avg_loss':>10} {'vs_first_touch':>15}")
    base = None
    for ss in SUSTAIN_OPTIONS:
        total, saved = 0.0, 0
        for L in loaded:
            pnl, stopped = simulate_sustained(L["series"], L["entry_ts"], L["entry_price"],
                                              False, EXIT_LEVEL, ss)
            total += pnl
            if stopped:
                saved += 1
        if base is None:
            base = total
        print(f"{ss:>8} {saved:>6} {total:>+10.2f} {total/len(loaded):>+10.2f} "
              f"{total-base:>+15.2f}")
    return loaded


def main():
    cost_side()
    benefit_side()


if __name__ == "__main__":
    main()
