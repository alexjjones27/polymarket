"""Follow-up to backtest_stop_loss.py.

That script measured the cost of false stop-outs across ALL historical entries,
which span [close-90, close+300]. But post-timing-fix the live system enters
almost entirely in [close-90, close+0] -- the pre-close stretch where the outcome
is still genuinely uncertain and a winning position is far more likely to dip
below a stop level and recover. Blending in the post-close entries (where the
outcome is nearly locked and dips are rare) understates the false-stop rate for
the regime we actually trade.

This re-runs the cost side restricted to entry buckets, so the stop-loss decision
is made against the matching regime rather than a flattering average.
"""
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(r"D:\Finance\polymarket")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
import backtest_btc_5m_sustained as sust
from backtest_stop_loss import simulate, prepare, EXIT_LEVELS, THRESHOLD, SUSTAIN_S

BUCKETS = [
    ("pre-close  [-90, 0)", -90, 0),
    ("post-close [0, +120)", 0, 120),
    ("post-close [+120, +300]", 120, 301),
]


def run(prepared, label):
    print(f"\n=== {label} ===")
    for bname, blo, bhi in BUCKETS:
        rows = []
        for p in prepared:
            r = sust.find_sustained_crossing(p, THRESHOLD, SUSTAIN_S)
            if not r:
                continue
            if not (blo <= r["secs_after_close"] < bhi):
                continue
            rows.append((p, r))
        if not rows:
            continue
        wins = sum(1 for _, r in rows if r["won"])
        print(f"\n  {bname}: n={len(rows)}  wins={wins} losses={len(rows)-wins}")
        print(f"  {'exit_at':>8} {'stops':>6} {'false_stops':>12} {'false_rate':>11} "
              f"{'total_pnl':>10} {'vs_base':>9} {'$/trade':>9}")
        baseline = None
        for lvl in EXIT_LEVELS:
            total, stops, false_stops = 0.0, 0, 0
            for p, r in rows:
                entry_ts = p["end_s"] + r["secs_after_close"]
                series = p["by_side"][r["side"]]
                pnl, stopped = simulate(series, entry_ts, r["price"], r["won"], lvl)
                total += pnl
                if stopped:
                    stops += 1
                    if r["won"]:
                        false_stops += 1
            if baseline is None:
                baseline = total
            tag = "none" if lvl is None else f"{lvl:.2f}"
            delta = total - baseline
            print(f"  {tag:>8} {stops:>6} {false_stops:>12} {false_stops/len(rows):>10.1%} "
                  f"{total:>+10.2f} {delta:>+9.2f} {delta/len(rows):>+9.4f}")


def main():
    is_prep = prepare("btc_5m_events.json")
    oos_prep = prepare("btc_5m_events_oos.json")
    run(is_prep, "IN-SAMPLE by entry-timing bucket")
    run(oos_prep, "OOS by entry-timing bucket")


if __name__ == "__main__":
    main()
