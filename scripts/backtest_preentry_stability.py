"""Tests requiring the price to have been HIGH AND STABLE in the seconds immediately
before entry, rather than having just spiked there.

This is a different hypothesis from backtest_window_history.py, which failed. That
one looked at the EARLY window (T-300..T-90) -- far from entry, and it turned out
carried no signal. This looks at the moments immediately before we buy: the concern
is a price that jumps 0.50 -> 0.90 in the last 10-15 seconds and flips straight back,
versus one that has genuinely held high for 30 seconds.

The current rule only asks for 5 seconds above threshold, which by construction
cannot distinguish those two cases.

Two families tested:

  LONGER SUSTAIN   require the threshold to hold for 10/15/20/30s instead of 5.
                   Costs later and higher-priced entries, and some windows will
                   never qualify.

  PRE-ENTRY FLOOR  keep the 5s sustain, but additionally require the price not to
                   have been below some floor at any point in the preceding 30s.
                   This targets the spike case directly while keeping entry timing
                   unchanged, so it should cost far less than a longer sustain.

Target is "price later dips below 0.70" (our stop would fire), which is ~10x more
frequent than an outright loss in print data and is what is actually costing money.
Both in-sample and OOS, because four entry filters have now failed OOS after looking
good in-sample.
"""
import json
import random
import statistics as st
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scipy import stats as sps

REPO = Path(r"D:\Finance\polymarket")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
import backtest_btc_5m_sustained as sust

THRESHOLD = 0.94
STOP_LEVEL = 0.70
SAMPLE_N = 400
STAKE = 5.0
SUSTAINS = [5, 10, 15, 20, 30]
FLOORS = [None, 0.70, 0.80, 0.85, 0.90]
LOOKBACK_S = 30


def fee(shares, p):
    return shares * 0.07 * p * (1 - p)


def prepare(cache):
    events = json.loads((REPO / "data" / "raw" / "polymarket" / cache).read_text())["events"]
    random.seed(42)
    sample = random.sample(events, min(SAMPLE_N, len(events)))
    out = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futs = {pool.submit(sust.prepare_event, e): e for e in sample}
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception:
                continue
            if r:
                out.append(r)
    return out


def find_entry(series, threshold, sustain_s, floor, lookback_s):
    """First confirmed crossing, optionally requiring no dip below `floor` in the
    lookback_s seconds before confirmation."""
    n = len(series)
    i = 0
    while i < n:
        ts, p = series[i][0], series[i][1]
        if p >= threshold:
            deadline = ts + sustain_s
            ok = True
            last = (ts, p)
            j = i + 1
            while j < n and series[j][0] <= deadline:
                if series[j][1] < threshold:
                    ok = False
                    break
                last = series[j][0], series[j][1]
                j += 1
            # require the sustain window to actually contain the full duration
            if ok and last[0] - ts >= sustain_s - 1.5:
                if floor is not None:
                    lo = last[0] - lookback_s
                    prior = [q for t, q, *_ in series if lo <= t <= last[0]]
                    if prior and min(prior) < floor:
                        i = j + 1
                        continue
                return last
            i = j + 1 if not ok else i + 1
            continue
        i += 1
    return None


def evaluate(prepared, threshold, sustain_s, floor):
    rows = []
    for p in prepared:
        for side, series in p["by_side"].items():
            if not series:
                continue
            e = find_entry(series, threshold, sustain_s, floor, LOOKBACK_S)
            if not e:
                continue
            ets, epx = e
            after = [q for t, q, *_ in series if t > ets]
            won = p["up_resolved"] if side == "Up" else (not p["up_resolved"])
            rows.append({"px": epx, "won": won,
                         "stopped": bool(after and min(after) < STOP_LEVEL)})
            break
    return rows


def summarize(rows, n_windows, label):
    if not rows:
        print(f"  {label:>28}   no qualifying entries")
        return
    n = len(rows)
    stops = sum(1 for r in rows if r["stopped"])
    wins = sum(1 for r in rows if r["won"])
    px = st.mean(r["px"] for r in rows)
    pnl = 0.0
    for r in rows:
        sh = STAKE / r["px"]
        pnl += (sh - STAKE - fee(sh, r["px"])) if r["won"] else (-STAKE - fee(sh, r["px"]))
    print(f"  {label:>28} {n:>5} {n/n_windows:>6.0%} {px:>7.4f} {stops/n:>9.1%} "
          f"{wins/n:>8.2%} {pnl:>+9.2f} {pnl/(n*STAKE):>+8.2%}")


def main():
    for cache, lab in [("btc_5m_events.json", "IN-SAMPLE"), ("btc_5m_events_oos.json", "OOS")]:
        prep = prepare(cache)
        print(f"\n{'='*96}\n{lab}  ({len(prep)} windows)\n{'='*96}")
        print(f"  {'rule':>28} {'n':>5} {'fire%':>6} {'avg_px':>7} {'stop_rate':>9} "
              f"{'win_rate':>8} {'pnl':>9} {'return':>8}")
        print("  -- longer sustain --")
        for s in SUSTAINS:
            summarize(evaluate(prep, THRESHOLD, s, None), len(prep), f"sustain {s}s")
        print("  -- 5s sustain + no dip below floor in prior 30s --")
        for fl in FLOORS:
            if fl is None:
                continue
            summarize(evaluate(prep, THRESHOLD, 5, fl), len(prep), f"floor {fl:.2f} (30s)")


if __name__ == "__main__":
    main()
