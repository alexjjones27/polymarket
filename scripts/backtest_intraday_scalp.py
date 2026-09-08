"""First-passage backtest: does the target get hit BEFORE the stop?

Stage 2b showed a real, monotone signal -- P(+0.10 move within 60s) rises from
59.28% in the lowest top-of-book-imbalance decile to 70.31% in the highest, while
P(-0.10 move) falls from 68.31% to 58.08%. But BOTH probabilities are high, because
inside these windows the price usually swings 10 cents in each direction at some
point. So "the target gets hit" is nearly meaningless on its own. What decides
profitability is which is reached FIRST.

This walks each entry forward tick by tick and records whichever of target, stop or
time limit arrives first, with the full cost of actually doing it:

  entry   buy at the ASK, pay fee 0.07*p*(1-p)
  exit    sell at the BID, pay the fee again
  so the mid must travel the whole spread plus both fees before a cent is made

Two controls that matter:
  - restricted to mid 0.30-0.70, because a contract at 0.95 mechanically cannot rise
    0.10 and the raw signal was heavily confounded by price level (mid differed
    0.5723 vs 0.3969 between movers and non-movers)
  - split chronologically, since a favourable regime in the back half has flattered
    two previous candidates in this project
"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

from scipy import stats as sps

REPO = Path(r"D:\Finance\polymarket")
FEE_RATE = 0.07
STAKE = 5.0
MID_LO, MID_HI = 0.30, 0.70
IMB_ENTRY = 0.75          # long signal: top-of-book strongly bid-heavy
IMB_ENTRY_SHORT = 0.25    # short signal: strongly ask-heavy (we buy the other side)
GRID = [(0.05, 0.05), (0.10, 0.05), (0.10, 0.10), (0.15, 0.10), (0.20, 0.10)]
MAX_HOLD_S = 120


def fee(p):
    return FEE_RATE * p * (1 - p)


def load():
    series = defaultdict(list)
    d = REPO / "data" / "raw" / "polymarket" / "book_snapshots"
    for f in sorted(d.glob("*.jsonl")):
        for line in f.open():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not r.get("bids") or not r.get("asks"):
                continue
            try:
                b = [(float(p), float(s)) for p, s in r["bids"]]
                a = [(float(p), float(s)) for p, s in r["asks"]]
            except (TypeError, ValueError):
                continue
            if not b or not a:
                continue
            series[(r["window_end"], r["side"])].append({
                "ts": r["ts"], "bid": b[0][0], "ask": a[0][0],
                "mid": (b[0][0] + a[0][0]) / 2.0,
                "imb1": b[0][1] / (b[0][1] + a[0][1]) if (b[0][1] + a[0][1]) else 0.5,
            })
    for k in series:
        series[k].sort(key=lambda x: x["ts"])
    return series


def simulate(rows, i, target, stop):
    """Buy at the ask now; walk forward to whichever of target/stop/timeout is first."""
    e = rows[i]
    entry_ask = e["ask"]
    if not (MID_LO <= e["mid"] <= MID_HI):
        return None
    shares = STAKE / entry_ask
    cost = shares * entry_ask + shares * fee(entry_ask)
    tgt_mid = e["mid"] + target
    stp_mid = e["mid"] - stop
    for j in range(i + 1, len(rows)):
        x = rows[j]
        if x["ts"] - e["ts"] > MAX_HOLD_S:
            break
        if x["mid"] <= stp_mid:
            proceeds = shares * x["bid"] - shares * fee(x["bid"])
            return {"pnl": proceeds - cost, "outcome": "stop", "held": x["ts"] - e["ts"]}
        if x["mid"] >= tgt_mid:
            proceeds = shares * x["bid"] - shares * fee(x["bid"])
            return {"pnl": proceeds - cost, "outcome": "target", "held": x["ts"] - e["ts"]}
    # timeout: mark out at the last bid we saw
    last = None
    for j in range(i + 1, len(rows)):
        if rows[j]["ts"] - e["ts"] > MAX_HOLD_S:
            break
        last = rows[j]
    if last is None:
        return None
    proceeds = shares * last["bid"] - shares * fee(last["bid"])
    return {"pnl": proceeds - cost, "outcome": "timeout", "held": last["ts"] - e["ts"]}


def main():
    series = load()
    print(f"{sum(len(v) for v in series.values())} snapshots, {len(series)} series")
    print(f"entry when imb1 >= {IMB_ENTRY}, mid in [{MID_LO},{MID_HI}], "
          f"max hold {MAX_HOLD_S}s\n")

    # collect entries in time order so the split is chronological
    entries = []
    for key, rows in series.items():
        for i, r in enumerate(rows):
            if r["imb1"] >= IMB_ENTRY and MID_LO <= r["mid"] <= MID_HI:
                entries.append((r["ts"], key, i))
    entries.sort()
    print(f"{len(entries)} candidate entries\n")
    if len(entries) < 100:
        print("too few entries")
        return

    print("=" * 100)
    print("FIRST-PASSAGE RESULTS  (buy the ask, sell the bid, both fees paid)")
    print("=" * 100)
    print(f"  {'target':>7} {'stop':>6} {'sample':>12} {'n':>5} {'hit_tgt':>8} {'hit_stop':>9} "
          f"{'timeout':>8} {'avg_pnl':>9} {'total':>10} {'return':>8}")
    for target, stop in GRID:
        res = []
        for ts, key, i in entries:
            r = simulate(series[key], i, target, stop)
            if r:
                res.append(r)
        if len(res) < 50:
            continue
        half = len(res) // 2
        for lab, seg in (("first half", res[:half]), ("second half", res[half:]),
                         ("FULL", res)):
            n = len(seg)
            tg = sum(1 for x in seg if x["outcome"] == "target") / n
            sp = sum(1 for x in seg if x["outcome"] == "stop") / n
            to = sum(1 for x in seg if x["outcome"] == "timeout") / n
            tot = sum(x["pnl"] for x in seg)
            print(f"  {target:>7.2f} {stop:>6.2f} {lab:>12} {n:>5} {tg:>7.1%} {sp:>8.1%} "
                  f"{to:>7.1%} {tot/n:>+9.4f} {tot:>+10.2f} {tot/(n*STAKE):>+7.2%}")
        print()


if __name__ == "__main__":
    main()
