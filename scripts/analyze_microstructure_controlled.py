"""Follow-up to analyze_microstructure.py, which found very large correlations
between book imbalance and forward price change (rho -0.64 at mid>=0.90, 30s).

A rho that big in financial data is nearly always an artifact, and there is an
obvious candidate here. These are bounded [0,1] contracts. Near the boundary the
ask side has only a tick or two of room while the bid side spreads over many, so
imbalance is mechanically tied to the price level -- and forward change is also
mechanically tied to it, since a 0.98 contract can rise at most 0.02 but fall 0.98.
Both variables driven by the same third variable produces exactly this.

Two controls:

  1. WITHIN NARROW MID BANDS. If imbalance still predicts inside a 0.02-wide band
     where the mechanical effect is nearly constant, the signal is real.

  2. AGAINST ACTUAL OUTCOMES. Correlation with forward mid change is not money.
     The decision-relevant question is whether imbalance at OUR entry moment
     separates trades that survived from trades that got stopped out. That is
     tested directly against the live trade log.
"""
import csv
import json
import statistics as st
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

from scipy import stats as sps

REPO = Path(r"D:\Finance\polymarket")
BANDS = [(0.90, 0.93), (0.93, 0.95), (0.95, 0.97), (0.97, 0.99), (0.99, 1.001)]
HORIZON = 10


def load_series():
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
                bid = float(r["bids"][0][0]); ask = float(r["asks"][0][0])
                bd = sum(float(x[1]) for x in r["bids"])
                ad = sum(float(x[1]) for x in r["asks"])
                bd1 = float(r["bids"][0][1]); ad1 = float(r["asks"][0][1])
            except (TypeError, ValueError, IndexError):
                continue
            if bd + ad <= 0 or bd1 + ad1 <= 0:
                continue
            series[(r["window_end"], r["side"])].append({
                "ts": r["ts"], "mid": (bid + ask) / 2.0, "bid": bid,
                "imb": bd / (bd + ad), "imb1": bd1 / (bd1 + ad1),
                "spread": ask - bid,
            })
    for k in series:
        series[k].sort(key=lambda x: x["ts"])
    return series


def main():
    series = load_series()
    obs = []
    for rows in series.values():
        for i, r in enumerate(rows):
            fut = [x for x in rows[i+1:] if x["ts"] - r["ts"] <= HORIZON]
            if fut:
                o = dict(r)
                o["fwd"] = fut[-1]["mid"] - r["mid"]
                obs.append(o)
    print(f"{len(obs)} observations with {HORIZON}s forward returns\n")

    print("=" * 78)
    print("CONTROL 1: does imbalance predict WITHIN a narrow mid band?")
    print("=" * 78)
    print("  if the signal is mechanical, it vanishes once price level is held fixed")
    print(f"\n  {'mid band':>14} {'n':>6} {'imb rho':>9} {'p':>10} {'imb1 rho':>10} {'p':>10}")
    for lo, hi in BANDS:
        g = [o for o in obs if lo <= o["mid"] < hi]
        if len(g) < 80:
            continue
        r1, p1 = sps.spearmanr([o["imb"] for o in g], [o["fwd"] for o in g])
        r2, p2 = sps.spearmanr([o["imb1"] for o in g], [o["fwd"] for o in g])
        f1 = "*" if p1 < 0.01 else " "
        f2 = "*" if p2 < 0.01 else " "
        print(f"  [{lo:.2f},{hi:.2f}) {len(g):>6} {r1:>+9.4f}{f1} {p1:>9.2e} "
              f"{r2:>+10.4f}{f2} {p2:>9.2e}")

    print("\n" + "=" * 78)
    print("CONTROL 2: does imbalance at OUR entry separate survivors from stop-outs?")
    print("=" * 78)
    log = REPO / "results" / "btc_5m_live" / "trade_log.csv"
    trades = [r for r in csv.DictReader(open(log)) if r.get("resolved_won") in ("True", "False")]

    def tt(r):
        return datetime.strptime(r["trade_time"], "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone(timedelta(hours=1))).timestamp()

    matched = []
    for t in trades:
        key = (int(t["window_end"]), t["side"])
        rows = series.get(key)
        if not rows:
            continue
        et = tt(t)
        snap = None
        for x in rows:
            if x["ts"] <= et + 1.0:
                snap = x
            else:
                break
        if snap:
            matched.append({
                "imb": snap["imb"], "imb1": snap["imb1"], "spread": snap["spread"],
                "stopped": t.get("exited") == "True",
                "won": t["resolved_won"] == "True",
            })
    print(f"  matched {len(matched)} live trades to a book snapshot at entry")
    if len(matched) < 20:
        print("  too few matches -- collector has not covered enough live trades yet")
        return
    bad = [m for m in matched if m["stopped"]]
    good = [m for m in matched if not m["stopped"]]
    print(f"  survived: {len(good)}   stopped out: {len(bad)}\n")
    if not bad:
        print("  no stop-outs in the matched set")
        return
    print(f"  {'feature':>10} {'survived':>10} {'stopped':>10} {'diff':>9} {'p':>9}")
    for f in ["imb", "imb1", "spread"]:
        a = [m[f] for m in good]
        b = [m[f] for m in bad]
        try:
            _, p = sps.mannwhitneyu(a, b, alternative="two-sided")
        except Exception:
            p = float("nan")
        flag = "  <--" if p < 0.05 else ""
        print(f"  {f:>10} {st.mean(a):>10.4f} {st.mean(b):>10.4f} "
              f"{st.mean(a)-st.mean(b):>+9.4f} {p:>9.4f}{flag}")

    print(f"\n  every stopped-out trade's imbalance at entry:")
    for m in bad:
        print(f"    imb={m['imb']:.4f}  imb1={m['imb1']:.4f}  spread={m['spread']:.3f}")
    print(f"  survivor imbalance range: {min(m['imb'] for m in good):.4f} "
          f"to {max(m['imb'] for m in good):.4f}")


if __name__ == "__main__":
    main()
