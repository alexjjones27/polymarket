"""Can we trade the MOVE instead of the settlement?

Every strategy tried so far bought a contract and held it to resolution, which means
carrying binary risk: a 0.97 favourite pays 3c when right and costs 97c when wrong.
That asymmetry, not the signal, is what made the tail so punishing.

Trading the move is a different game. Buy at 0.60, sell at 0.80, and the position is
never exposed to settlement at all:

  gross                     +0.200
  half-spread in and out    -0.010
  fee in   0.07*.6*.4       -0.0168
  fee out  0.07*.8*.2       -0.0112
  net                       +0.162/share  = +27% on a 0.60 entry

So the economics are excellent IF such moves can be anticipated. This asks whether
they can.

Note this is NOT the imbalance test that already failed. That one asked whether
top-of-book imbalance predicts the NEXT TICK, and the answer was yes but with a
2-second half-life, far inside our 155ms+250ms reaction budget. Anticipating a 10-20
cent move over 30-60 seconds is a different question about sustained order flow, and
the earlier result says nothing about it either way.

Three stages, in order, so the thing dies early if it is going to:
  1. do large moves even happen often enough to matter?
  2. does anything observable in the book precede them?
  3. only if both hold, a full backtest with entry, exit, stop and costs.
"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

from scipy import stats as sps

REPO = Path(r"D:\Finance\polymarket")
HORIZONS = [15, 30, 60, 120]
MOVE_SIZES = [0.05, 0.10, 0.20]


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
                bids = [(float(p), float(s)) for p, s in r["bids"]]
                asks = [(float(p), float(s)) for p, s in r["asks"]]
            except (TypeError, ValueError):
                continue
            if not bids or not asks:
                continue
            bd = sum(s for _, s in bids)
            ad = sum(s for _, s in asks)
            if bd + ad <= 0:
                continue
            series[(r["window_end"], r["side"])].append({
                "ts": r["ts"], "rel": r["ts"] - r["close_ts"],
                "bid": bids[0][0], "ask": asks[0][0],
                "mid": (bids[0][0] + asks[0][0]) / 2.0,
                "spread": asks[0][0] - bids[0][0],
                "imb1": bids[0][1] / (bids[0][1] + asks[0][1]),
                "imb": bd / (bd + ad),
                "bd": bd, "ad": ad,
                "bids": bids, "asks": asks,
            })
    for k in series:
        series[k].sort(key=lambda x: x["ts"])
    return series


def main():
    series = load()
    n_snap = sum(len(v) for v in series.values())
    print(f"{n_snap} snapshots across {len(series)} (window, side) series\n")

    # ---------- STAGE 1: do large moves happen? ----------
    print("=" * 92)
    print("STAGE 1: how often does the mid move enough to be worth trading?")
    print("=" * 92)
    print(f"  {'horizon':>8} {'n':>7} " + " ".join(f"{'>=+'+str(m):>10}" for m in MOVE_SIZES)
          + "   (upward moves only; downward are symmetric)")
    obs = []
    for rows in series.values():
        for i, r in enumerate(rows):
            o = dict(r)
            for h in HORIZONS:
                fut = [x for x in rows[i+1:] if x["ts"] - r["ts"] <= h]
                o[f"max{h}"] = max((x["mid"] for x in fut), default=None)
                o[f"min{h}"] = min((x["mid"] for x in fut), default=None)
                o[f"end{h}"] = fut[-1]["mid"] if fut else None
            obs.append(o)
    for h in HORIZONS:
        g = [o for o in obs if o.get(f"max{h}") is not None]
        if not g:
            continue
        counts = []
        for m in MOVE_SIZES:
            c = sum(1 for o in g if o[f"max{h}"] - o["mid"] >= m)
            counts.append(f"{c/len(g):>9.2%}")
        print(f"  {h:>7}s {len(g):>7} " + " ".join(counts))

    # ---------- STAGE 2: does the book foreshadow them? ----------
    print("\n" + "=" * 92)
    print("STAGE 2: does anything in the book precede a large upward move?")
    print("=" * 92)
    print("  comparing book state before a >=+0.10 move against all other moments\n")
    for h in (30, 60):
        g = [o for o in obs if o.get(f"max{h}") is not None and 0.15 < o["mid"] < 0.85]
        if len(g) < 200:
            continue
        movers = [o for o in g if o[f"max{h}"] - o["mid"] >= 0.10]
        rest = [o for o in g if o[f"max{h}"] - o["mid"] < 0.10]
        if len(movers) < 30:
            print(f"  horizon {h}s: only {len(movers)} movers, too few")
            continue
        print(f"  horizon {h}s   movers={len(movers)}  others={len(rest)}")
        print(f"  {'feature':>12} {'before move':>13} {'otherwise':>11} {'diff':>9} {'p':>9}")
        for feat in ("imb1", "imb", "spread", "bd", "ad", "mid"):
            a = [o[feat] for o in movers]
            b = [o[feat] for o in rest]
            try:
                _, p = sps.mannwhitneyu(a, b, alternative="two-sided")
            except Exception:
                p = float("nan")
            flag = "  <--" if p < 0.01 else ""
            print(f"  {feat:>12} {st.mean(a):>13.4f} {st.mean(b):>11.4f} "
                  f"{st.mean(a)-st.mean(b):>+9.4f} {p:>9.4f}{flag}")
        print()

    # ---------- STAGE 2b: is the signal monotone and usable? ----------
    print("=" * 92)
    print("STAGE 2b: P(>=+0.10 move in 60s) by top-of-book imbalance decile")
    print("=" * 92)
    g = [o for o in obs if o.get("max60") is not None and 0.15 < o["mid"] < 0.85]
    if len(g) >= 200:
        g.sort(key=lambda o: o["imb1"])
        k = max(1, len(g) // 10)
        print(f"  {'decile':>7} {'imb1 range':>18} {'n':>6} {'P(+0.10 up)':>13} "
              f"{'P(-0.10 down)':>14}")
        for i in range(10):
            seg = g[i*k:(i+1)*k] if i < 9 else g[9*k:]
            if not seg:
                continue
            up = sum(1 for o in seg if o["max60"] - o["mid"] >= 0.10) / len(seg)
            dn = sum(1 for o in seg if o["mid"] - o["min60"] >= 0.10) / len(seg)
            print(f"  {i+1:>7} {seg[0]['imb1']:>8.3f}-{seg[-1]['imb1']:<9.3f} "
                  f"{len(seg):>6} {up:>12.2%} {dn:>13.2%}")


if __name__ == "__main__":
    main()
