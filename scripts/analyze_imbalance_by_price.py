"""Where does the imbalance edge actually live, and does the fee eat it?

The signal is real and monotonic: top-of-book imbalance predicts the next 10s of mid
movement, with a +0.037 spread between top and bottom deciles against a 0.0113
round-trip spread cost. But at mid 0.90-0.99 -- the only band we have ever traded --
the predicted move shrinks to +0.0017 and is swamped by costs. A 0.97 contract has
only 0.03 of room above it, so there is little for the signal to predict.

That points somewhere we have never looked: the middle of the price distribution,
where there is room to move.

The obstacle is the fee, which is 0.07 * p * (1-p) per share and therefore MAXIMAL
at p = 0.50 (0.0175/share) and near zero at the extremes (0.0007 at p = 0.99). So
the fee is largest exactly where the signal is largest. This computes, band by band,
whether the predictable move survives the half-spread plus the fee at that price.

If it does not clear anywhere, the conclusion is clean and worth stating plainly:
the inefficiency is real, and the fee schedule is precisely what makes it
unharvestable -- which is presumably why it persists.
"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

from scipy import stats as sps

REPO = Path(r"D:\Finance\polymarket")
FEE_RATE = 0.07
BANDS = [(0.05, 0.20), (0.20, 0.35), (0.35, 0.50), (0.50, 0.65),
         (0.65, 0.80), (0.80, 0.90), (0.90, 0.95), (0.95, 1.00)]
HORIZON = 10
DECILE = 0.20   # top/bottom 20% of imb1 within each band


def load_obs():
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
                bd1 = float(r["bids"][0][1]); ad1 = float(r["asks"][0][1])
            except (TypeError, ValueError, IndexError):
                continue
            if bd1 + ad1 <= 0:
                continue
            series[(r["window_end"], r["side"])].append({
                "ts": r["ts"], "bid": bid, "ask": ask, "mid": (bid + ask) / 2.0,
                "spread": ask - bid, "imb1": bd1 / (bd1 + ad1),
            })
    for k in series:
        series[k].sort(key=lambda x: x["ts"])
    obs = []
    for rows in series.values():
        for i, r in enumerate(rows):
            fut = [x for x in rows[i+1:] if x["ts"] - r["ts"] <= HORIZON]
            if fut:
                o = dict(r)
                o["fwd"] = fut[-1]["mid"] - r["mid"]
                obs.append(o)
    return obs


def main():
    obs = load_obs()
    print(f"{len(obs)} observations, horizon {HORIZON}s\n")
    print("=" * 96)
    print("BUY-SIDE EDGE BY PRICE BAND  (top imbalance quintile, buying the ask)")
    print("=" * 96)
    print(f"  {'mid band':>13} {'n':>5} {'n_top':>6} {'mean_fwd':>10} {'half_spr':>9} "
          f"{'fee/share':>10} {'NET':>10} {'verdict':>14}")
    best = None
    for lo, hi in BANDS:
        g = [o for o in obs if lo <= o["mid"] < hi]
        if len(g) < 60:
            continue
        g.sort(key=lambda o: o["imb1"])
        k = max(10, int(len(g) * DECILE))
        top = g[-k:]
        m = st.mean(o["fwd"] for o in top)
        px = st.mean(o["ask"] for o in top)
        hs = st.mean(o["spread"] for o in top) / 2.0
        fee = FEE_RATE * px * (1 - px)
        net = m - hs - fee
        v = "PROFITABLE" if net > 0 else ""
        if best is None or net > best[1]:
            best = ((lo, hi), net)
        print(f"  [{lo:.2f},{hi:.2f}) {len(g):>5} {k:>6} {m:>+10.5f} {hs:>9.5f} "
              f"{fee:>10.5f} {net:>+10.5f} {v:>14}")

    print("\n" + "=" * 96)
    print("THE PROBLEM, STATED DIRECTLY: signal size vs fee size, by price")
    print("=" * 96)
    print(f"  {'mid band':>13} {'decile_spread':>15} {'fee at that px':>16} "
          f"{'round_trip_spr':>16} {'signal - costs':>16}")
    for lo, hi in BANDS:
        g = [o for o in obs if lo <= o["mid"] < hi]
        if len(g) < 60:
            continue
        g.sort(key=lambda o: o["imb1"])
        k = max(10, int(len(g) * DECILE))
        bot, top = g[:k], g[-k:]
        ds = st.mean(o["fwd"] for o in top) - st.mean(o["fwd"] for o in bot)
        px = st.mean(o["mid"] for o in g)
        fee = FEE_RATE * px * (1 - px)
        spr = st.mean(o["spread"] for o in g)
        print(f"  [{lo:.2f},{hi:.2f}) {ds:>+15.5f} {fee:>16.5f} {spr:>16.5f} "
              f"{ds - fee - spr:>+16.5f}")

    print("\n  fee = 0.07*p*(1-p): maximal at p=0.50 (0.01750), near zero at the")
    print("  extremes (0.00069 at p=0.99). The signal is largest where the fee is")
    print("  largest, and smallest where the fee is smallest.")


if __name__ == "__main__":
    main()
