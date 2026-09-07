"""Searches the collected order-book snapshots for a predictive signal.

Everything tested so far ran on trade PRINTS, which cannot see the book. These
snapshots carry full 10-level depth on both sides, so they support the classical
microstructure questions we have never been able to ask:

  1. IMBALANCE      does bid_depth/(bid+ask) predict the next move? This is the
                    best-documented microstructure signal in real markets -- when
                    resting buy interest dominates, price tends to rise. If it works
                    here it is an edge that has nothing to do with our failed
                    price-level rules.

  2. DEPTH DECAY    every stop-out showed depth collapsing (4,297 -> 726 shares in
                    one case). We only ever looked at that AFTER the price moved.
                    The real question is whether depth starts thinning BEFORE the
                    price does -- if so it is a genuine early warning, which is the
                    one thing no price-based filter could ever provide.

  3. SPREAD         does a widening spread lead a move, or only accompany it?

Method: for each snapshot, compute features from the book alone, then measure the
realised price change over the following 5/10/30 seconds using only later snapshots
of that same window and side. No lookahead beyond the horizon being measured.

Reported as Spearman correlation (robust to the non-normal distributions here) plus
bucketed means, so a monotone relationship is visible even if the correlation is
weak.
"""
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from scipy import stats as sps

REPO = Path(r"D:\Finance\polymarket")
HORIZONS = [5, 10, 30]


def load():
    series = defaultdict(list)
    d = REPO / "data" / "raw" / "polymarket" / "book_snapshots"
    n = 0
    for f in sorted(d.glob("*.jsonl")):
        for line in f.open():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not r.get("bids") or not r.get("asks"):
                continue
            try:
                bid = float(r["bids"][0][0])
                ask = float(r["asks"][0][0])
                bd = sum(float(x[1]) for x in r["bids"])
                ad = sum(float(x[1]) for x in r["asks"])
                bd1 = float(r["bids"][0][1])
                ad1 = float(r["asks"][0][1])
            except (TypeError, ValueError, IndexError):
                continue
            if bd + ad <= 0:
                continue
            series[(r["window_end"], r["side"])].append({
                "ts": r["ts"], "rel": r["ts"] - r["close_ts"],
                "bid": bid, "ask": ask, "mid": (bid + ask) / 2.0,
                "spread": ask - bid,
                "imb": bd / (bd + ad),          # full-ladder imbalance
                "imb1": bd1 / (bd1 + ad1) if (bd1 + ad1) > 0 else 0.5,  # top-of-book
                "bd": bd, "ad": ad,
            })
            n += 1
    for k in series:
        series[k].sort(key=lambda x: x["ts"])
    return series, n


def build_observations(series):
    """For each snapshot, attach forward mid-change at each horizon, and the
    trailing change in depth (to test whether depth leads price)."""
    obs = []
    for key, rows in series.items():
        for i, r in enumerate(rows):
            o = dict(r)
            # trailing depth change over the previous ~10s
            prior = [x for x in rows[:i] if r["ts"] - x["ts"] <= 10]
            o["bd_chg"] = (r["bd"] / prior[0]["bd"] - 1.0) if prior and prior[0]["bd"] > 0 else 0.0
            o["ad_chg"] = (r["ad"] / prior[0]["ad"] - 1.0) if prior and prior[0]["ad"] > 0 else 0.0
            ok = False
            for h in HORIZONS:
                fut = [x for x in rows[i+1:] if x["ts"] - r["ts"] <= h]
                if fut:
                    o[f"fwd{h}"] = fut[-1]["mid"] - r["mid"]
                    ok = True
                else:
                    o[f"fwd{h}"] = None
            if ok:
                obs.append(o)
    return obs


def corr(obs, feat, h):
    pairs = [(o[feat], o[f"fwd{h}"]) for o in obs if o.get(f"fwd{h}") is not None]
    if len(pairs) < 50:
        return None
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    if len(set(a)) < 3:
        return None
    rho, p = sps.spearmanr(a, b)
    return rho, p, len(pairs)


def buckets(obs, feat, h, edges):
    print(f"\n  {feat} vs forward {h}s mid change:")
    print(f"  {'bucket':>18} {'n':>6} {'mean_fwd':>10} {'median':>9} {'frac_up':>8}")
    for lo, hi in zip(edges, edges[1:]):
        g = [o for o in obs if o.get(f"fwd{h}") is not None and lo <= o[feat] < hi]
        if len(g) < 30:
            continue
        f = [o[f"fwd{h}"] for o in g]
        print(f"  [{lo:>6.2f},{hi:>5.2f}) {len(g):>6} {st.mean(f):>+10.5f} "
              f"{st.median(f):>+9.5f} {sum(1 for x in f if x > 0)/len(f):>7.1%}")


def main():
    series, n = load()
    print(f"loaded {n} snapshots across {len(series)} (window, side) series")
    obs = build_observations(series)
    print(f"built {len(obs)} observations with forward returns\n")
    if len(obs) < 200:
        print("not enough data yet")
        return

    print("=" * 74)
    print("SPEARMAN CORRELATION WITH FORWARD MID CHANGE")
    print("=" * 74)
    print(f"  {'feature':>10} {'horizon':>8} {'rho':>9} {'p':>10} {'n':>7}")
    for feat in ["imb", "imb1", "spread", "bd_chg", "ad_chg", "mid"]:
        for h in HORIZONS:
            r = corr(obs, feat, h)
            if r is None:
                continue
            rho, p, nn = r
            flag = "  <-- SIGNIFICANT" if p < 0.01 else ("  <- p<0.05" if p < 0.05 else "")
            print(f"  {feat:>10} {h:>7}s {rho:>+9.4f} {p:>10.2e} {nn:>7}{flag}")

    print("\n" + "=" * 74)
    print("BUCKETED: full-ladder imbalance")
    print("=" * 74)
    buckets(obs, "imb", 10, [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.01])

    print("\n" + "=" * 74)
    print("BUCKETED: trailing bid-depth change (does depth LEAD price?)")
    print("=" * 74)
    buckets(obs, "bd_chg", 10, [-1.0, -0.5, -0.25, -0.05, 0.05, 0.25, 10.0])

    # restrict to the regime we actually traded, where it matters
    high = [o for o in obs if o["mid"] >= 0.90]
    print("\n" + "=" * 74)
    print(f"RESTRICTED TO mid >= 0.90 (the band we trade) -- n={len(high)}")
    print("=" * 74)
    print(f"  {'feature':>10} {'horizon':>8} {'rho':>9} {'p':>10} {'n':>7}")
    for feat in ["imb", "imb1", "spread", "bd_chg"]:
        for h in HORIZONS:
            r = corr(high, feat, h)
            if r is None:
                continue
            rho, p, nn = r
            flag = "  <-- SIGNIFICANT" if p < 0.01 else ("  <- p<0.05" if p < 0.05 else "")
            print(f"  {feat:>10} {h:>7}s {rho:>+9.4f} {p:>10.2e} {nn:>7}{flag}")
    if high:
        buckets(high, "imb", 10, [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.01])


if __name__ == "__main__":
    main()
