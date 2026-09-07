"""Is the top-of-book imbalance signal actually TRADEABLE?

analyze_microstructure_controlled.py established that imb1 (top-of-book bid share)
survives the price-level control: its sign is consistently positive within every
narrow mid band, reaching rho +0.51 (p=3e-39) in [0.97,0.99). That is a real
statistical relationship, not the mechanical artifact that sank the full-ladder
version.

But a correlation with forward mid change is not money. To act on it we must cross
the spread, and the spread here is consistently 0.010. So the question is purely
one of magnitude: does the predictable move exceed the cost of capturing it?

Measured here:
  - mean forward mid change by imb1 bucket, in absolute price terms
  - the spread that would have to be crossed
  - the edge net of a half-spread each way plus the taker fee
  - the same on the top vs bottom decile of imb1, which is the most favourable
    framing the signal can be given

If the net is negative even in the extreme buckets, the signal is real but not
harvestable, and that is the end of it.
"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

from scipy import stats as sps

REPO = Path(r"D:\Finance\polymarket")
HORIZONS = [5, 10, 30]
FEE_RATE = 0.07


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
            o = dict(r)
            keep = False
            for h in HORIZONS:
                fut = [x for x in rows[i+1:] if x["ts"] - r["ts"] <= h]
                o[f"fwd{h}"] = (fut[-1]["mid"] - r["mid"]) if fut else None
                keep = keep or fut != []
            if keep:
                obs.append(o)
    return obs


def main():
    obs = load_obs()
    print(f"{len(obs)} observations\n")
    print(f"median spread: {st.median([o['spread'] for o in obs]):.4f}  "
          f"mean: {st.mean([o['spread'] for o in obs]):.4f}")
    print("crossing costs a half-spread each way; a round trip costs a full spread\n")

    print("=" * 84)
    print("FORWARD MID CHANGE BY TOP-OF-BOOK IMBALANCE  (all mids)")
    print("=" * 84)
    edges = [0.0, 0.1, 0.25, 0.4, 0.6, 0.75, 0.9, 1.01]
    for h in [5, 10, 30]:
        print(f"\n  horizon {h}s")
        print(f"  {'imb1 bucket':>16} {'n':>6} {'mean_fwd':>10} {'half_spread':>12} "
              f"{'net vs cost':>12}")
        for lo, hi in zip(edges, edges[1:]):
            g = [o for o in obs if o.get(f"fwd{h}") is not None and lo <= o["imb1"] < hi]
            if len(g) < 40:
                continue
            m = st.mean(o[f"fwd{h}"] for o in g)
            hs = st.mean(o["spread"] for o in g) / 2.0
            net = abs(m) - hs
            flag = "  <-- EXCEEDS COST" if net > 0 else ""
            print(f"  [{lo:.2f},{hi:.2f}) {len(g):>6} {m:>+10.5f} {hs:>12.5f} "
                  f"{net:>+12.5f}{flag}")

    print("\n" + "=" * 84)
    print("MOST FAVOURABLE FRAMING: top vs bottom decile of imb1")
    print("=" * 84)
    for h in HORIZONS:
        g = [o for o in obs if o.get(f"fwd{h}") is not None]
        if len(g) < 100:
            continue
        g.sort(key=lambda o: o["imb1"])
        k = max(1, len(g) // 10)
        bot, top = g[:k], g[-k:]
        mb = st.mean(o[f"fwd{h}"] for o in bot)
        mt = st.mean(o[f"fwd{h}"] for o in top)
        spread_cost = st.mean(o["spread"] for o in g)
        try:
            _, p = sps.mannwhitneyu([o[f"fwd{h}"] for o in bot],
                                    [o[f"fwd{h}"] for o in top], alternative="two-sided")
        except Exception:
            p = float("nan")
        print(f"\n  horizon {h}s   (n={k} per decile, Mann-Whitney p={p:.2e})")
        print(f"    bottom decile imb1 (ask-heavy): mean fwd {mb:>+.5f}")
        print(f"    top decile    imb1 (bid-heavy): mean fwd {mt:>+.5f}")
        print(f"    spread between deciles         : {mt-mb:>+.5f}")
        print(f"    round-trip spread cost         : {spread_cost:>+.5f}")
        verdict = "TRADEABLE" if (mt - mb) > spread_cost else "NOT tradeable -- cost exceeds signal"
        print(f"    -> {verdict}")

    print("\n" + "=" * 84)
    print("AND: is a one-way bet on the top decile profitable at these prices?")
    print("=" * 84)
    g = [o for o in obs if o.get("fwd10") is not None and 0.90 <= o["mid"] <= 0.99]
    g.sort(key=lambda o: o["imb1"])
    k = max(1, len(g) // 10)
    top = g[-k:]
    if top:
        m = st.mean(o["fwd10"] for o in top)
        px = st.mean(o["ask"] for o in top)
        fee = FEE_RATE * px * (1 - px)
        hs = st.mean(o["spread"] for o in top) / 2.0
        print(f"  buying the ask on the top imb1 decile at mid 0.90-0.99 (n={len(top)}):")
        print(f"    predicted 10s mid gain {m:>+.5f}")
        print(f"    half-spread paid       {hs:>+.5f}")
        print(f"    taker fee per share    {fee:>+.5f}")
        print(f"    NET                    {m - hs - fee:>+.5f}")
        print(f"    -> {'PROFITABLE' if m - hs - fee > 0 else 'NOT profitable'}")


if __name__ == "__main__":
    main()
