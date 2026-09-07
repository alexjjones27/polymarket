"""Is the apparent Up/Down arbitrage real, or a timing artifact?

The sweep found ask(Up)+ask(Down) below 1.00 in 4.34% of paired observations, the
best at 0.7500 -- 25 cents per pair of what would be riskless money, since the two
sides must sum to exactly $1.00 at settlement.

That is almost certainly too good to be true, and there is an obvious reason. The
bot polls the two sides SEQUENTIALLY, roughly 155ms apart, and records a snapshot
only when top-of-book changes. So a "pair" is not simultaneous. If the book reprices
between the two reads, the pair straddles two different market states and the sum
becomes meaningless. A 25c violation is far more likely to be that than a standing
gift.

Decisive test: if these are artifacts, the violations will cluster at LARGE time
gaps between the two observations, and vanish as the gap goes to zero. If some
survive at near-zero gaps with real depth behind them, they are worth pursuing.

Also checks executable size, because an arb on 2 shares is not a business.
"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

REPO = Path(r"D:\Finance\polymarket")
GAPS = [(0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 1e9)]


def main():
    # index snapshots per window/side, keeping full timestamps
    per = defaultdict(lambda: defaultdict(list))
    d = REPO / "data" / "raw" / "polymarket" / "book_snapshots"
    for f in sorted(d.glob("*.jsonl")):
        for line in f.open():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not r.get("asks") or not r.get("bids"):
                continue
            try:
                rec = {
                    "ts": r["ts"],
                    "ask": float(r["asks"][0][0]), "ask_sz": float(r["asks"][0][1]),
                    "bid": float(r["bids"][0][0]), "bid_sz": float(r["bids"][0][1]),
                }
            except (TypeError, ValueError, IndexError):
                continue
            per[r["window_end"]][r["side"]].append(rec)
    for w in per:
        for s in per[w]:
            per[w][s].sort(key=lambda x: x["ts"])

    # for each Up observation, find the nearest-in-time Down observation
    pairs = []
    for w, sides in per.items():
        ups, downs = sides.get("Up"), sides.get("Down")
        if not ups or not downs:
            continue
        for u in ups:
            best = min(downs, key=lambda x: abs(x["ts"] - u["ts"]))
            pairs.append({
                "gap": abs(best["ts"] - u["ts"]),
                "asum": u["ask"] + best["ask"],
                "bsum": u["bid"] + best["bid"],
                "ask_sz": min(u["ask_sz"], best["ask_sz"]),
                "bid_sz": min(u["bid_sz"], best["bid_sz"]),
            })
    print(f"{len(pairs)} Up/Down pairs\n")

    print("=" * 84)
    print("DOES THE 'ARBITRAGE' VANISH AS THE TWO READS GET SIMULTANEOUS?")
    print("=" * 84)
    print(f"  {'gap between reads':>20} {'n':>6} {'mean ask_sum':>13} "
          f"{'% below 1.00':>13} {'min':>8} {'mean bid_sum':>13} {'% above 1.00':>13}")
    for lo, hi in GAPS:
        g = [p for p in pairs if lo <= p["gap"] < hi]
        if len(g) < 20:
            continue
        a = [p["asum"] for p in g]
        b = [p["bsum"] for p in g]
        lab = f"{lo:.2f}-{hi:.2f}s" if hi < 1e8 else f">{lo:.0f}s"
        print(f"  {lab:>20} {len(g):>6} {st.mean(a):>13.4f} "
              f"{sum(1 for x in a if x < 1.0)/len(a):>12.2%} {min(a):>8.4f} "
              f"{st.mean(b):>13.4f} {sum(1 for x in b if x > 1.0)/len(b):>12.2%}")

    print("\n" + "=" * 84)
    print("THE TIGHTEST PAIRS ONLY (reads within 250ms)")
    print("=" * 84)
    tight = [p for p in pairs if p["gap"] < 0.25]
    if tight:
        a = [p["asum"] for p in tight]
        viol = [p for p in tight if p["asum"] < 1.0]
        print(f"  n={len(tight)}   mean ask_sum {st.mean(a):.4f}   min {min(a):.4f}")
        print(f"  violations below 1.00: {len(viol)} = {len(viol)/len(tight):.2%}")
        if viol:
            print(f"\n  {'ask_sum':>9} {'profit/pair':>12} {'min size':>10} "
                  f"{'max $ profit':>13} {'gap_ms':>8}")
            for p in sorted(viol, key=lambda x: x["asum"])[:12]:
                prof = 1.0 - p["asum"]
                sz = p["ask_sz"]
                print(f"  {p['asum']:>9.4f} {prof:>+12.4f} {sz:>10.1f} "
                      f"{prof*sz:>13.2f} {p['gap']*1000:>8.0f}")
            tot = sum((1.0 - p["asum"]) * p["ask_sz"] for p in viol)
            print(f"\n  total theoretical profit across all tight violations: ${tot:.2f}")
            print(f"  over {len(set())if False else 'the collection period'} "
                  f"-- and this ignores fees, which apply to BOTH legs")
    else:
        print("  no pairs within 250ms")

    print("\n" + "=" * 84)
    print("READ THIS BEFORE GETTING EXCITED")
    print("=" * 84)
    print("  A true arb needs both legs filled at the quoted prices. We poll the two")
    print("  sides sequentially, so even a 'tight' pair is two reads, not one atomic")
    print("  view. Executing means two more round trips at 155ms each, by which time")
    print("  a real mispricing of this size is long gone -- the same latency wall")
    print("  that killed the imbalance signal and the maker strategy.")


if __name__ == "__main__":
    main()
