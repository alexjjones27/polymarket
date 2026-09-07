"""Can we harvest the edge as a MAKER rather than a taker?

Both edges found so far died the same way. The 0.95+ calibration edge is
1.3-1.7c/share, and taking it costs a 0.5c half-spread plus fee. The imbalance
signal is larger but decays inside 2s, faster than our 155ms+250ms loop can act.
Crossing the spread is what kills both.

A resting order inverts that: it EARNS the spread instead of paying it, and earns a
maker rebate instead of paying 0.07*p*(1-p). At p=0.97 that swing is roughly
0.5c + 0.2c = 0.7c per share, against an available edge of 1.3-1.7c. It is the
difference between a thin loss and a real margin.

I rejected this earlier for a specific reason: a resting bid only fills when someone
sells into it, which is when the market is moving against you, and I could not model
queue position from trade prints. The book snapshots fix exactly that -- they carry
depth at every level over time, so we can watch the queue drain.

Model, deliberately pessimistic where uncertain:
  - we join the BACK of the queue at level L, so every share resting there is ahead
  - the queue advances only when depth at L falls (a fill or a cancel ahead of us --
    either way we move up)
  - we fill once cumulative advancement covers the depth that was ahead of us AND
    the market is actually trading at or below L
  - if the price gaps straight through L we are filled at L, adversely, in full
  - if the price never returns to L we simply do not trade

Outcome is then measured against the window's real resolution. Break-even for a fill
at L is a win rate of L itself, before the rebate.
"""
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from scipy import stats as sps

REPO = Path(r"D:\Finance\polymarket")
sys.path.insert(0, str(REPO / "src"))
import polymarket_final_pct as pmf

OFFSETS = [0.01, 0.02, 0.03, 0.05]   # how far below the prevailing mid we rest
ARM_AT = -80                          # seconds relative to close when we place it
MIN_ARM_MID = 0.90                    # only rest on the favourite side
FEE_RATE = 0.07


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
            except (TypeError, ValueError, IndexError):
                continue
            series[(r["window_end"], r["side"])].append({
                "ts": r["ts"], "rel": r["ts"] - r["close_ts"],
                "bid": bid, "ask": ask, "mid": (bid + ask) / 2.0,
                "bids": [(float(p), float(s)) for p, s in r["bids"]],
                "asks": [(float(p), float(s)) for p, s in r["asks"]],
            })
    for k in series:
        series[k].sort(key=lambda x: x["ts"])
    return series


def depth_at(levels, px, tol=1e-9):
    for p, s in levels:
        if abs(p - px) < 1e-6:
            return s
    return 0.0


def resolve(window_end, side, cache):
    key = window_end
    if key not in cache:
        try:
            res = pmf._get(pmf.GAMMA_BASE, "/events", {"slug": f"btc-updown-5m-{window_end}"})
            m = res[0]["markets"][0]
            pr = pmf._safe_json_list(m.get("outcomePrices"))
            cache[key] = float(pr[0]) == 1.0 if m.get("closed") and len(pr) == 2 else None
        except Exception:
            cache[key] = None
    up = cache[key]
    if up is None:
        return None
    return up if side == "Up" else (not up)


def simulate(rows, offset, resolved):
    """Rest a bid `offset` below the mid prevailing at ARM_AT. Returns fill info."""
    arm = None
    for r in rows:
        if r["rel"] >= ARM_AT:
            arm = r
            break
    if arm is None:
        return None
    level = round(arm["mid"] - offset, 2)
    if level <= 0.02 or level >= 1.0:
        return None

    ahead = depth_at(arm["bids"], level)   # everyone already queued at our price
    advanced = 0.0
    prev = ahead
    for r in rows:
        if r["ts"] <= arm["ts"]:
            continue
        # gapped through us: we are filled in full, adversely
        if r["bid"] < level:
            return {"filled": True, "px": level, "gapped": True, "rel": r["rel"],
                    "won": resolved}
        d = depth_at(r["bids"], level)
        if d < prev:
            advanced += (prev - d)
        prev = d
        # our turn, and the market is trading right at our level
        if advanced >= ahead and abs(r["bid"] - level) < 1e-6:
            return {"filled": True, "px": level, "gapped": False, "rel": r["rel"],
                    "won": resolved}
    return {"filled": False, "px": level}


def main():
    series = load_series()
    cache = {}
    print(f"{len(series)} (window, side) series in the snapshot set")
    print(f"resting a bid at mid-offset, armed at T{ARM_AT}s\n")

    print("=" * 92)
    print("MAKER FILL SIMULATION WITH QUEUE POSITION")
    print("=" * 92)
    print(f"  {'offset':>7} {'armed':>6} {'filled':>7} {'fill%':>6} {'gapped':>7} "
          f"{'avg_px':>7} {'win_rate':>9} {'breakeven':>10} {'edge':>8}")
    for off in OFFSETS:
        results = []
        for (we, side), rows in series.items():
            # Only rest on the FAVOURITE. Resting on both sides indiscriminately
            # puts half the orders on longshots at ~0.45, which is not the strategy
            # and drags the average price to a coin flip.
            arm = next((x for x in rows if x["rel"] >= ARM_AT), None)
            if arm is None or arm["mid"] < MIN_ARM_MID:
                continue
            won = resolve(we, side, cache)
            if won is None:
                continue
            r = simulate(rows, off, won)
            if r:
                results.append(r)
        if not results:
            continue
        fills = [r for r in results if r["filled"]]
        if not fills:
            print(f"  {off:>7.2f} {len(results):>6} {0:>7} {'--':>6}")
            continue
        gapped = sum(1 for r in fills if r["gapped"])
        wins = sum(1 for r in fills if r["won"])
        px = st.mean(r["px"] for r in fills)
        wr = wins / len(fills)
        print(f"  {off:>7.2f} {len(results):>6} {len(fills):>7} "
              f"{len(fills)/len(results):>5.0%} {gapped:>7} {px:>7.4f} {wr:>8.1%} "
              f"{px:>10.4f} {wr-px:>+8.4f}")

    print("\n" + "=" * 92)
    print("WHY IT MATTERS: maker vs taker economics on the SAME edge")
    print("=" * 92)
    p = 0.97
    print(f"  at p={p}:")
    print(f"    taker pays half-spread ~0.0050 + fee {FEE_RATE*p*(1-p):.5f} "
          f"= {0.0050 + FEE_RATE*p*(1-p):.5f} per share")
    print(f"    maker pays neither, and earns a rebate instead")
    print(f"    calibration edge available in the 0.95+ band: 0.0134 to 0.0173")
    print(f"    -> taker keeps {0.0134 - 0.0050 - FEE_RATE*p*(1-p):.5f}, "
          f"maker keeps 0.0134+")

    print("\n  CAVEAT: fills are counted when the queue drains OR the price gaps")
    print("  through us. A drain may be cancellations rather than trades, which")
    print("  would make real fills rarer than modelled. The gapped column is the")
    print("  adverse subset -- the market traded through, so those fills are the")
    print("  ones most likely to lose.")


if __name__ == "__main__":
    main()
