"""Can we MAKE markets here rather than take them?

This is a different question from analyze_maker_queue.py, which tested a directional
resting bid held to settlement. Market making means quoting both sides, earning the
spread, and staying flat -- never carrying direction.

The decisive test is the classic one: realised short-horizon volatility against the
spread. A maker earns at most half the spread per fill, and loses whatever the mid
moves against them before they can unload. If typical movement over the time it
takes to turn a position exceeds the half-spread, informed flow picks you off faster
than uninformed flow pays you, and no amount of skill fixes it.

Reported per time-to-close bucket, because there is one regime worth special
attention. AFTER the close, these markets settle on a BTC price that is already
fixed -- the outcome cannot change, only become known. If volatility collapses there
while a spread persists, that is the one place market making could be near-riskless.
That is the specific hypothesis this is built to check.

Also computes the rebate that would be required to break even, since Polymarket pays
makers a rebate whose size we have never established. If the required rebate is
larger than any plausible schedule, the idea is closed regardless.
"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

REPO = Path(r"D:\Finance\polymarket")
HORIZONS = [1, 2, 5, 10]
BUCKETS = [(-90, -60), (-60, -30), (-30, 0), (0, 60), (60, 180), (180, 300)]


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
                bid = float(r["bids"][0][0]); ask = float(r["asks"][0][0])
                bsz = float(r["bids"][0][1]); asz = float(r["asks"][0][1])
            except (TypeError, ValueError, IndexError):
                continue
            series[(r["window_end"], r["side"])].append({
                "ts": r["ts"], "rel": r["ts"] - r["close_ts"],
                "bid": bid, "ask": ask, "mid": (bid + ask) / 2.0,
                "spread": ask - bid, "bsz": bsz, "asz": asz,
            })
    for k in series:
        series[k].sort(key=lambda x: x["ts"])
    return series


def main():
    series = load()
    obs = []
    for rows in series.values():
        for i, r in enumerate(rows):
            o = dict(r)
            any_f = False
            for h in HORIZONS:
                fut = [x for x in rows[i+1:] if x["ts"] - r["ts"] <= h]
                o[f"d{h}"] = abs(fut[-1]["mid"] - r["mid"]) if fut else None
                o[f"s{h}"] = (fut[-1]["mid"] - r["mid"]) if fut else None
                any_f = any_f or bool(fut)
            if any_f:
                obs.append(o)
    print(f"{len(obs)} observations from {len(series)} (window, side) series\n")

    print("=" * 94)
    print("THE DECISIVE TEST: does the mid move more than the half-spread you earn?")
    print("=" * 94)
    print("  a maker collects at most half the spread; it loses whatever the mid moves")
    print("  against it before the position can be unwound\n")
    print(f"  {'horizon':>8} {'n':>6} {'med |dmid|':>11} {'mean |dmid|':>12} "
          f"{'half_spread':>12} {'P(move>half)':>13} {'verdict':>22}")
    for h in HORIZONS:
        g = [o for o in obs if o.get(f"d{h}") is not None]
        if len(g) < 100:
            continue
        d = [o[f"d{h}"] for o in g]
        hs = st.mean(o["spread"] for o in g) / 2.0
        frac = sum(1 for x in d if x > hs) / len(d)
        v = "picked off" if st.median(d) > hs else "spread covers median move"
        print(f"  {h:>7}s {len(g):>6} {st.median(d):>11.5f} {st.mean(d):>12.5f} "
              f"{hs:>12.5f} {frac:>12.1%} {v:>22}")

    print("\n" + "=" * 94)
    print("BY TIME TO CLOSE -- is there a calm regime where making works?")
    print("=" * 94)
    print("  after the close the outcome is already fixed; if vol collapses while a")
    print("  spread persists, that is where making would be near-riskless\n")
    print(f"  {'bucket':>16} {'n':>6} {'avg_mid':>8} {'avg_spread':>11} "
          f"{'med |dmid5s|':>13} {'edge/fill':>10} {'verdict':>14}")
    for lo, hi in BUCKETS:
        g = [o for o in obs if lo <= o["rel"] < hi and o.get("d5") is not None]
        if len(g) < 40:
            continue
        spr = st.mean(o["spread"] for o in g)
        mv = st.median(o["d5"] for o in g)
        edge = spr / 2.0 - mv
        v = "VIABLE" if edge > 0 else ""
        print(f"  [{lo:>4},{hi:>4}) {len(g):>6} {st.mean(o['mid'] for o in g):>8.4f} "
              f"{spr:>11.5f} {mv:>13.5f} {edge:>+10.5f} {v:>14}")

    print("\n" + "=" * 94)
    print("REQUIRED REBATE TO BREAK EVEN")
    print("=" * 94)
    g = [o for o in obs if o.get("d5") is not None]
    spr = st.mean(o["spread"] for o in g)
    mv = st.mean(o["d5"] for o in g)
    px = st.mean(o["mid"] for o in g)
    need = mv - spr / 2.0
    print(f"  average spread          {spr:.5f}   (half = {spr/2:.5f})")
    print(f"  average 5s |mid move|   {mv:.5f}")
    print(f"  shortfall per fill      {need:+.5f} per share")
    print(f"  as a fraction of price  {need/px:.2%}")
    print(f"\n  a maker rebate would have to exceed {need:.5f}/share to break even.")
    print(f"  for scale, the TAKER fee at p={px:.2f} is {0.07*px*(1-px):.5f}/share, so the")
    print(f"  required rebate is {need/(0.07*px*(1-px)):.1f}x the entire taker fee.")

    print("\n" + "=" * 94)
    print("INVENTORY RISK: the structural problem with binary expiry")
    print("=" * 94)
    print("  a normal market maker holds inventory across time and mean-reverts out.")
    print("  here the contract settles to 0 or 1 within minutes, so unsold inventory")
    print("  is not a position to manage -- it is a coin flip that resolves against")
    print("  you roughly as often as the price implies.")
    stuck = [o for o in obs if o.get("d10") is not None and o["d10"] > 0.10]
    print(f"\n  observations where the mid moved >0.10 within 10s: "
          f"{len(stuck)}/{len([o for o in obs if o.get('d10') is not None])} "
          f"= {len(stuck)/max(1,len([o for o in obs if o.get('d10') is not None])):.1%}")
    print(f"  each such move is ~{0.10/(spr/2):.0f}x the half-spread earned on a fill.")


if __name__ == "__main__":
    main()
