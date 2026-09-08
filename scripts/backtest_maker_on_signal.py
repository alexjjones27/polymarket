"""The one untested combination: rest orders WHERE the signal says, rather than
taking whatever the signal points at.

Everything killed so far crossed the spread and paid 0.07*p*(1-p) as a taker. The
scalp needed the mid to travel 4.5c before earning a cent. A maker pays neither the
spread nor that fee, so the round trip roughly halves -- and the entry price is below
the mid rather than above it.

The earlier resting-bid test failed, but it rested BLINDLY: a fixed offset under the
favourite, filled 64-88% of the time by the price gapping straight through. That is
pure adverse selection, and it is what the spread exists to compensate makers for.

The refinement is to rest only when the imbalance signal is bullish. A resting bid
fills when price comes down to it; if the book is strongly bid-heavy that dip is more
likely to be transient noise inside an uptrend rather than the front of a collapse.
So the signal is used to separate the dips worth catching from the ones that run you
over.

Three arms, so the signal's contribution is isolated rather than assumed:
  BLIND     rest regardless of the book (reproduces the known failure)
  BULLISH   rest only when imb1 >= IMB_HI
  BEARISH   rest only when imb1 <= IMB_LO (should be WORSE if the signal is real --
            a falsification check, not decoration)

Costs modelled conservatively:
  entry  maker, filled at our resting price, fee assumed ZERO (no rebate credited,
         so any real rebate only improves this)
  exit   taker, crossing to the bid, paying half-spread and the full taker fee
  queue  we join the BACK, advancing only as depth ahead of us drains, and we are
         filled in full if the price gaps through our level
"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

REPO = Path(r"D:\Finance\polymarket")
FEE_RATE = 0.07
STAKE = 5.0
MID_LO, MID_HI = 0.30, 0.70
IMB_HI, IMB_LO = 0.70, 0.30
OFFSETS = [0.01, 0.02, 0.03]
TARGETS = [0.03, 0.05, 0.10]
STOP = 0.10
MAX_WAIT_S = 60      # how long the resting order stays live
MAX_HOLD_S = 120     # how long we hold once filled


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
                "bids": b,
            })
    for k in series:
        series[k].sort(key=lambda x: x["ts"])
    return series


def depth_at(bids, px):
    for p, s in bids:
        if abs(p - px) < 1e-6:
            return s
    return 0.0


def simulate(rows, i, offset, target):
    """Rest a bid `offset` below mid; if filled, exit at target/stop/timeout."""
    e = rows[i]
    level = round(e["mid"] - offset, 2)
    if level <= 0.02 or level >= 1.0:
        return None
    ahead = depth_at(e["bids"], level)
    advanced = 0.0
    prev = ahead
    fill = None
    for j in range(i + 1, len(rows)):
        x = rows[j]
        if x["ts"] - e["ts"] > MAX_WAIT_S:
            break
        if x["bid"] < level:                      # gapped through: filled adversely
            fill = (j, level, True)
            break
        d = depth_at(x["bids"], level)
        if d < prev:
            advanced += (prev - d)
        prev = d
        if advanced >= ahead and abs(x["bid"] - level) < 1e-6:
            fill = (j, level, False)
            break
    if fill is None:
        return {"filled": False}

    fi, fpx, gapped = fill
    shares = STAKE / fpx
    cost = shares * fpx                            # maker: no spread, no fee assumed
    tgt = fpx + target
    stp = fpx - STOP
    ft = rows[fi]["ts"]
    for j in range(fi + 1, len(rows)):
        x = rows[j]
        if x["ts"] - ft > MAX_HOLD_S:
            break
        if x["mid"] <= stp or x["mid"] >= tgt:
            proceeds = shares * x["bid"] - shares * fee(x["bid"])
            return {"filled": True, "gapped": gapped, "pnl": proceeds - cost,
                    "outcome": "target" if x["mid"] >= tgt else "stop"}
    last = None
    for j in range(fi + 1, len(rows)):
        if rows[j]["ts"] - ft > MAX_HOLD_S:
            break
        last = rows[j]
    if last is None:
        return {"filled": False}
    proceeds = shares * last["bid"] - shares * fee(last["bid"])
    return {"filled": True, "gapped": gapped, "pnl": proceeds - cost, "outcome": "timeout"}


def main():
    series = load()
    print(f"{sum(len(v) for v in series.values())} snapshots, {len(series)} series")
    print(f"maker entry (fee assumed ZERO), taker exit (half-spread + full fee)")
    print(f"queue: join the back, advance as depth drains, gap-through fills in full\n")

    arms = {
        "BLIND": lambda r: True,
        "BULLISH": lambda r: r["imb1"] >= IMB_HI,
        "BEARISH": lambda r: r["imb1"] <= IMB_LO,
    }
    for offset in OFFSETS:
        for target in TARGETS:
            print("=" * 104)
            print(f"REST {offset:.2f} BELOW MID   TARGET +{target:.2f}   STOP -{STOP:.2f}")
            print("=" * 104)
            print(f"  {'arm':>9} {'cands':>7} {'filled':>7} {'fill%':>7} {'gapped':>8} "
                  f"{'hit_tgt':>8} {'avg_pnl':>9} {'total':>9} {'return':>8}")
            for name, cond in arms.items():
                res = []
                cands = 0
                for key, rows in series.items():
                    for i, r in enumerate(rows):
                        if not (MID_LO <= r["mid"] <= MID_HI) or not cond(r):
                            continue
                        cands += 1
                        s = simulate(rows, i, offset, target)
                        if s and s["filled"]:
                            res.append(s)
                if len(res) < 30:
                    print(f"  {name:>9} {cands:>7} {len(res):>7}   (too few fills)")
                    continue
                gp = sum(1 for x in res if x["gapped"]) / len(res)
                tg = sum(1 for x in res if x["outcome"] == "target") / len(res)
                tot = sum(x["pnl"] for x in res)
                inv = len(res) * STAKE
                print(f"  {name:>9} {cands:>7} {len(res):>7} {len(res)/max(1,cands):>6.1%} "
                      f"{gp:>7.1%} {tg:>7.1%} {tot/len(res):>+9.4f} {tot:>+9.2f} "
                      f"{tot/inv:>+7.2%}")
            print()


if __name__ == "__main__":
    main()
