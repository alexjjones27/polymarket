"""Does the hour-08 reversion clear its costs? The decisive test.

Established so far. BTC 4h reverses one step: after an up window the next is up
46.14%, after a down window 55.17% (p=0.0031, n=1,099). Within that, windows closing
at 08:00 UTC -- covering 04:00-08:00 UTC, the Asian session ahead of the European
open -- concentrate the effect far more strongly:

  BTC hour 08  after-up 37.93%  after-down 61.46%  spread +23.53%  p=0.0019
  ETH hour 08  after-up 36.90%  after-down 60.61%  spread +23.70%  p=0.0018
  SOL hour 08  after-up 39.33%  after-down 53.19%  spread +13.87%  p=0.0752

Hour 08 was chosen by searching six hours on BTC, so ETH and SOL were meant to be the
independent check -- but they are not independent: BTC and ETH share an outcome 85%
of the time, all three 74.7%. So the corroboration is partial at best, and BTC alone
is the real evidence.

The whole-sample 4h reversion already failed to clear costs: it trades near 0.52
where the fee 0.07*p*(1-p) peaks at 0.0175/share, break-even is ~53.5%, and 54.46%
observed left a margin inside the noise. Hour 08's much larger spread is the only
reason to think the trade is different. At 61-62% against ~53.5% the margin would be
7-8pp, which WOULD clear.

So this measures money, on hour-08 windows only, splitting chronologically, with the
real ask and the real fee.
"""
import json
import statistics as st
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from scipy import stats as sps

REPO = Path(r"D:\Finance\polymarket")
sys.path.insert(0, str(REPO / "src"))
import polymarket_final_pct as pmf

STAKE = 5.0
HALF_SPREAD = 0.005
TARGET_HOUR = 8
OFFSETS = [-14000, -13500, -12600]


def fee(p):
    return 0.07 * p * (1 - p)


def fetch_trades(cond):
    out, off = [], 0
    while True:
        try:
            pg = pmf._get(pmf.DATA_API_BASE, "/trades",
                          {"market": cond, "limit": 500, "offset": off})
        except Exception:
            break
        if not pg:
            break
        out.extend(pg)
        if len(pg) < 500:
            break
        off += 500
        if off > 2000:
            break
    return out


def build(e):
    try:
        m = e["markets"][0]
        pr = m.get("outcomePrices")
        pr = json.loads(pr) if isinstance(pr, str) else pr
        if not m.get("closed") or len(pr) != 2:
            return None
        end_s = int(pmf.pd.Timestamp(e["endDate"]).timestamp())
        cond = m.get("conditionId")
        if not cond:
            return None
    except Exception:
        return None
    ser = []
    for t in sorted(fetch_trades(cond) or [], key=lambda x: x["timestamp"]):
        try:
            p = float(t["price"])
        except (TypeError, ValueError):
            continue
        rel = t["timestamp"] - end_s
        if t.get("outcome") == "Up":
            ser.append((rel, p))
        elif t.get("outcome") == "Down":
            ser.append((rel, 1.0 - p))
    return {"up": float(pr[0]) == 1.0, "ser": ser}


def price_at(ser, off):
    last = None
    for rel, p in ser:
        if rel <= off:
            last = p
        else:
            break
    return last


def load(asset):
    p = REPO / "data" / "raw" / "polymarket" / f"{asset}_updown_4h_extended.json"
    evs = json.loads(p.read_text())["events"]
    out = []
    for e in evs:
        try:
            dt = datetime.fromisoformat(e["endDate"].replace("Z", "+00:00"))
        except Exception:
            continue
        out.append((dt, e))
    out.sort(key=lambda x: x[0])
    return out


def main():
    for asset in ("btc", "eth"):
        evs = load(asset)
        # indices of hour-08 windows, and their immediate predecessors
        idx = [i for i, (dt, _) in enumerate(evs)
               if dt.astimezone(timezone.utc).hour == TARGET_HOUR and i > 0]
        need = sorted(set(idx) | {i - 1 for i in idx})
        print(f"\n{asset.upper()}: {len(idx)} hour-08 windows; fetching {len(need)} ...",
              flush=True)
        built = {}
        with ThreadPoolExecutor(max_workers=12) as pool:
            futs = {pool.submit(build, evs[i][1]): i for i in need}
            for f in as_completed(futs):
                try:
                    r = f.result()
                except Exception:
                    continue
                if r:
                    built[futs[f]] = r

        for off in OFFSETS:
            trades = []
            for i in idx:
                cur, prev = built.get(i), built.get(i - 1)
                if not cur or not prev:
                    continue
                px_up = price_at(cur["ser"], off)
                if px_up is None or not (0.05 < px_up < 0.95):
                    continue
                if prev["up"]:
                    entry, won = 1.0 - px_up, (not cur["up"])   # buy Down
                else:
                    entry, won = px_up, cur["up"]                # buy Up
                ask = min(0.98, entry + HALF_SPREAD)
                shares = STAKE / ask
                cost = shares * ask + shares * fee(ask)
                trades.append({"pnl": (shares - cost) if won else -cost,
                               "won": won, "ask": ask, "cost": cost})
            if len(trades) < 40:
                continue
            h = len(trades) // 2
            print(f"\n  entry at close{off}s   ({len(trades)} trades)")
            print(f"  {'sample':>14} {'n':>4} {'win':>8} {'avg_ask':>8} {'breakeven':>10} "
                  f"{'margin':>8} {'pnl':>9} {'return':>9} {'p vs BE':>8}")
            for lab, seg in (("first half", trades[:h]), ("second half", trades[h:]),
                             ("FULL", trades)):
                n = len(seg)
                w = sum(1 for t in seg if t["won"])
                pnl = sum(t["pnl"] for t in seg)
                inv = sum(t["cost"] for t in seg)
                aa = st.mean(t["ask"] for t in seg)
                be = aa + fee(aa)
                se = (be * (1 - be) / n) ** 0.5
                z = (w / n - be) / se if se else 0
                pv = 1 - sps.norm.cdf(z)
                flag = "  <== clears" if pv < 0.05 else ""
                print(f"  {lab:>14} {n:>4} {w/n:>7.2%} {aa:>8.4f} {be:>10.4f} "
                      f"{w/n-be:>+8.2%} {pnl:>+9.2f} {pnl/inv:>+8.2%} {pv:>8.4f}{flag}")


if __name__ == "__main__":
    main()
