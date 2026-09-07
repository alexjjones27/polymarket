"""Strategy backtest of the 4h mean reversion.

Established on 1,100 windows: after an UP window the next is up 46.14%, after a DOWN
window 55.17%. Spread +9.03%, Fisher p=0.0031, same direction in all three disjoint
chronological thirds. That part is solid.

Whether it is TRADEABLE depends on the market not already pricing it, and the
gap-based test of that was shown to be unreliable -- run as a negative control on
the 5m and 15m series, which have no serial dependence at all (p=0.74 and p=0.91
over 8,056 and 2,880 windows), it still produced spurious +7pp "gaps" at n~200. A
measurement that finds a 7pp effect where none exists cannot certify a 5pp one.

So this measures money instead of gaps:

  after an UP window   -> buy DOWN early in the next window
  after a DOWN window  -> buy UP early in the next window

paying the ask (modelled as the print plus the 0.005 half-spread, the tick floor
being 0.01 at every horizon) and the real taker fee 0.07*p*(1-p), which near 0.50 is
a punishing 0.0175/share -- the largest it ever gets, and the reason this trade is
far more fee-sensitive than the 0.99-priced favourites we traded before.

Split chronologically into halves so the second is genuinely out of sample.
"""
import json
import statistics as st
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scipy import stats as sps

REPO = Path(r"D:\Finance\polymarket")
sys.path.insert(0, str(REPO / "src"))
import polymarket_final_pct as pmf

STAKE = 5.0
HALF_SPREAD = 0.005
OFFSETS = [-14000, -13500, -12600, -10800]
SRC = "btc_updown_4h_extended"


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


def main():
    p = REPO / "data" / "raw" / "polymarket" / f"{SRC}.json"
    evs = sorted(json.loads(p.read_text())["events"], key=lambda e: e.get("endDate", ""))
    print(f"{len(evs)} 4h windows; fetching prices ...", flush=True)

    built = {}
    with ThreadPoolExecutor(max_workers=12) as pool:
        futs = {pool.submit(build, e): i for i, e in enumerate(evs)}
        done = 0
        for f in as_completed(futs):
            done += 1
            try:
                r = f.result()
            except Exception:
                continue
            if r:
                built[futs[f]] = r
            if done % 250 == 0:
                print(f"  {done}/{len(evs)}", flush=True)
    print(f"  built {len(built)}\n")

    for off in OFFSETS:
        trades = []
        for i in range(1, len(evs)):
            cur, prev = built.get(i), built.get(i - 1)
            if not cur or not prev:
                continue
            px_up = price_at(cur["ser"], off)
            if px_up is None or not (0.05 < px_up < 0.95):
                continue
            # reversion: bet AGAINST the previous window's direction
            if prev["up"]:
                entry, won = 1.0 - px_up, (not cur["up"])      # buy Down
            else:
                entry, won = px_up, cur["up"]                  # buy Up
            ask = min(0.98, entry + HALF_SPREAD)
            shares = STAKE / ask
            cost = shares * ask + shares * fee(ask)
            trades.append({"i": i, "pnl": (shares - cost) if won else -cost,
                           "won": won, "ask": ask, "cost": cost})
        if len(trades) < 100:
            continue
        half = len(trades) // 2
        print(f"{'='*88}")
        print(f"ENTRY AT close{off}s   ({len(trades)} trades)")
        print(f"{'='*88}")
        print(f"  {'sample':>12} {'n':>5} {'win_rate':>9} {'avg_ask':>8} {'breakeven':>10} "
              f"{'pnl':>9} {'return':>9} {'p vs 50%':>9}")
        for lab, seg in (("IN-SAMPLE", trades[:half]), ("OUT-OF-SAMPLE", trades[half:]),
                         ("combined", trades)):
            n = len(seg)
            w = sum(1 for t in seg if t["won"])
            pnl = sum(t["pnl"] for t in seg)
            inv = sum(t["cost"] for t in seg)
            aa = st.mean(t["ask"] for t in seg)
            be = aa + fee(aa)
            pv = sps.binomtest(w, n, 0.5).pvalue
            print(f"  {lab:>12} {n:>5} {w/n:>8.2%} {aa:>8.4f} {be:>10.4f} "
                  f"{pnl:>+9.2f} {pnl/inv:>+8.2%} {pv:>9.4f}")
        print()


if __name__ == "__main__":
    main()
