"""Calibration curve for the longer-horizon BTC up/down markets.

This is the tool that produced the only trustworthy answers on the 5-minute market.
It samples BOTH sides of every window at a fixed time before close and buckets by
price, so exactly one side of each window wins and there is no selection at all. On
the 5m series it found the [0.95,1.00) band underpriced by 1.3-1.7c/share -- real,
replicated across both samples, and too thin to harvest after a 0.5c half-spread
plus fee.

The question here is whether a longer horizon widens that gap. The spread is 0.010
at every horizon because it is the tick floor, so the cost side is fixed while the
horizon grows 3x (15m) and 48x (4h). If the mispricing scales with horizon at all,
the ratio improves.

Sampled at several offsets, because on a longer contract the interesting region may
sit much further from expiry than it did on a 5-minute one.

Break-even for buying at price p is a win rate of p plus the fee 0.07*p*(1-p); the
net_edge column is already net of that fee but NOT of the half-spread, which is a
further 0.005 if taking. Anything below +0.005 net is unharvestable as a taker.
"""
import json
import random
import statistics as st
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scipy import stats as sps

REPO = Path(r"D:\Finance\polymarket")
sys.path.insert(0, str(REPO / "src"))
import polymarket_final_pct as pmf

SAMPLE_N = 400
HALF_SPREAD = 0.005
CONFIGS = [
    ("btc_updown_15m", [-600, -300, -120, -60], 900),
    ("btc_updown_4h", [-7200, -3600, -1800, -600], 14400),
]
BANDS = [(i / 20, (i + 1) / 20) for i in range(20)]


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
        if off > 3000:
            break
    return out


def build(event):
    try:
        m = event["markets"][0]
        pr = pmf._safe_json_list(m.get("outcomePrices"))
        if not m.get("closed") or len(pr) != 2:
            return None
        up_won = float(pr[0]) == 1.0
        end_s = int(pmf.pd.Timestamp(event["endDate"]).timestamp())
        cond = m.get("conditionId")
        if not cond:
            return None
    except Exception:
        return None
    trades = fetch_trades(cond)
    if not trades:
        return None
    ser = []
    for t in sorted(trades, key=lambda x: x["timestamp"]):
        try:
            p = float(t["price"])
        except (TypeError, ValueError):
            continue
        rel = t["timestamp"] - end_s
        if t.get("outcome") == "Up":
            ser.append((rel, p))
        elif t.get("outcome") == "Down":
            ser.append((rel, 1.0 - p))
    if not ser:
        return None
    return {"ser": ser, "up_won": up_won}


def price_at(ser, offset):
    last = None
    for rel, p in ser:
        if rel <= offset:
            last = p
        else:
            break
    return last


def load(cache):
    p = REPO / "data" / "raw" / "polymarket" / f"{cache}.json"
    if not p.exists():
        return []
    evs = json.loads(p.read_text())["events"]
    random.seed(42)
    sample = random.sample(evs, min(SAMPLE_N, len(evs)))
    out = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        futs = {pool.submit(build, e): e for e in sample}
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception:
                continue
            if r:
                out.append(r)
    return out


def calibrate(rows, offset, label):
    buckets = {}
    for r in rows:
        for side in ("Up", "Down"):
            ser = r["ser"] if side == "Up" else [(x, 1.0 - y) for x, y in r["ser"]]
            px = price_at(ser, offset)
            if px is None:
                continue
            won = r["up_won"] if side == "Up" else (not r["up_won"])
            for lo, hi in BANDS:
                if lo <= px < hi:
                    b = buckets.setdefault((lo, hi), [0, 0])
                    b[0] += 1
                    b[1] += 1 if won else 0
                    break
    printed = False
    for (lo, hi), (n, w) in sorted(buckets.items()):
        if n < 40:
            continue
        mid = (lo + hi) / 2
        act = w / n
        net = act - mid - fee(mid)
        harvest = net - HALF_SPREAD
        pv = sps.binomtest(w, n, min(0.999, mid + fee(mid)), alternative="two-sided").pvalue
        tag = ""
        if pv < 0.01 and harvest > 0:
            tag = "  <== CLEARS SPREAD TOO"
        elif pv < 0.01 and net > 0:
            tag = "  <- edge, but under spread"
        elif pv < 0.01:
            tag = "  <- overpriced"
        if not printed:
            print(f"\n  at close{offset:+d}s:")
            print(f"  {'band':>14} {'n':>5} {'implied':>8} {'actual':>8} {'net_edge':>9} "
                  f"{'-half_spr':>10} {'p':>8}")
            printed = True
        print(f"  [{lo:.2f},{hi:.2f}) {n:>5} {mid:>8.3f} {act:>8.3f} {net:>+9.4f} "
              f"{harvest:>+10.4f} {pv:>8.4f}{tag}")


def main():
    for base, offsets, wlen in CONFIGS:
        for suffix, lab in [("_events", "IN-SAMPLE"), ("_events_oos", "OOS")]:
            rows = load(base + suffix)
            if not rows:
                print(f"\n{base} {lab}: no data")
                continue
            print(f"\n{'='*84}\n{base}  {lab}  ({len(rows)} windows, {wlen}s each)\n{'='*84}")
            for off in offsets:
                calibrate(rows, off, lab)


if __name__ == "__main__":
    main()
