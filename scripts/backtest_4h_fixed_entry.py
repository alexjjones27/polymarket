"""Strategy backtest for the 4h favourite edge -- not a calibration curve.

The calibration showed the [0.95,1.00) band is underpriced at close-600s in all four
samples, by +0.0144 to +0.0197 net of fee. That is a statement about a price bucket,
not about a rule. This tests an actual rule with actual execution, because the whole
5m disaster came from exactly that gap: the backtest described a strategy we were
not running.

The rule is pre-specified from the calibration, deliberately, so it is not fitted:

  at ENTRY_OFFSET seconds before close, look once at the prevailing price
  if it is >= THRESHOLD, buy at the ask, size to STAKE, hold to settlement
  no sustained crossing, no path condition -- a single fixed-time observation,
  which is precisely what the calibration measured

Execution modelled honestly given we have prints but not books for history:
  - we pay the ASK, modelled as the print plus a half-spread. The spread is the
    0.01 tick floor at every horizon we measured live, so half-spread = 0.005.
  - the real taker fee 0.07*p*(1-p) is charged on entry
  - MIN_SHARES=5 is enforced, as the exchange does

Reports in-sample and OOS separately, plus max drawdown and worst trade, because a
99.6% win rate with a full-stake tail is exactly the shape that looked fine on the
5m market and was not.
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

STAKE = 5.0
MIN_SHARES = 5.0
HALF_SPREAD = 0.005
FEE_RATE = 0.07
SAMPLE_N = 400

# pre-specified from the calibration
PRIMARY = {"btc_updown_4h": (-600, 0.95), "btc_updown_15m": (-60, 0.95)}
# exploratory sweeps, flagged as such
OFFSETS = {"btc_updown_4h": [-1800, -1200, -600, -300], "btc_updown_15m": [-120, -60, -30]}
THRESHOLDS = [0.90, 0.95, 0.97, 0.99]


def fee(shares, p):
    return shares * FEE_RATE * p * (1 - p)


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
    tr = fetch_trades(cond)
    if not tr:
        return None
    ser = []
    for t in sorted(tr, key=lambda x: x["timestamp"]):
        try:
            p = float(t["price"])
        except (TypeError, ValueError):
            continue
        rel = t["timestamp"] - end_s
        if t.get("outcome") == "Up":
            ser.append((rel, p))
        elif t.get("outcome") == "Down":
            ser.append((rel, 1.0 - p))
    return {"ser": ser, "up_won": up_won} if ser else None


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


def price_at(ser, offset):
    last = None
    for rel, p in ser:
        if rel <= offset:
            last = p
        else:
            break
    return last


def run(rows, offset, threshold):
    trades = []
    for r in rows:
        for side in ("Up", "Down"):
            ser = r["ser"] if side == "Up" else [(x, 1.0 - y) for x, y in r["ser"]]
            px = price_at(ser, offset)
            if px is None or px < threshold or px >= 1.0:
                continue
            ask = min(0.999, px + HALF_SPREAD)      # we pay the ask
            shares = max(MIN_SHARES, STAKE / ask)
            cost = shares * ask + fee(shares, ask)
            won = r["up_won"] if side == "Up" else (not r["up_won"])
            trades.append({"pnl": (shares - cost) if won else -cost,
                           "won": won, "ask": ask, "cost": cost})
            break   # at most one side can be above 0.95
    return trades


def summarize(trades, label, per_day):
    if not trades:
        print(f"  {label:>10}  no qualifying trades")
        return None
    n = len(trades)
    w = sum(1 for t in trades if t["won"])
    pnl = sum(t["pnl"] for t in trades)
    inv = sum(t["cost"] for t in trades)
    # drawdown over the sequence as ordered
    cum, peak, dd = 0.0, 0.0, 0.0
    for t in trades:
        cum += t["pnl"]
        peak = max(peak, cum)
        dd = min(dd, cum - peak)
    worst = min(t["pnl"] for t in trades)
    lo = sps.beta.ppf(0.025, w, n - w + 1) if w < n else (0.025) ** (1 / n)
    print(f"  {label:>10} {n:>5} {w/n:>8.2%} [{lo:>6.2%}] {st.mean(t['ask'] for t in trades):>7.4f} "
          f"{pnl:>+9.2f} {pnl/inv:>+8.2%} {dd:>+9.2f} {worst:>+8.2f} "
          f"{pnl/n*per_day*(n/len(trades)):>+8.2f}")
    return {"n": n, "wr": w / n, "pnl": pnl, "inv": inv}


def main():
    for base in ("btc_updown_4h", "btc_updown_15m"):
        per_day = 6 if "4h" in base else 96
        ins = load(base + "_events")
        oos = load(base + "_events_oos")
        if not ins or not oos:
            print(f"{base}: missing data")
            continue
        off, thr = PRIMARY[base]
        print(f"\n{'='*104}")
        print(f"{base}   PRE-SPECIFIED RULE: buy at close{off}s if price >= {thr}")
        print(f"{'='*104}")
        print(f"  {'sample':>10} {'n':>5} {'win_rate':>8} {'[95%lo]':>8} {'avg_ask':>7} "
              f"{'pnl':>9} {'return':>8} {'max_dd':>9} {'worst':>8} {'$/trade':>8}")
        a = summarize(run(ins, off, thr), "IN-SAMPLE", per_day)
        b = summarize(run(oos, off, thr), "OOS", per_day)
        if a and b:
            tot = a["pnl"] + b["pnl"]
            ntot = a["n"] + b["n"]
            print(f"\n  combined: {ntot} trades, ${tot:+.2f}, "
                  f"{tot/(a['inv']+b['inv']):+.2%} on turnover, "
                  f"${tot/ntot:+.4f}/trade -> ${tot/ntot*per_day:+.2f}/day at {per_day} windows/day")

        print(f"\n  EXPLORATORY sweep (multiple comparisons -- treat as descriptive only)")
        print(f"  {'offset':>8} {'thresh':>7} {'IS n':>6} {'IS ret':>8} {'OOS n':>6} {'OOS ret':>8} "
              f"{'both+?':>7}")
        for o in OFFSETS[base]:
            for t in THRESHOLDS:
                ti = run(ins, o, t)
                to = run(oos, o, t)
                if len(ti) < 20 or len(to) < 20:
                    continue
                ri = sum(x["pnl"] for x in ti) / sum(x["cost"] for x in ti)
                ro = sum(x["pnl"] for x in to) / sum(x["cost"] for x in to)
                mark = "yes" if ri > 0 and ro > 0 else ""
                print(f"  {o:>8} {t:>7.2f} {len(ti):>6} {ri:>+7.2%} {len(to):>6} "
                      f"{ro:>+7.2%} {mark:>7}")


if __name__ == "__main__":
    main()
