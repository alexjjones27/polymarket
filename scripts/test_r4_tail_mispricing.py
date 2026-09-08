"""Direct test of the paper's R4 claim on the exact market type it names.

R4 says a crypto short-horizon bot harvests a TAIL mispricing: "top wins dominated by
extreme ROIs (>1000%) on contracts whose entry price was below $0.05 -- events the
market priced at <5% but which the operator's model assigned materially higher
probability... long convexity at near-zero premium when the market under-prices the
right-tail of crypto price moves." The stated market type is 15min-4h Up/Down on
BTC/ETH/XRP/SOL.

This project measured the opposite in 5-MINUTE up/down markets: buying anything under
0.30 lost $1,100-1,400 per sample, with win rates of 0.9-4.9% against prices of
1.6-6.9%. But 5m is outside the paper's stated 15min-4h range, so that is not yet a
refutation of R4 -- it is a different contract.

So this tests the 15m and 4h series held in cache, at the price bands R4 actually
names. For every moment a side traded below a threshold, how often did it go on to
win? If the tail is under-priced the realised rate exceeds the price paid.

Also reports the ROI shape, because R4's evidence is not an average -- it is a claim
about extreme winners ("dozens of zero-priced YES wins", one at 78,690% ROI). A
strategy can be net-losing and still produce spectacular individual wins, so the
distribution matters as much as the mean, and both are reported.
"""
import json
import statistics as st
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scipy import stats as sps

REPO = Path(r"D:\Finance\polymarket")
sys.path.insert(0, str(REPO / "src"))
import polymarket_final_pct as pmf

SERIES = [
    ("BTC 15m", ["btc_updown_15m_events", "btc_updown_15m_events_oos"], 900),
    ("BTC 4h", ["btc_updown_4h_extended"], 14400),
    ("ETH 4h", ["eth_updown_4h_extended"], 14400),
    ("SOL 4h", ["sol_updown_4h_extended"], 14400),
]
THRESHOLDS = [0.02, 0.05, 0.10, 0.20]
SAMPLE = 320
FEE_RATE = 0.07


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
        if off > 1500:
            break
    return out


def build(e):
    try:
        m = e["markets"][0]
        pr = m.get("outcomePrices")
        pr = json.loads(pr) if isinstance(pr, str) else pr
        if not m.get("closed") or len(pr) != 2:
            return None
        up_won = float(pr[0]) == 1.0
        cond = m.get("conditionId")
        if not cond:
            return None
    except Exception:
        return None
    ser = {"Up": [], "Down": []}
    for t in sorted(fetch_trades(cond) or [], key=lambda x: x["timestamp"]):
        try:
            p = float(t["price"])
        except (TypeError, ValueError):
            continue
        if t.get("outcome") == "Up":
            ser["Up"].append(p); ser["Down"].append(1.0 - p)
        elif t.get("outcome") == "Down":
            ser["Down"].append(p); ser["Up"].append(1.0 - p)
    if len(ser["Up"]) < 10:
        return None
    return {"up_won": up_won, "ser": ser}


def main():
    import random
    for label, caches, wlen in SERIES:
        evs = {}
        for c in caches:
            p = REPO / "data" / "raw" / "polymarket" / f"{c}.json"
            if p.exists():
                for e in json.loads(p.read_text())["events"]:
                    try:
                        evs[e["slug"]] = e
                    except Exception:
                        continue
        evs = list(evs.values())
        if len(evs) < 100:
            print(f"{label}: only {len(evs)} cached, skipping")
            continue
        random.seed(42)
        samp = random.sample(evs, min(SAMPLE, len(evs)))
        built = []
        with ThreadPoolExecutor(max_workers=14) as pool:
            futs = [pool.submit(build, e) for e in samp]
            for f in as_completed(futs):
                try:
                    r = f.result()
                except Exception:
                    continue
                if r:
                    built.append(r)
        print(f"\n{'='*96}")
        print(f"{label}   ({len(built)} windows with trade history)")
        print(f"{'='*96}")
        print(f"  {'buy below':>10} {'n obs':>7} {'mean px':>9} {'win rate':>9} "
              f"{'edge':>9} {'p':>9} {'net/share':>11} {'best ROI':>10}")
        for thr in THRESHOLDS:
            obs = []
            for r in built:
                for side in ("Up", "Down"):
                    won = r["up_won"] if side == "Up" else (not r["up_won"])
                    # every moment this side traded below the threshold
                    for p in r["ser"][side]:
                        if 0.001 < p < thr:
                            obs.append((p, won))
            if len(obs) < 40:
                continue
            n = len(obs)
            mp = st.mean(p for p, _ in obs)
            k = sum(1 for _, w in obs if w)
            rate = k / n
            pv = sps.binomtest(k, n, min(0.999, max(0.001, mp))).pvalue
            fee = FEE_RATE * mp * (1 - mp)
            net = rate - mp - fee
            best = max(((1.0 - p) / p) for p, w in obs if w) if k else 0.0
            flag = ""
            if pv < 0.05:
                flag = "  <-- UNDER" if rate > mp else "  <-- OVER"
            print(f"  {thr:>10.2f} {n:>7} {mp:>9.4f} {rate:>9.4f} {rate-mp:>+9.4f} "
                  f"{pv:>9.4f} {net:>+11.4f} {best:>9.0f}%{flag}")

        # ROI distribution at the headline threshold, since R4's evidence is extremes
        obs = []
        for r in built:
            for side in ("Up", "Down"):
                won = r["up_won"] if side == "Up" else (not r["up_won"])
                for p in r["ser"][side]:
                    if 0.001 < p < 0.05:
                        obs.append((p, won))
        if obs:
            wins = [(1.0 - p) / p for p, w in obs if w]
            n = len(obs)
            print(f"\n  sub-5c ROI shape: {len(wins)} winners of {n} entries "
                  f"({len(wins)/n:.2%})")
            if wins:
                wins.sort()
                print(f"    winner ROIs: median {st.median(wins)*100:.0f}%, "
                      f"p90 {wins[int(0.9*len(wins))]*100:.0f}%, max {max(wins)*100:.0f}%")
            gross = sum(((1.0 - p) if w else -p) for p, w in obs)
            print(f"    net per $1 staked across ALL sub-5c entries: "
                  f"${gross/sum(p for p, _ in obs):+.4f}")


if __name__ == "__main__":
    main()
