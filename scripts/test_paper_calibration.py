"""Tests the central falsifiable claim of the SSRN paper across Polymarket categories.

The paper argues every profitable wallet runs the same trade, E[P] = q(p* - pm - c),
and that the mispricing p* - pm lives in identifiable price bands. Two of its eleven
regimes make claims that can be checked directly against resolved markets:

  R7  Lottery      sub-20c YES shares are UNDER-priced. Stated explicitly: "a 5c
                   contract whose true probability is 8% has expected return 60%".
  R1/R11 Politics  >55c YES shares are UNDER-priced -- "buys at 65c when integrated
                   polling implies 80%". This bucket is ~$140M of the $177M panel.

  R4  Crypto       contracts under $0.05 under-price the right tail.

That last one contradicts what this project measured directly: in BTC 5-minute
up/down markets, buying anything under 0.30 lost $1,100-1,400 per sample, with win
rates of 0.9-4.9% against prices of 1.6-6.9%. Long-shots there were badly OVER-priced,
which is the ordinary favourite-long-shot effect, not its reverse.

Both can be true if calibration differs by CATEGORY, which is the actual question.
The paper's long-shot claim rests on politics and event markets; my contrary finding
is from 5-minute crypto contracts. So this measures the calibration curve separately
per category rather than pooling them.

Two entry prices are recorded per market, because they answer different questions:
  vwap      volume-weighted average of every trade -- what buyers actually paid in
            aggregate, which is the DCA operator's entry and the paper's own metric
  mid_life  the price at the midpoint of the market's traded life -- a neutral
            snapshot, unweighted by where volume clustered
"""
import json
import random
import statistics as st
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scipy import stats as sps

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import polymarket_final_pct as pmf

PAGES = 120
PER_PAGE = 40
SAMPLE = 1600
MIN_VOLUME = 5000.0   # ignore illiquid legs that never trade away from zero
BANDS = [(0.00, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.35), (0.35, 0.55),
         (0.55, 0.70), (0.70, 0.85), (0.85, 0.95), (0.95, 1.00)]

CATS = {
    "crypto": ("crypto", "bitcoin", "ethereum", "solana", "btc", "eth"),
    "politics": ("politics", "election", "trump", "senate", "congress", "president",
                 "geopolitics", "cabinet"),
    "sports": ("sports", "nba", "nfl", "mlb", "soccer", "football", "epl", "ufc",
               "tennis", "cricket"),
}


def categorise(ev):
    labels = " ".join((t.get("label") or "").lower() for t in (ev.get("tags") or []))
    title = (ev.get("title") or "").lower()
    blob = labels + " " + title
    for cat, keys in CATS.items():
        if any(k in blob for k in keys):
            return cat
    return "other"


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


def build(job):
    cat, m = job
    try:
        pr = m.get("outcomePrices")
        pr = json.loads(pr) if isinstance(pr, str) else pr
        if not m.get("closed") or not pr or len(pr) != 2:
            return None
        yes_won = float(pr[0]) == 1.0
        cond = m.get("conditionId")
        if not cond:
            return None
    except Exception:
        return None
    trades = fetch_trades(cond)
    if len(trades) < 8:
        return None
    rows = []
    for t in trades:
        try:
            p = float(t["price"]); s = float(t.get("size", 0) or 0)
        except (TypeError, ValueError):
            continue
        if t.get("outcome") == "Yes":
            rows.append((t["timestamp"], p, s))
        elif t.get("outcome") == "No":
            rows.append((t["timestamp"], 1.0 - p, s))
    if len(rows) < 8:
        return None
    rows.sort()
    tot = sum(s for _, _, s in rows)
    if tot <= 0:
        return None
    vwap = sum(p * s for _, p, s in rows) / tot
    mid_life = rows[len(rows) // 2][1]
    return {"cat": cat, "yes": yes_won, "vwap": vwap, "mid": mid_life,
            "n_trades": len(rows)}


def main():
    print("collecting resolved events ...", flush=True)
    evs = []
    for pg in range(PAGES):
        try:
            b = pmf._get(pmf.GAMMA_BASE, "/events",
                         {"closed": "true", "limit": PER_PAGE, "offset": pg * PER_PAGE,
                          "order": "endDate", "ascending": "false"})
        except Exception:
            break
        if not b:
            break
        evs.extend(b)
    print(f"{len(evs)} resolved events")

    # Without a volume floor the sample is swamped by near-zero legs of multi-outcome
    # sports events: a first pass drew 708/900 sports markets and put 99 of 189 usable
    # observations in the sub-5c band, leaving nothing to test the >55c claim against.
    jobs = []
    for e in evs:
        cat = categorise(e)
        for m in (e.get("markets") or []):
            if not m.get("closed"):
                continue
            try:
                if float(m.get("volumeNum") or m.get("volume") or 0) < MIN_VOLUME:
                    continue
            except (TypeError, ValueError):
                continue
            jobs.append((cat, m))
    random.seed(42)
    random.shuffle(jobs)
    jobs = jobs[:SAMPLE]
    byc = defaultdict(int)
    for c, _ in jobs:
        byc[c] += 1
    print(f"{len(jobs)} markets sampled: " + ", ".join(f"{k}={v}" for k, v in byc.items()))
    print("fetching trade history ...", flush=True)

    out = []
    with ThreadPoolExecutor(max_workers=14) as pool:
        futs = [pool.submit(build, j) for j in jobs]
        done = 0
        for f in as_completed(futs):
            done += 1
            try:
                r = f.result()
            except Exception:
                continue
            if r:
                out.append(r)
            if done % 200 == 0:
                print(f"  {done}/{len(jobs)} ({len(out)} usable)", flush=True)
    print(f"\n{len(out)} markets with usable trade history\n")

    for key, label in (("vwap", "VOLUME-WEIGHTED ENTRY (what buyers actually paid)"),
                       ("mid", "MID-LIFE SNAPSHOT PRICE")):
        print("=" * 104)
        print(f"CALIBRATION BY CATEGORY -- {label}")
        print("=" * 104)
        print("  positive edge = the band resolves YES MORE often than its price implies")
        for cat in ("politics", "crypto", "sports", "other", "ALL"):
            g = out if cat == "ALL" else [r for r in out if r["cat"] == cat]
            if len(g) < 60:
                continue
            print(f"\n  {cat.upper()}  (n={len(g)})")
            print(f"  {'band':>14} {'n':>5} {'mean px':>9} {'YES rate':>9} "
                  f"{'edge':>9} {'p':>9}")
            for lo, hi in BANDS:
                b = [r for r in g if lo <= r[key] < hi]
                if len(b) < 20:
                    continue
                mp = st.mean(r[key] for r in b)
                k = sum(1 for r in b if r["yes"])
                rate = k / len(b)
                pv = sps.binomtest(k, len(b), min(0.999, max(0.001, mp))).pvalue
                flag = ""
                if pv < 0.05:
                    flag = "  <-- UNDER-priced" if rate > mp else "  <-- OVER-priced"
                print(f"  [{lo:.2f},{hi:.2f}) {len(b):>5} {mp:>9.4f} {rate:>9.4f} "
                      f"{rate-mp:>+9.4f} {pv:>9.4f}{flag}")
        print()


if __name__ == "__main__":
    main()
