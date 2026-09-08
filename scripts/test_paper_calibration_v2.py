"""Calibration by category, with the lookahead bias removed.

The first attempt was contaminated and its output should be discarded. It priced each
market by the volume-weighted average of EVERY trade in its life, which includes all
the trades made after the outcome was effectively known. Markets that resolved YES
accumulate late fills near $1 and markets that resolved NO accumulate them near $0, so
the VWAP encodes the answer. The symptom was unmistakable once looked at: every band
above 0.55 showed a YES rate of exactly 1.0000 and every band below 0.35 exactly
0.0000. That is not a calibration curve, it is the outcome measured twice.

The "mid-life" price had a weaker form of the same problem, being the median trade by
COUNT rather than by time -- and trade volume concentrates near resolution, so the
median trade sits much later in the market's life than halfway.

This version samples the price at a fixed LEAD TIME before the market's end date, so
nothing after that instant can influence it. Three leads are reported, because a claim
that a band is mispriced should not depend on when you look:

  168h (1 week), 72h (3 days), 24h (1 day) before resolution

The claims under test remain those of the paper:
  R7      sub-20c YES shares are UNDER-priced ("a 5c contract whose true probability
          is 8% returns 60% per trade")
  R1/R11  >55c YES shares are UNDER-priced, ~$140M of the $177M panel
  R4      crypto contracts under $0.05 under-price the right tail
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
SAMPLE = 1800
MIN_VOLUME = 5000.0
FRACS = [0.25, 0.50, 0.75]   # fraction through the market's TRADING life, by time
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
    blob = (" ".join((t.get("label") or "").lower() for t in (ev.get("tags") or []))
            + " " + (ev.get("title") or "").lower())
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
        end_s = int(pmf.pd.Timestamp(m.get("endDate") or m.get("endDateIso")).timestamp())
        if not cond:
            return None
    except Exception:
        return None
    trades = fetch_trades(cond)
    if len(trades) < 10:
        return None
    rows = []
    for t in trades:
        try:
            p = float(t["price"])
        except (TypeError, ValueError):
            continue
        if t.get("outcome") == "Yes":
            rows.append((t["timestamp"], p))
        elif t.get("outcome") == "No":
            rows.append((t["timestamp"], 1.0 - p))
    if len(rows) < 10:
        return None
    rows.sort()

    # A fixed lead before endDate does not work: endDate is frequently long after the
    # outcome was actually decided, so "168h before endDate" lands after the event and
    # every price is already 0.00 or 1.00. Sampling by fraction of the market's real
    # TRADING life keeps us inside the period when it was genuinely live, and uses only
    # trades up to that instant, so there is no lookahead.
    t0, t1 = rows[0][0], rows[-1][0]
    span = t1 - t0
    if span < 3600:
        return None
    out = {"cat": cat, "yes": yes_won, "span_h": span / 3600.0}
    for f in FRACS:
        cutoff = t0 + f * span
        prior = [p for ts, p in rows if ts <= cutoff]
        out[f"f{int(f*100)}"] = prior[-1] if len(prior) >= 3 else None
    return out if any(out.get(f"f{int(f*100)}") is not None for f in FRACS) else None


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
    print(f"{len(jobs)} markets sampled; fetching trade history ...", flush=True)

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
            if done % 300 == 0:
                print(f"  {done}/{len(jobs)} ({len(out)} usable)", flush=True)
    print(f"\n{len(out)} markets usable\n")

    import statistics as _st
    print(f"median market trading life: {_st.median([r['span_h'] for r in out]):.1f}h\n")
    for f in FRACS:
        key = f"f{int(f*100)}"
        avail = [r for r in out if r.get(key) is not None]
        print("=" * 100)
        print(f"PRICE AT {int(f*100)}% THROUGH THE MARKET'S TRADING LIFE   (n={len(avail)})")
        print("=" * 100)
        for cat in ("politics", "crypto", "sports", "other", "ALL"):
            g = avail if cat == "ALL" else [r for r in avail if r["cat"] == cat]
            if len(g) < 80:
                continue
            print(f"\n  {cat.upper()}  (n={len(g)})")
            print(f"  {'band':>14} {'n':>5} {'mean px':>9} {'YES rate':>9} {'edge':>9} {'p':>9}")
            for lo, hi in BANDS:
                b = [r for r in g if lo <= r[key] < hi]
                if len(b) < 25:
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
