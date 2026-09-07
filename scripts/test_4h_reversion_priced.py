"""Is the 4h mean reversion already in the price?

Pooled over 720 windows, BTC 4h up/down shows clear reversion: after an up window
the next is up 44.13% of the time, after a down window 55.12%. Fisher p=0.0036, and
it replicates independently in both halves (p=0.0446, p=0.0261, same direction). It
survives Bonferroni for the six series tested.

That is a real property of the price series. It is only an EDGE if the market fails
to price it. If traders know, the quote at the start of a window already leans --
"Up" would trade near 0.55 after a down window rather than 0.50 -- and there is
nothing to take.

So: for each window, look up the previous window's outcome, read the market price
early in the current window, and compare that price to the realised frequency in
that conditional bucket. The gap between them, if any, is the edge.

This is the decisive test, and unlike everything else in this project it carries no
latency requirement at all -- the previous window's outcome is known hours before
the next one opens, so an order can be placed at leisure.
"""
import json
import random
import statistics as st
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from scipy import stats as sps

REPO = Path(r"D:\Finance\polymarket")
sys.path.insert(0, str(REPO / "src"))
import polymarket_final_pct as pmf

WINDOW = 14400
OFFSETS = [-14000, -13500, -12600, -10800, -7200]   # early in the window
SAMPLE = 500


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
    tr = fetch_trades(cond)
    ser = []
    for t in sorted(tr or [], key=lambda x: x["timestamp"]):
        try:
            p = float(t["price"])
        except (TypeError, ValueError):
            continue
        rel = t["timestamp"] - end_s
        if t.get("outcome") == "Up":
            ser.append((rel, p))
        elif t.get("outcome") == "Down":
            ser.append((rel, 1.0 - p))
    return {"end_s": end_s, "up": float(pr[0]) == 1.0, "ser": ser}


def price_at(ser, off):
    last = None
    for rel, p in ser:
        if rel <= off:
            last = p
        else:
            break
    return last


def main():
    evs = []
    for c in ("btc_updown_4h_events", "btc_updown_4h_events_oos"):
        p = REPO / "data" / "raw" / "polymarket" / f"{c}.json"
        if p.exists():
            evs.extend(json.loads(p.read_text())["events"])
    # de-dupe and order
    seen = {}
    for e in evs:
        try:
            seen[e["slug"]] = e
        except Exception:
            continue
    evs = sorted(seen.values(), key=lambda e: e.get("endDate", ""))
    print(f"{len(evs)} unique 4h windows")

    random.seed(42)
    keep = set(range(len(evs)))
    if len(evs) > SAMPLE:
        keep = set(random.sample(range(1, len(evs)), SAMPLE))  # never index 0, needs a prior
    todo = sorted({i for i in keep} | {i - 1 for i in keep if i > 0})
    print(f"fetching trades for {len(todo)} windows (sampled + their predecessors)...",
          flush=True)

    built = {}
    with ThreadPoolExecutor(max_workers=12) as pool:
        futs = {pool.submit(build, evs[i]): i for i in todo}
        done = 0
        for f in as_completed(futs):
            i = futs[f]
            done += 1
            try:
                r = f.result()
            except Exception:
                continue
            if r:
                built[i] = r
            if done % 150 == 0:
                print(f"  {done}/{len(todo)}", flush=True)

    print("\n" + "=" * 92)
    print("IS THE REVERSION PRICED IN?  market price vs realised rate, by prior outcome")
    print("=" * 92)
    for off in OFFSETS:
        rows = []
        for i in sorted(keep):
            cur, prev = built.get(i), built.get(i - 1)
            if not cur or not prev:
                continue
            px = price_at(cur["ser"], off)
            if px is None or not (0.02 < px < 0.98):
                continue
            rows.append({"prev_up": prev["up"], "px": px, "up": cur["up"]})
        if len(rows) < 60:
            continue
        print(f"\n  at window_open+{WINDOW+off}s  (close{off}s)   n={len(rows)}")
        print(f"  {'prior':>12} {'n':>5} {'mean price(Up)':>16} {'actual up-rate':>16} "
              f"{'gap':>9} {'p':>9}")
        for lab, sel in (("after UP", True), ("after DOWN", False)):
            g = [r for r in rows if r["prev_up"] is sel]
            if len(g) < 25:
                continue
            mp = st.mean(r["px"] for r in g)
            act = sum(1 for r in g if r["up"]) / len(g)
            pv = sps.binomtest(sum(1 for r in g if r["up"]), len(g),
                               min(0.999, max(0.001, mp))).pvalue
            flag = "  <-- MISPRICED" if pv < 0.05 else ""
            print(f"  {lab:>12} {len(g):>5} {mp:>16.4f} {act:>16.4f} "
                  f"{act-mp:>+9.4f} {pv:>9.4f}{flag}")


if __name__ == "__main__":
    main()
