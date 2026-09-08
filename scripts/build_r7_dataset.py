"""Builds a reusable dataset for stress-testing R7 (long-shots under-priced).

The first pass found sub-20c bands resolving YES more often than their price implies
(ALL [0.00,0.05): +0.0149, p=0.0074; POLITICS [0.10,0.20): +0.2673, p=0.0000). That
result carries one serious known weakness and several unknowns, and this dataset is
built to attack all of them rather than to confirm the finding.

The known weakness is INDEPENDENCE. Legs of the same negRisk event are strongly
negatively correlated -- exactly one resolves YES -- so sampling 20 legs of one
election gives 20 observations carrying roughly one event's worth of information, and
every p-value computed across them is optimistic. Recording event_id and
n_outcomes_in_event allows clustering, and recording whether a market is a STANDALONE
binary (its event has exactly one market) isolates a subsample with no clustering
problem at all.

The unknowns are stability and tradeability, so this also records the resolution
date (for time splits), the category, the sampled price at several fractions of the
market's trading life, and the trade count and volume, since a result that only
exists in thinly-traded markets is not one you can act on.

Saved to CSV so the analyses can be re-run without re-fetching.
"""
import csv
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import polymarket_final_pct as pmf

PAGES = 200
PER_PAGE = 40
SAMPLE = 4000
MIN_VOLUME = 2000.0
FRACS = [0.10, 0.25, 0.40, 0.60]
OUT = REPO / "data" / "raw" / "polymarket" / "r7_dataset.csv"

CATS = {
    "crypto": ("crypto", "bitcoin", "ethereum", "solana", "btc", "eth", "xrp"),
    "politics": ("politics", "election", "trump", "senate", "congress", "president",
                 "geopolitics", "cabinet", "governor"),
    "sports": ("sports", "nba", "nfl", "mlb", "soccer", "football", "epl", "ufc",
               "tennis", "cricket", "golf", "hockey"),
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
    ev_id, ev_slug, cat, n_out, neg, m = job
    try:
        pr = m.get("outcomePrices")
        pr = json.loads(pr) if isinstance(pr, str) else pr
        if not m.get("closed") or not pr or len(pr) != 2:
            return None
        yes_won = float(pr[0]) == 1.0
        cond = m.get("conditionId")
        if not cond:
            return None
        vol = float(m.get("volumeNum") or m.get("volume") or 0)
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
    t0, t1 = rows[0][0], rows[-1][0]
    span = t1 - t0
    if span < 3600:
        return None
    rec = {"event_id": ev_id, "event_slug": ev_slug, "cat": cat,
           "n_outcomes": n_out, "neg_risk": bool(neg),
           "standalone": n_out == 1, "yes": int(yes_won),
           "volume": round(vol, 2), "n_trades": len(rows),
           "span_h": round(span / 3600.0, 2),
           "closed_utc": datetime.fromtimestamp(t1, timezone.utc).isoformat()}
    for f in FRACS:
        cutoff = t0 + f * span
        prior = [p for ts, p in rows if ts <= cutoff]
        rec[f"p{int(f*100)}"] = round(prior[-1], 4) if len(prior) >= 3 else ""
    return rec


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
        if pg % 40 == 0 and pg:
            print(f"  {len(evs)} events ...", flush=True)
    print(f"{len(evs)} resolved events")

    jobs = []
    for e in evs:
        ms = [m for m in (e.get("markets") or []) if m.get("closed")]
        if not ms:
            continue
        cat = categorise(e)
        for m in ms:
            try:
                if float(m.get("volumeNum") or m.get("volume") or 0) < MIN_VOLUME:
                    continue
            except (TypeError, ValueError):
                continue
            jobs.append((e.get("id"), e.get("slug"), cat, len(ms), e.get("negRisk"), m))
    random.seed(42)
    random.shuffle(jobs)
    jobs = jobs[:SAMPLE]
    print(f"{len(jobs)} markets queued; fetching ...", flush=True)

    out = []
    with ThreadPoolExecutor(max_workers=16) as pool:
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
            if done % 500 == 0:
                print(f"  {done}/{len(jobs)} ({len(out)} usable)", flush=True)

    if not out:
        print("nothing collected")
        return
    cols = list(out[0].keys())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out)
    n_events = len({r["event_slug"] for r in out})
    n_alone = sum(1 for r in out if r["standalone"])
    print(f"\nwrote {len(out)} markets across {n_events} events -> {OUT.name}")
    print(f"  standalone binaries (no clustering problem): {n_alone}")


if __name__ == "__main__":
    main()
