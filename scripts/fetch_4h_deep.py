"""Extends the 4h history as far back as the series goes.

The hour-08 reversion rests on 183 BTC observations, which is the binding weakness:
p=0.0068 becomes 0.041 once corrected for having searched six hours, and the two
chronological halves show 64.8% then 58.7%, which could be decay or could be noise.
More history separates those without waiting months for forward data.

The previous fetch stopped at 1,100 windows (~183 days) because it was capped, not
because the data ran out. This probes much further back and merges into the existing
caches, keeping whatever already exists.
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(r"D:\Finance\polymarket")
sys.path.insert(0, str(REPO / "src"))
import polymarket_final_pct as pmf

SECS = 14400
MAX_BACK = 3200          # ~1.4 years
MISS_LIMIT = 120         # generous: gaps in the series should not end the scan


def fetch(asset):
    path = REPO / "data" / "raw" / "polymarket" / f"{asset}_updown_4h_extended.json"
    existing = {}
    if path.exists():
        for e in json.loads(path.read_text())["events"]:
            try:
                existing[e["slug"]] = e
            except Exception:
                continue
    print(f"{asset}: starting from {len(existing)} cached", flush=True)

    now = int(time.time())
    base = (now // SECS) * SECS
    miss = 0
    added = 0
    for i in range(1, MAX_BACK):
        slug = f"{asset}-updown-4h-{base - SECS*i}"
        if slug in existing:
            miss = 0
            continue
        try:
            r = pmf._get(pmf.GAMMA_BASE, "/events", {"slug": slug})
        except Exception:
            miss += 1
            if miss > MISS_LIMIT:
                break
            continue
        if not r:
            miss += 1
            if miss > MISS_LIMIT:
                print(f"  {asset}: series ends at ~{i} slots back", flush=True)
                break
            continue
        miss = 0
        e = r[0]
        m = e["markets"][0] if e.get("markets") else None
        if m and m.get("closed"):
            existing[slug] = e
            added += 1
            if added % 200 == 0:
                print(f"  {asset}: +{added} (total {len(existing)})", flush=True)

    evs = sorted(existing.values(), key=lambda e: e.get("endDate", ""))
    path.write_text(json.dumps({"events": evs}))
    print(f"{asset}: {len(evs)} total windows (+{added} new)", flush=True)


if __name__ == "__main__":
    for a in ("btc", "eth"):
        fetch(a)
