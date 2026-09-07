"""Builds historical event caches for the longer-horizon BTC up/down series.

The 5-minute market was closed out for structural reasons: its available edge
(1.3-1.7c/share) barely cleared the 0.5c half-spread plus fee, the microstructure
signal decayed inside 2s against our 155ms latency, and market making needed a
rebate 3x the taker fee.

The 15m and 4h series share the SAME 0.010 spread -- it is the tick floor, not a
risk price -- while giving 3x and 48x the horizon. That changes the ratio of edge to
cost without requiring anything new to be true.

15m is the primary target: 96 instances/day means ~2,700 a month, enough for the
disjoint in-sample/OOS split that caught every false positive in the 5m work. 4h has
better structure but only ~180/month, too thin to validate the same way.

Caches are written in the same shape as the 5m ones so every existing analysis
script can be pointed at them unchanged.
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(r"D:\Finance\polymarket")
sys.path.insert(0, str(REPO / "src"))
import polymarket_final_pct as pmf

OUT = REPO / "data" / "raw" / "polymarket"
SERIES = [("btc-updown-15m", 900, 30), ("btc-updown-4h", 14400, 120)]


def fetch(prefix, secs, days):
    now = int(time.time())
    base = (now // secs) * secs
    n = int(days * 86400 / secs)
    print(f"{prefix}: probing {n} slots back ({days}d)", flush=True)
    events, misses = [], 0
    for i in range(1, n + 1):
        slug = f"{prefix}-{base - secs*i}"
        try:
            r = pmf._get(pmf.GAMMA_BASE, "/events", {"slug": slug})
        except Exception:
            misses += 1
            if misses > 40:
                break
            continue
        if not r:
            misses += 1
            if misses > 40:
                print(f"  stopping: {misses} consecutive misses", flush=True)
                break
            continue
        misses = 0
        e = r[0]
        m = e["markets"][0] if e.get("markets") else None
        if not m or not m.get("closed"):
            continue
        events.append(e)
        if len(events) % 200 == 0:
            print(f"  {len(events)} resolved events", flush=True)
    return events


def main():
    for prefix, secs, days in SERIES:
        evs = fetch(prefix, secs, days)
        if not evs:
            print(f"{prefix}: nothing fetched\n")
            continue
        # split in half chronologically: in-sample = older, oos = newer
        evs.sort(key=lambda e: e.get("endDate", ""))
        mid = len(evs) // 2
        name = prefix.replace("-", "_")
        (OUT / f"{name}_events.json").write_text(json.dumps({"events": evs[:mid]}))
        (OUT / f"{name}_events_oos.json").write_text(json.dumps({"events": evs[mid:]}))
        print(f"{prefix}: {len(evs)} resolved -> {mid} in-sample / {len(evs)-mid} oos\n")


if __name__ == "__main__":
    main()
