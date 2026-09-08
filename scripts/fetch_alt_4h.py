"""Fetches ETH and SOL 4h up/down history, to test the BTC reversion independently.

BTC 4h shows serial reversion: after an up window the next is up 46.14%, after a down
window 55.17%, spread +9.03%, p=0.0031 over 1,100 windows, same direction in all
three chronological thirds. Within that, windows closing at 08:00 UTC show a much
larger +23.53% spread which survives a split (+29.2% then +17.8%) and has a
plausible mechanism -- that window covers 04:00-08:00 UTC, the Asian session ahead of
the European open.

Neither claim can be settled on BTC alone. The subgroup was selected by searching six
hours on the same data, and a multiple-comparison check puts P(some hour looks this
extreme by chance) at 0.27. ETH and SOL are genuinely independent samples: if the
same reversion, and especially the same hour-08 concentration, appears in all three
assets, it is a property of crypto 4h windows rather than a BTC-specific artifact.
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(r"D:\Finance\polymarket")
sys.path.insert(0, str(REPO / "src"))
import polymarket_final_pct as pmf

SECS = 14400
MAX_BACK = 1600


def fetch(asset):
    now = int(time.time())
    base = (now // SECS) * SECS
    evs, miss = [], 0
    for i in range(1, MAX_BACK):
        try:
            r = pmf._get(pmf.GAMMA_BASE, "/events",
                         {"slug": f"{asset}-updown-4h-{base - SECS*i}"})
        except Exception:
            miss += 1
            if miss > 60:
                break
            continue
        if not r:
            miss += 1
            if miss > 60:
                break
            continue
        miss = 0
        e = r[0]
        m = e["markets"][0] if e.get("markets") else None
        if m and m.get("closed"):
            evs.append(e)
        if len(evs) and len(evs) % 200 == 0:
            print(f"  {asset}: {len(evs)}", flush=True)
    evs.sort(key=lambda e: e.get("endDate", ""))
    out = REPO / "data" / "raw" / "polymarket" / f"{asset}_updown_4h_extended.json"
    out.write_text(json.dumps({"events": evs}))
    print(f"{asset}: {len(evs)} resolved windows -> {out.name}", flush=True)
    return len(evs)


if __name__ == "__main__":
    for a in ("eth", "sol"):
        fetch(a)
