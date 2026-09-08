"""Systematic study of time-of-day effects, across every timeframe and asset held.

The hour-08 result came from searching six hours on one series. That is a thin basis,
and the multiple-comparison correction (p=0.0068 -> 0.041) barely survives. This does
the search properly and, more importantly, sets up an independent check the earlier
work could not.

Two distinct hypotheses are tested, and they are not the same thing:

  BASE RATE     is any hour biased up or down? A persistent drift would be the
                simplest edge available, and needs no conditioning at all.

  REVERSION     does the previous window's outcome predict this one, within an hour?
                This is where hour-08 showed +23.5% on the 4h series.

The independent check: the 4h hour-08 window covers 04:00-08:00 UTC. If that period
is genuinely special, the 15m and 5m contracts covering the SAME wall-clock hours
should show it too. Those are different contracts with different traders and vastly
more observations -- 2,880 and 8,056 windows against 183 -- so they are a real test
rather than a restatement. If 04:00-08:00 is unremarkable there, the 4h finding is
probably a six-way multiple-comparison artifact.

Every p-value is reported alongside its Bonferroni threshold for the number of tests
actually run in that family, because the whole point of the exercise is not to fool
ourselves twice.
"""
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from scipy import stats as sps

REPO = Path(r"D:\Finance\polymarket")

SERIES = [
    ("BTC 5m", ["btc_5m_events", "btc_5m_events_oos"], 300),
    ("BTC 15m", ["btc_updown_15m_events", "btc_updown_15m_events_oos"], 900),
    ("BTC 4h", ["btc_updown_4h_extended"], 14400),
    ("ETH 4h", ["eth_updown_4h_extended"], 14400),
    ("SOL 4h", ["sol_updown_4h_extended"], 14400),
]
ASIAN = (4, 5, 6, 7)   # wall-clock hours covered by the 4h hour-08 window


def load(caches):
    seen = {}
    for c in caches:
        p = REPO / "data" / "raw" / "polymarket" / f"{c}.json"
        if not p.exists():
            continue
        for e in json.loads(p.read_text())["events"]:
            try:
                m = e["markets"][0]
                pr = m.get("outcomePrices")
                pr = json.loads(pr) if isinstance(pr, str) else pr
                if not m.get("closed") or len(pr) != 2:
                    continue
                dt = datetime.fromisoformat(e["endDate"].replace("Z", "+00:00"))
                seen[e["slug"]] = (dt, float(pr[0]) == 1.0)
            except Exception:
                continue
    return sorted(seen.values())


def rev_stats(pairs):
    """pairs: list of (prev_up, this_up)."""
    a = b = c = d = 0
    for prev, cur in pairs:
        if prev:
            a, b = (a + 1, b) if cur else (a, b + 1)
        else:
            c, d = (c + 1, d) if cur else (c, d + 1)
    if a + b < 10 or c + d < 10:
        return None
    _, p = sps.fisher_exact([[a, b], [c, d]])
    return {"n": a + b + c + d, "after_up": a / (a + b), "after_dn": c / (c + d),
            "spread": c / (c + d) - a / (a + b), "p": p}


def main():
    all_series = {}
    for name, caches, wlen in SERIES:
        rows = load(caches)
        if len(rows) < 300:
            print(f"{name}: only {len(rows)} windows, skipping")
            continue
        all_series[name] = rows
        print(f"{name}: {len(rows)} windows")

    # ---------------- BASE RATE BY HOUR ----------------
    print("\n" + "=" * 96)
    print("1. BASE RATE BY HOUR -- is any hour biased up or down?")
    print("=" * 96)
    for name, rows in all_series.items():
        byh = defaultdict(list)
        for dt, up in rows:
            byh[dt.astimezone(timezone.utc).hour].append(up)
        tested = [h for h in byh if len(byh[h]) >= 60]
        alpha = 0.05 / max(1, len(tested))
        hits = []
        for h in sorted(tested):
            v = byh[h]
            k = sum(v)
            p = sps.binomtest(k, len(v), 0.5).pvalue
            if p < 0.05:
                hits.append((h, len(v), k / len(v), p, p < alpha))
        print(f"\n  {name}  ({len(tested)} hours tested, Bonferroni alpha={alpha:.4f})")
        if not hits:
            print("    no hour reaches even nominal significance")
        for h, n, r, p, surv in hits:
            print(f"    hour {h:02d}  n={n:>5}  up_rate={r:>6.2%}  p={p:.4f}  "
                  f"{'SURVIVES Bonferroni' if surv else 'nominal only'}")

    # ---------------- REVERSION BY HOUR ----------------
    print("\n" + "=" * 96)
    print("2. REVERSION BY HOUR -- does the prior window predict, within an hour?")
    print("=" * 96)
    for name, rows in all_series.items():
        byh = defaultdict(list)
        for i in range(1, len(rows)):
            h = rows[i][0].astimezone(timezone.utc).hour
            byh[h].append((rows[i-1][1], rows[i][1]))
        tested = [h for h in byh if len(byh[h]) >= 60]
        alpha = 0.05 / max(1, len(tested))
        print(f"\n  {name}  ({len(tested)} hours tested, Bonferroni alpha={alpha:.4f})")
        hits = []
        for h in sorted(tested):
            s = rev_stats(byh[h])
            if s and s["p"] < 0.05:
                hits.append((h, s))
        if not hits:
            print("    no hour reaches even nominal significance")
        for h, s in hits:
            print(f"    hour {h:02d}  n={s['n']:>5}  after-up {s['after_up']:>6.2%}  "
                  f"after-down {s['after_dn']:>6.2%}  spread {s['spread']:>+7.2%}  "
                  f"p={s['p']:.4f}  {'SURVIVES Bonferroni' if s['p'] < alpha else 'nominal only'}")

    # ---------------- THE INDEPENDENT CHECK ----------------
    print("\n" + "=" * 96)
    print("3. THE REAL TEST: is 04:00-08:00 UTC special in the FINER timeframes?")
    print("=" * 96)
    print("  the 4h hour-08 window covers these wall-clock hours. 15m and 5m contracts")
    print("  over the same period are different markets with far more observations, so")
    print("  they can confirm or refute the 4h finding rather than restate it.\n")
    print(f"  {'series':>10} {'period':>16} {'n':>6} {'after-up':>10} {'after-down':>11} "
          f"{'spread':>8} {'p':>8}")
    for name, rows in all_series.items():
        for lab, hours in (("04:00-08:00", ASIAN),
                           ("all other hours", tuple(h for h in range(24) if h not in ASIAN))):
            pairs = [(rows[i-1][1], rows[i][1]) for i in range(1, len(rows))
                     if rows[i][0].astimezone(timezone.utc).hour in hours]
            s = rev_stats(pairs)
            if not s:
                continue
            flag = "  <--" if s["p"] < 0.05 else ""
            print(f"  {name:>10} {lab:>16} {s['n']:>6} {s['after_up']:>9.2%} "
                  f"{s['after_dn']:>10.2%} {s['spread']:>+8.2%} {s['p']:>8.4f}{flag}")


if __name__ == "__main__":
    main()
