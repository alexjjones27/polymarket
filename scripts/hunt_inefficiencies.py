"""Broad sweep for inefficiencies never tested, using data already cached.

Everything tried so far concerned the price path INSIDE a window. These test
properties ACROSS windows and across the book, which is a different search space
entirely, and most of it costs no API calls at all.

1. SERIAL DEPENDENCE   does one window's outcome predict the next? These resolve on
                       whether BTC rose over the interval, and consecutive intervals
                       are adjacent slices of one price series, so momentum or
                       reversal at the window level is physically plausible. If it
                       exists and the market does not price it, that is exploitable
                       with no latency sensitivity whatsoever.

2. RUNS / STREAKS      after k consecutive ups, is P(up) still 50%? A gambler's-
                       fallacy or hot-hand bias in the crowd would show here.

3. HOUR OF DAY         we tested day-of-week and found nothing. Hour was never
                       tested, and BTC volatility is strongly diurnal.

4. BASE RATE           is the Up rate actually 50%? A persistent drift would be the
                       simplest edge of all.

5. BOOK ARBITRAGE      ask(Up) + ask(Down) must be >= 1.00 or it is free money. This
                       is the most basic check in any binary market and I have never
                       run it. Uses the collected snapshots.

Nothing here depends on reacting quickly, which is what killed every previous
candidate.
"""
import json
import statistics as st
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from scipy import stats as sps

REPO = Path(r"D:\Finance\polymarket")
CACHES = [
    ("btc_5m_events", "5m"), ("btc_5m_events_oos", "5m-oos"),
    ("btc_updown_15m_events", "15m"), ("btc_updown_15m_events_oos", "15m-oos"),
    ("btc_updown_4h_events", "4h"), ("btc_updown_4h_events_oos", "4h-oos"),
]


def load_outcomes(cache):
    p = REPO / "data" / "raw" / "polymarket" / f"{cache}.json"
    if not p.exists():
        return []
    out = []
    for e in json.loads(p.read_text())["events"]:
        try:
            m = e["markets"][0]
            pr = m.get("outcomePrices")
            pr = json.loads(pr) if isinstance(pr, str) else pr
            if not m.get("closed") or not pr or len(pr) != 2:
                continue
            end = datetime.fromisoformat(e["endDate"].replace("Z", "+00:00"))
            out.append({"end": end, "up": float(pr[0]) == 1.0})
        except Exception:
            continue
    out.sort(key=lambda x: x["end"])
    return out


def main():
    print("=" * 88)
    print("1-4.  CROSS-WINDOW TESTS ON RESOLUTION DATA (no API calls)")
    print("=" * 88)
    print(f"\n  {'series':>10} {'n':>6} {'up_rate':>9} {'p vs 50%':>10} "
          f"{'P(up|prev up)':>14} {'P(up|prev dn)':>14} {'serial p':>10}")
    allseries = {}
    for cache, lab in CACHES:
        o = load_outcomes(cache)
        if len(o) < 100:
            continue
        allseries[lab] = o
        n = len(o)
        ups = sum(1 for x in o if x["up"])
        p_base = sps.binomtest(ups, n, 0.5).pvalue
        # serial: consecutive only (windows are contiguous in these caches)
        a = b = c = d = 0
        for i in range(1, n):
            if o[i-1]["up"]:
                if o[i]["up"]: a += 1
                else: b += 1
            else:
                if o[i]["up"]: c += 1
                else: d += 1
        pu_pu = a / (a + b) if a + b else float("nan")
        pu_pd = c / (c + d) if c + d else float("nan")
        _, ser_p = sps.fisher_exact([[a, b], [c, d]])
        flag = "  <--" if ser_p < 0.05 or p_base < 0.05 else ""
        print(f"  {lab:>10} {n:>6} {ups/n:>8.2%} {p_base:>10.4f} "
              f"{pu_pu:>13.2%} {pu_pd:>13.2%} {ser_p:>10.4f}{flag}")

    print("\n" + "=" * 88)
    print("2.  STREAKS -- is P(up) still 50% after k consecutive ups?")
    print("=" * 88)
    for lab, o in allseries.items():
        if len(o) < 500:
            continue
        print(f"\n  {lab} (n={len(o)})")
        print(f"  {'after k ups':>13} {'n':>6} {'P(next up)':>12} {'p':>9}")
        for k in (1, 2, 3, 4, 5):
            nxt = []
            run = 0
            for x in o:
                if run >= k:
                    nxt.append(x["up"])
                run = run + 1 if x["up"] else 0
            if len(nxt) < 40:
                continue
            u = sum(nxt)
            pv = sps.binomtest(u, len(nxt), 0.5).pvalue
            f = "  <--" if pv < 0.05 else ""
            print(f"  {k:>13} {len(nxt):>6} {u/len(nxt):>11.2%} {pv:>9.4f}{f}")

    print("\n" + "=" * 88)
    print("3.  HOUR OF DAY (UTC) -- never tested; BTC vol is strongly diurnal")
    print("=" * 88)
    for lab in ("5m", "15m"):
        o = allseries.get(lab)
        if not o:
            continue
        byh = defaultdict(list)
        for x in o:
            byh[x["end"].astimezone(timezone.utc).hour].append(x["up"])
        print(f"\n  {lab}:  (Bonferroni alpha for 24 tests = {0.05/24:.4f})")
        hits = []
        for h in sorted(byh):
            v = byh[h]
            if len(v) < 60:
                continue
            u = sum(v)
            pv = sps.binomtest(u, len(v), 0.5).pvalue
            if pv < 0.05:
                hits.append((h, len(v), u / len(v), pv))
        if hits:
            for h, n, r, pv in hits:
                sig = "SURVIVES BONFERRONI" if pv < 0.05 / 24 else "nominal only"
                print(f"    hour {h:02d}:00  n={n:>5}  up_rate={r:>6.2%}  p={pv:.4f}  {sig}")
        else:
            print("    no hour reaches even nominal significance")

    print("\n" + "=" * 88)
    print("5.  BOOK ARBITRAGE -- does ask(Up) + ask(Down) ever fall below 1.00?")
    print("=" * 88)
    snaps = defaultdict(dict)
    d = REPO / "data" / "raw" / "polymarket" / "book_snapshots"
    for f in sorted(d.glob("*.jsonl")):
        for line in f.open():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not r.get("asks") or not r.get("bids"):
                continue
            try:
                a = float(r["asks"][0][0]); b = float(r["bids"][0][0])
            except (TypeError, ValueError, IndexError):
                continue
            key = (r["window_end"], round(r["ts"]))
            snaps[key][r["side"]] = {"ask": a, "bid": b}
    paired = [v for v in snaps.values() if "Up" in v and "Down" in v]
    print(f"  {len(paired)} simultaneous Up/Down book observations")
    if paired:
        asum = [v["Up"]["ask"] + v["Down"]["ask"] for v in paired]
        bsum = [v["Up"]["bid"] + v["Down"]["bid"] for v in paired]
        arb_buy = [x for x in asum if x < 1.0]
        arb_sell = [x for x in bsum if x > 1.0]
        print(f"    ask(Up)+ask(Down): mean {st.mean(asum):.4f}  min {min(asum):.4f}")
        print(f"      below 1.00 (buy both cheap): {len(arb_buy)} "
              f"= {len(arb_buy)/len(asum):.2%}")
        print(f"    bid(Up)+bid(Down): mean {st.mean(bsum):.4f}  max {max(bsum):.4f}")
        print(f"      above 1.00 (sell both rich): {len(arb_sell)} "
              f"= {len(arb_sell)/len(bsum):.2%}")
        if arb_buy:
            print(f"      best buy-side arb: {min(asum):.4f} "
                  f"-> {(1.0-min(asum))*100:.2f}c per pair before fees")
        if arb_sell:
            print(f"      best sell-side arb: {max(bsum):.4f} "
                  f"-> {(max(bsum)-1.0)*100:.2f}c per pair before fees")


if __name__ == "__main__":
    main()
