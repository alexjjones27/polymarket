"""Stress-tests R7 (long-shots under-priced) against its own weaknesses.

The naive result -- sub-20c bands resolving YES more often than priced -- rests on
2,596 markets that come from only 480 events, 5.4 legs each. Legs of one negRisk event
are strongly negatively correlated because exactly one of them resolves YES, so those
2,596 rows carry closer to 480 events' worth of independent information and every
p-value computed across markets is optimistic. That is the first thing to fix.

Five tests, ordered so the claim dies early if it is going to:

  1 NAIVE            reproduce the original, as the benchmark to beat
  2 CLUSTER BOOTSTRAP  resample EVENTS with replacement rather than markets, which is
                     the correct inference when observations cluster
  3 ONE LEG PER EVENT  a single randomly chosen leg from each event, repeated over
                     many draws -- independence by construction, at the cost of power
  4 STANDALONE ONLY  markets whose event has exactly one market, so there is no
                     clustering to correct for at all
  5 TIME SPLIT       first half against second half by resolution date

Then, only if it survives: TRADEABILITY. An edge of +1.5pp on a contract priced at
0.012 is meaningless if the spread at that price is 0.002, because the tick is 0.001
and you cross half of it on entry. Reported as edge net of a half-tick and the fee.
"""
import csv
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from scipy import stats as sps

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "raw" / "polymarket" / "r7_dataset.csv"
BANDS = [(0.00, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.35), (0.35, 0.55)]
PRICE_KEY = "p25"
BOOT = 4000
FEE_RATE = 0.04       # politics/event schedule; crypto is 0.07
HALF_TICK = 0.0005    # tick is 0.001 on these markets


def load():
    rows = []
    with open(DATA, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                px = r.get(PRICE_KEY)
                if not px:
                    continue
                rows.append({
                    "ev": r["event_slug"], "cat": r["cat"],
                    "yes": int(r["yes"]), "px": float(px),
                    "standalone": r["standalone"] == "True",
                    "closed": r["closed_utc"], "vol": float(r["volume"]),
                    "n_out": int(r["n_outcomes"]),
                })
            except Exception:
                continue
    return rows


def band_stats(rows, lo, hi):
    b = [r for r in rows if lo <= r["px"] < hi]
    if len(b) < 25:
        return None
    mp = st.mean(r["px"] for r in b)
    k = sum(r["yes"] for r in b)
    return {"n": len(b), "mp": mp, "rate": k / len(b), "edge": k / len(b) - mp, "k": k}


def main():
    rows = load()
    n_ev = len({r["ev"] for r in rows})
    print(f"{len(rows)} markets across {n_ev} events "
          f"({len(rows)/n_ev:.1f} legs per event)")
    print(f"price sampled at {PRICE_KEY} = 25% through trading life\n")

    # ---------- 1. NAIVE ----------
    print("=" * 92)
    print("1. NAIVE (treats every market as independent -- the original result)")
    print("=" * 92)
    print(f"  {'band':>14} {'n':>6} {'price':>8} {'realised':>9} {'edge':>9} {'p':>9}")
    for lo, hi in BANDS:
        s = band_stats(rows, lo, hi)
        if not s:
            continue
        p = sps.binomtest(s["k"], s["n"], min(0.999, max(0.001, s["mp"]))).pvalue
        flag = "  <-- UNDER" if (p < 0.05 and s["edge"] > 0) else ("  <-- OVER" if p < 0.05 else "")
        print(f"  [{lo:.2f},{hi:.2f}) {s['n']:>6} {s['mp']:>8.4f} {s['rate']:>9.4f} "
              f"{s['edge']:>+9.4f} {p:>9.4f}{flag}")

    # ---------- 2. CLUSTER BOOTSTRAP ----------
    print("\n" + "=" * 92)
    print("2. CLUSTER BOOTSTRAP (resamples EVENTS, the correct unit of independence)")
    print("=" * 92)
    by_ev = defaultdict(list)
    for r in rows:
        by_ev[r["ev"]].append(r)
    evs = list(by_ev)
    random.seed(42)
    print(f"  {'band':>14} {'edge':>9} {'95% CI':>22} {'P(edge>0)':>11}")
    for lo, hi in BANDS:
        base = band_stats(rows, lo, hi)
        if not base:
            continue
        draws = []
        for _ in range(BOOT):
            samp = []
            for _ in range(len(evs)):
                samp.extend(by_ev[random.choice(evs)])
            s = band_stats(samp, lo, hi)
            if s:
                draws.append(s["edge"])
        if len(draws) < 100:
            continue
        draws.sort()
        loci = draws[int(0.025 * len(draws))]
        hici = draws[int(0.975 * len(draws))]
        pgt = sum(1 for d in draws if d > 0) / len(draws)
        flag = "  <-- holds" if loci > 0 else ""
        print(f"  [{lo:.2f},{hi:.2f}) {base['edge']:>+9.4f} "
              f"[{loci:>+8.4f},{hici:>+8.4f}] {pgt:>10.1%}{flag}")

    # ---------- 3. ONE LEG PER EVENT ----------
    print("\n" + "=" * 92)
    print("3. ONE RANDOM LEG PER EVENT (independent by construction, 2000 draws)")
    print("=" * 92)
    print(f"  {'band':>14} {'mean n':>7} {'mean edge':>10} {'% draws p<0.05 & +':>20}")
    for lo, hi in BANDS:
        edges, sig, ns = [], 0, []
        for i in range(2000):
            random.seed(1000 + i)
            one = [random.choice(v) for v in by_ev.values()]
            s = band_stats(one, lo, hi)
            if not s:
                continue
            edges.append(s["edge"]); ns.append(s["n"])
            p = sps.binomtest(s["k"], s["n"], min(0.999, max(0.001, s["mp"]))).pvalue
            if p < 0.05 and s["edge"] > 0:
                sig += 1
        if not edges:
            continue
        print(f"  [{lo:.2f},{hi:.2f}) {st.mean(ns):>7.0f} {st.mean(edges):>+10.4f} "
              f"{sig/len(edges):>19.1%}")

    # ---------- 4. STANDALONE ----------
    print("\n" + "=" * 92)
    print("4. STANDALONE BINARIES ONLY (no clustering possible)")
    print("=" * 92)
    alone = [r for r in rows if r["standalone"]]
    print(f"  n={len(alone)}")
    print(f"  {'band':>14} {'n':>6} {'price':>8} {'realised':>9} {'edge':>9} {'p':>9}")
    for lo, hi in BANDS:
        s = band_stats(alone, lo, hi)
        if not s:
            continue
        p = sps.binomtest(s["k"], s["n"], min(0.999, max(0.001, s["mp"]))).pvalue
        print(f"  [{lo:.2f},{hi:.2f}) {s['n']:>6} {s['mp']:>8.4f} {s['rate']:>9.4f} "
              f"{s['edge']:>+9.4f} {p:>9.4f}")

    # ---------- 5. TIME SPLIT ----------
    print("\n" + "=" * 92)
    print("5. TIME SPLIT by resolution date")
    print("=" * 92)
    srt = sorted(rows, key=lambda r: r["closed"])
    half = len(srt) // 2
    for lab, seg in (("first half", srt[:half]), ("second half", srt[half:])):
        print(f"\n  {lab} (n={len(seg)}, to {seg[-1]['closed'][:10]})")
        print(f"  {'band':>14} {'n':>6} {'price':>8} {'realised':>9} {'edge':>9} {'p':>9}")
        for lo, hi in BANDS:
            s = band_stats(seg, lo, hi)
            if not s:
                continue
            p = sps.binomtest(s["k"], s["n"], min(0.999, max(0.001, s["mp"]))).pvalue
            print(f"  [{lo:.2f},{hi:.2f}) {s['n']:>6} {s['mp']:>8.4f} {s['rate']:>9.4f} "
                  f"{s['edge']:>+9.4f} {p:>9.4f}")

    # ---------- TRADEABILITY ----------
    print("\n" + "=" * 92)
    print("TRADEABILITY: does the edge survive the half-tick and the fee?")
    print("=" * 92)
    print(f"  {'band':>14} {'edge':>9} {'half-tick':>10} {'fee':>9} {'net':>9} "
          f"{'net/stake':>10}")
    for lo, hi in BANDS:
        s = band_stats(rows, lo, hi)
        if not s:
            continue
        fee = FEE_RATE * s["mp"] * (1 - s["mp"])
        net = s["edge"] - HALF_TICK - fee
        print(f"  [{lo:.2f},{hi:.2f}) {s['edge']:>+9.4f} {HALF_TICK:>10.4f} "
              f"{fee:>9.4f} {net:>+9.4f} {net/s['mp']:>+9.1%}")


if __name__ == "__main__":
    main()
