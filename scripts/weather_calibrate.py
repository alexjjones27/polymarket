"""Measures the forecast's bias against markets that have ALREADY resolved.

Before any edge can be claimed, one question has to be answered: does a free GFS
ensemble at a grid point actually predict the specific thermometer these markets
resolve on? A grid cell is not the Hong Kong Observatory, and the difference is
systematic -- urban siting, elevation, distance to water -- not random noise. If the
model runs 2C cold for Beijing every day, then "the market is wrong about Beijing"
would be a statement about my model, not about the market.

Settled markets give this away for free. The winning bracket sits pinned near 1.00, so
today's already-resolved markets reveal the true bracket, and the same forecast API
can be asked what it thought. No waiting.

Reports per city:
  the resolved bracket, the deterministic forecast, the ensemble mean and spread
  whether the ensemble's modal bracket matched
  whether the truth even fell inside the ensemble's full range -- if it routinely does
    not, the ensemble is under-dispersed and any probability derived from it is
    overconfident, which is the classic way to invent an edge that is not there
"""
import statistics as st
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
from weather_edge import (CITIES, find_temperature_events, parse_city_date,
                          market_brackets, resolved_bracket, ensemble_members,
                          deterministic_max, bracket_contains)


def label_of(br):
    lo, hi = br
    if lo is None:
        return f"<={hi}"
    if hi is None:
        return f">={lo}"
    return str(lo)


def analyse(ev):
    title = ev.get("title") or ""
    city, _ = parse_city_date(title)
    if city not in CITIES:
        return None
    lat, lon, tz = CITIES[city]
    truth = resolved_bracket(ev)
    if truth is None:
        return None
    # the market's own date, taken from its endDate in local time
    try:
        end = datetime.fromisoformat((ev.get("endDate") or "").replace("Z", "+00:00"))
    except Exception:
        return None
    date_iso = end.date().isoformat()
    try:
        mem = ensemble_members(lat, lon, tz, date_iso)
        det = deterministic_max(lat, lon, tz, date_iso)
    except Exception as e:
        return {"city": city, "err": str(e)[:40]}
    if not mem:
        return {"city": city, "err": "no ensemble"}
    modal = None
    brs = [b["br"] for b in market_brackets(ev)]
    best = -1
    for br in brs:
        c = sum(1 for m in mem if bracket_contains(br, m))
        if c > best:
            best, modal = c, br
    inside = any(bracket_contains(truth, m) for m in mem)
    return {"city": city, "date": date_iso, "truth": truth, "det": det,
            "mean": st.mean(mem), "sd": st.pstdev(mem) if len(mem) > 1 else 0.0,
            "lo": min(mem), "hi": max(mem), "n": len(mem),
            "modal": modal, "hit": modal == truth, "inside": inside}


def main():
    print("finding settled temperature markets ...", flush=True)
    evs = find_temperature_events("true") + find_temperature_events("false")
    print(f"{len(evs)} temperature events; testing those already resolved\n")
    out = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = [pool.submit(analyse, e) for e in evs]
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception:
                continue
            if r:
                out.append(r)
    good = [r for r in out if "err" not in r]
    print(f"{len(good)} settled markets with a forecast\n")
    if not good:
        for r in out[:5]:
            print("  ", r)
        return

    print("=" * 104)
    print("FORECAST vs TRUTH on already-resolved markets")
    print("=" * 104)
    print(f"  {'city':>16} {'date':>11} {'truth':>7} {'det':>7} {'ens mean':>9} "
          f"{'sd':>5} {'range':>13} {'modal':>7} {'hit':>5} {'in range':>9}")
    for r in sorted(good, key=lambda x: x["city"]):
        print(f"  {r['city']:>16} {r['date']:>11} {label_of(r['truth']):>7} "
              f"{(f'{r['det']:.1f}' if r['det'] is not None else '-'):>7} "
              f"{r['mean']:>9.1f} {r['sd']:>5.2f} "
              f"{f'{r['lo']:.1f}-{r['hi']:.1f}':>13} {label_of(r['modal']):>7} "
              f"{('YES' if r['hit'] else 'no'):>5} {('yes' if r['inside'] else 'NO'):>9}")

    hits = sum(1 for r in good if r["hit"])
    inside = sum(1 for r in good if r["inside"])
    print(f"\n  modal bracket correct : {hits}/{len(good)} = {hits/len(good):.0%}")
    print(f"  truth inside ensemble : {inside}/{len(good)} = {inside/len(good):.0%}")
    print(f"  mean ensemble spread  : {st.mean(r['sd'] for r in good):.2f} C")

    # bias per city, using the midpoint of the resolved bracket as the observation
    print("\n" + "=" * 104)
    print("SYSTEMATIC BIAS (forecast minus truth); a stable per-city offset is correctable")
    print("=" * 104)
    print(f"  {'city':>16} {'truth mid':>10} {'ens mean':>9} {'bias':>7}")
    biases = []
    for r in sorted(good, key=lambda x: x["city"]):
        lo, hi = r["truth"]
        if lo is None or hi is None:
            continue      # open-ended bracket has no midpoint
        mid = lo + 0.5
        b = r["mean"] - mid
        biases.append(b)
        print(f"  {r['city']:>16} {mid:>10.1f} {r['mean']:>9.1f} {b:>+7.2f}")
    if biases:
        print(f"\n  mean bias {st.mean(biases):+.2f} C, sd {st.pstdev(biases):.2f} C "
              f"(n={len(biases)})")
        print("  a large sd means the error is city-specific, not a single correctable offset")


if __name__ == "__main__":
    main()
