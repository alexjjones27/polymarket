"""Is the bias the MODEL, or the approach?

The GFS ensemble ran 1.02C cold against the stations these markets resolve on, with a
city-specific sd of 1.11C, and contained the truth in only 60% of cases despite having
31 members. Both problems have to be fixed before any probability derived from it
means anything.

Two candidate explanations, and they have different remedies:

  the model    GFS at 0.25 degrees is coarse. ECMWF IFS is generally the better
               global model, and a multi-model blend usually beats any single one.
               If the bias mostly disappears with a better model, the approach is
               sound and just needed a better input.

  the geometry a grid cell is not a thermometer. If every model is cold by a similar
               city-specific amount, the error is siting -- urban heat island,
               elevation, distance to water -- and no model choice fixes it. It would
               instead need a per-city offset learned from history, which takes weeks
               of daily observations to estimate.

Compares several Open-Meteo models against the same 15 known outcomes. The question
is not which is most accurate in the abstract, but whether ANY of them is unbiased
enough at these specific stations to price an 11-bracket market.
"""
import json
import statistics as st
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
from weather_edge import (CITIES, find_temperature_events, parse_city_date,
                          resolved_bracket, UA)

MODELS = ["best_match", "ecmwf_ifs025", "gfs_seamless", "icon_seamless", "jma_seamless"]


def fc(lat, lon, tz, date_iso, model):
    url = ("https://api.open-meteo.com/v1/forecast"
           f"?latitude={lat}&longitude={lon}&daily=temperature_2m_max"
           f"&timezone={tz.replace('/', '%2F')}&start_date={date_iso}&end_date={date_iso}"
           f"&models={model}")
    try:
        d = json.load(urllib.request.urlopen(
            urllib.request.Request(url, headers=UA), timeout=40))
    except Exception:
        return None
    daily = d.get("daily", {})
    for k, v in daily.items():
        if k.startswith("temperature_2m_max") and isinstance(v, list) and v and v[0] is not None:
            return float(v[0])
    return None


def job(ev):
    title = ev.get("title") or ""
    city, _ = parse_city_date(title)
    if city not in CITIES:
        return None
    truth = resolved_bracket(ev)
    if truth is None or truth[0] is None or truth[1] is None:
        return None
    lat, lon, tz = CITIES[city]
    try:
        end = datetime.fromisoformat((ev.get("endDate") or "").replace("Z", "+00:00"))
    except Exception:
        return None
    date_iso = end.date().isoformat()
    row = {"city": city, "truth_mid": truth[0] + 0.5, "truth": truth}
    for m in MODELS:
        row[m] = fc(lat, lon, tz, date_iso, m)
    return row


def main():
    evs = find_temperature_events("true") + find_temperature_events("false")
    rows = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futs = [pool.submit(job, e) for e in evs]
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception:
                continue
            if r:
                rows.append(r)
    rows = [r for r in rows if any(r.get(m) is not None for m in MODELS)]
    print(f"{len(rows)} settled markets with a known bracket\n")
    if not rows:
        return

    print("=" * 104)
    print("PER-CITY FORECAST BY MODEL vs TRUTH")
    print("=" * 104)
    hdr = f"  {'city':>16} {'truth':>6}" + "".join(f"{m[:12]:>13}" for m in MODELS)
    print(hdr)
    for r in sorted(rows, key=lambda x: x["city"]):
        line = f"  {r['city']:>16} {r['truth_mid']:>6.1f}"
        for m in MODELS:
            v = r.get(m)
            line += f"{(f'{v:.1f}' if v is not None else '-'):>13}"
        print(line)

    print("\n" + "=" * 104)
    print("BIAS AND ACCURACY BY MODEL")
    print("=" * 104)
    print(f"  {'model':>16} {'n':>4} {'mean bias':>10} {'sd of bias':>11} "
          f"{'mean |err|':>11} {'bracket hit':>12}")
    for m in MODELS:
        errs = [r[m] - r["truth_mid"] for r in rows if r.get(m) is not None]
        if not errs:
            continue
        hit = sum(1 for r in rows if r.get(m) is not None
                  and int(r[m] // 1) == r["truth"][0])
        n = len(errs)
        print(f"  {m:>16} {n:>4} {st.mean(errs):>+10.2f} "
              f"{(st.pstdev(errs) if n > 1 else 0):>11.2f} "
              f"{st.mean(abs(e) for e in errs):>11.2f} {hit}/{n} = {hit/n:>5.0%}")

    print("\n" + "=" * 104)
    print("AFTER REMOVING EACH MODEL'S OWN MEAN BIAS (the best case for a global fix)")
    print("=" * 104)
    print(f"  {'model':>16} {'residual sd':>12} {'bracket hit':>12}")
    for m in MODELS:
        errs = [(r, r[m] - r["truth_mid"]) for r in rows if r.get(m) is not None]
        if len(errs) < 3:
            continue
        b = st.mean(e for _, e in errs)
        hit = sum(1 for r, _ in errs if int((r[m] - b) // 1) == r["truth"][0])
        resid = st.pstdev([e - b for _, e in errs])
        print(f"  {m:>16} {resid:>12.2f} {hit}/{len(errs)} = {hit/len(errs):>5.0%}")
    print("\n  a residual sd near or above 1.0C means the remaining error still spans")
    print("  more than one bracket, so even a perfectly de-biased model cannot pick")
    print("  the right bin reliably -- the market's 11 brackets are 1C wide.")


if __name__ == "__main__":
    main()
