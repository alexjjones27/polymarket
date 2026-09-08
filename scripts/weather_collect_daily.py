"""Daily collector for the temperature markets: forecast, market prices, and outcome.

The one-day calibration says the naive version of this strategy does not work. Free
grid-point forecasts run 0.65-1.90C cold against the stations these markets resolve
on, and after removing each model's own mean bias the residual sd is still 1.31-1.65C
against brackets that are 1.0C wide. A model whose error spans more than one bin
cannot pick the bin.

What that single day CANNOT settle is whether PER-CITY calibration rescues it. With
one observation per city, a city's fixed offset is inseparable from the day's random
error. Taipei was 3.19C cold and Tokyo 1.03C warm; if those are stable siting offsets
they are correctable, and if they are noise they are not. Distinguishing the two needs
weeks of daily observations per city, and nothing else will do it.

So this logs, every day, for every city market:
  the forecast from several models, at a fixed lead before the market's date
  the full 11-bracket market price vector at that moment
  and, once available, the bracket that actually resolved

After a few weeks that dataset answers three questions that matter and cannot be
answered now: is each city's bias stable enough to remove, is the residual then
smaller than a bracket, and -- the real question -- does the market's own price beat
the calibrated forecast anyway. If the market wins, the strategy is dead regardless of
how good the calibration gets.

Costs nothing and risks nothing: it places no orders and reads only public endpoints.
"""
import csv
import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import polymarket_final_pct as pmf
from weather_edge import (CITIES, find_temperature_events, parse_city_date,
                          parse_bracket, market_brackets, resolved_bracket, UA)

MODELS = ["best_match", "ecmwf_ifs025", "gfs_seamless", "icon_seamless"]
OUT_DIR = REPO / "results" / "weather"
OBS = OUT_DIR / "observations.csv"
LOG = OUT_DIR / "run_log.txt"

FIELDS = ["captured_utc", "city", "market_date", "slug", "settled",
          "resolved_lo", "resolved_hi",
          "best_match", "ecmwf_ifs025", "gfs_seamless", "icon_seamless",
          "ens_mean", "ens_sd", "ens_lo", "ens_hi", "ens_n",
          "market_dist", "market_mode", "market_liquidity"]


def log(msg):
    line = f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC  {msg}"
    print(line, flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def get_json(url, timeout=45):
    return json.load(urllib.request.urlopen(
        urllib.request.Request(url, headers=UA), timeout=timeout))


def model_fc(lat, lon, tz, date_iso, model):
    url = ("https://api.open-meteo.com/v1/forecast"
           f"?latitude={lat}&longitude={lon}&daily=temperature_2m_max"
           f"&timezone={tz.replace('/', '%2F')}&start_date={date_iso}&end_date={date_iso}"
           f"&models={model}")
    try:
        d = get_json(url)
    except Exception:
        return None
    for k, v in (d.get("daily") or {}).items():
        if k.startswith("temperature_2m_max") and isinstance(v, list) and v and v[0] is not None:
            return round(float(v[0]), 2)
    return None


def ensemble(lat, lon, tz, date_iso):
    url = ("https://ensemble-api.open-meteo.com/v1/ensemble"
           f"?latitude={lat}&longitude={lon}&daily=temperature_2m_max"
           f"&models=gfs025&timezone={tz.replace('/', '%2F')}"
           f"&start_date={date_iso}&end_date={date_iso}")
    try:
        d = get_json(url)
    except Exception:
        return []
    out = []
    for k, v in (d.get("daily") or {}).items():
        if k.startswith("temperature_2m_max") and isinstance(v, list) and v and v[0] is not None:
            out.append(float(v[0]))
    return out


def already_have(city, date_iso):
    if not OBS.exists():
        return set()
    keys = set()
    with open(OBS, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            keys.add((r["city"], r["market_date"], r["settled"]))
    return keys


def handle(ev, existing):
    title = ev.get("title") or ""
    city, _ = parse_city_date(title)
    if city not in CITIES:
        return None
    lat, lon, tz = CITIES[city]
    try:
        end = datetime.fromisoformat((ev.get("endDate") or "").replace("Z", "+00:00"))
    except Exception:
        return None
    date_iso = end.date().isoformat()
    truth = resolved_bracket(ev)
    settled = "True" if truth else "False"
    if (city, date_iso, settled) in existing:
        return None

    brs = market_brackets(ev)
    dist = []
    for b in brs:
        lo, hi = b["br"]
        tag = f"<={hi}" if lo is None else (f">={lo}" if hi is None else str(lo))
        px = b["mid"] if b["mid"] is not None else b["last"]
        dist.append(f"{tag}:{px if px is not None else ''}")
    mode = None
    best = -1.0
    for b in brs:
        px = b["mid"] if b["mid"] is not None else (b["last"] or 0)
        if px and px > best:
            lo, hi = b["br"]
            best = px
            mode = f"<={hi}" if lo is None else (f">={lo}" if hi is None else str(lo))

    mem = ensemble(lat, lon, tz, date_iso)
    import statistics as st
    rec = {
        "captured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "city": city, "market_date": date_iso, "slug": ev.get("slug"),
        "settled": settled,
        "resolved_lo": truth[0] if truth else "", "resolved_hi": truth[1] if truth else "",
        "ens_mean": round(st.mean(mem), 2) if mem else "",
        "ens_sd": round(st.pstdev(mem), 3) if len(mem) > 1 else "",
        "ens_lo": round(min(mem), 2) if mem else "",
        "ens_hi": round(max(mem), 2) if mem else "",
        "ens_n": len(mem),
        "market_dist": "|".join(dist), "market_mode": mode or "",
        "market_liquidity": ev.get("liquidity"),
    }
    for m in MODELS:
        rec[m] = model_fc(lat, lon, tz, date_iso, m)
    return rec


def main():
    log("weather collector starting (read-only; places no orders)")
    evs = find_temperature_events("false") + find_temperature_events("true")
    log(f"{len(evs)} temperature events discovered")
    existing = already_have(None, None)

    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(handle, e, existing) for e in evs]
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception as ex:
                log(f"  row failed: {str(ex)[:60]}")
                continue
            if r:
                rows.append(r)

    if not rows:
        log("nothing new to record")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    new = not OBS.exists()
    with open(OBS, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    s = sum(1 for r in rows if r["settled"] == "True")
    log(f"recorded {len(rows)} rows ({s} settled with a known outcome, "
        f"{len(rows)-s} still open) -> {OBS.name}")


if __name__ == "__main__":
    main()
