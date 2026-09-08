"""Prices Polymarket daily-temperature markets against a free numerical ensemble.

The market asks which integer-Celsius bracket will contain a city's daily maximum,
resolved from a named meteorological authority (e.g. the Hong Kong Observatory's
"Absolute Daily Max" to one decimal). There are ~20 such markets a day across world
cities, 11 brackets each, with $47k-$385k of liquidity.

This is the one strategy left that needs no speed and no forecasting skill: Open-Meteo
publishes a 31-member GFS ensemble for free, which yields a probability distribution
over exactly the quantity the market is pricing. If the two distributions disagree by
more than costs, that is a trade.

The obvious failure mode, and the reason this file exists before any trading does, is
that a raw ensemble at a grid point is NOT the station the market resolves on:

  under-dispersion  ensembles are habitually overconfident; taking 31 members as the
                    true distribution would manufacture edge out of nothing
  station basis     a grid cell is not the Hong Kong Observatory's thermometer.
                    Urban heat island, elevation and siting all bias it, and the bias
                    is per-city and systematic rather than random.

So this measures the bias before it trades on anything. Settled markets reveal the
true bracket -- the winning outcome sits at ~0.999 -- so today's already-resolved
markets are free ground truth for calibrating today's forecast, no waiting required.
"""
import json
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import polymarket_final_pct as pmf

# city -> (lat, lon, tz). Coordinates aim at the official observatory/airport where
# the resolution source sits, not the city centroid, since that is what is measured.
CITIES = {
    "hong kong": (22.302, 114.174, "Asia/Hong_Kong"),
    "beijing": (39.933, 116.283, "Asia/Shanghai"),
    "wuhan": (30.606, 114.058, "Asia/Shanghai"),
    "shanghai": (31.400, 121.467, "Asia/Shanghai"),
    "guangzhou": (23.167, 113.333, "Asia/Shanghai"),
    "chongqing": (29.583, 106.467, "Asia/Shanghai"),
    "qingdao": (36.067, 120.333, "Asia/Shanghai"),
    "shenzhen": (22.533, 114.100, "Asia/Shanghai"),
    "seoul": (37.571, 126.966, "Asia/Seoul"),
    "seoul (incheon)": (37.477, 126.625, "Asia/Seoul"),
    "busan": (35.104, 129.032, "Asia/Seoul"),
    "tokyo": (35.692, 139.750, "Asia/Tokyo"),
    "taipei": (25.038, 121.507, "Asia/Taipei"),
    "singapore": (1.359, 103.989, "Asia/Singapore"),
    "kuala lumpur": (3.122, 101.674, "Asia/Kuala_Lumpur"),
    "manila": (14.509, 121.019, "Asia/Manila"),
    "karachi": (24.906, 67.161, "Asia/Karachi"),
    "moscow": (55.756, 37.618, "Europe/Moscow"),
    "london": (51.479, -0.449, "Europe/London"),
    "paris": (48.727, 2.359, "Europe/Paris"),
    "munich": (48.354, 11.786, "Europe/Berlin"),
    "nyc": (40.779, -73.969, "America/New_York"),
    "miami": (25.791, -80.316, "America/New_York"),
    "atlanta": (33.630, -84.442, "America/New_York"),
    "panama city": (8.973, -79.556, "America/Panama"),
    "cape town": (-33.965, 18.602, "Africa/Johannesburg"),
    "wellington": (-41.327, 174.805, "Pacific/Auckland"),
}

UA = {"User-Agent": "polymarket-weather-research/1.0"}


def get_json(url, timeout=40):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                            timeout=timeout))


def parse_city_date(title):
    m = re.search(r"temperature in (.+?) on ([A-Z][a-z]+ \d+)", title)
    if not m:
        return None, None
    city = m.group(1).strip().lower()
    return city, m.group(2)


def parse_bracket(label):
    """'27°C or below' -> (None, 27); '28°C' -> (28, 28); '37°C or higher' -> (37, None)."""
    s = label.replace("°", "").replace("C", "").strip()
    nums = re.findall(r"-?\d+", s)
    if not nums:
        return None
    v = int(nums[0])
    low = "below" in label.lower() or "under" in label.lower() or "or less" in label.lower()
    high = "higher" in label.lower() or "above" in label.lower() or "or more" in label.lower()
    if low:
        return (None, v)
    if high:
        return (v, None)
    return (v, v)


def bracket_contains(br, temp):
    """Brackets are integer bins on a one-decimal reading: 28 means 28.0 <= t < 29.0."""
    lo, hi = br
    t = int(temp // 1)
    if lo is None:
        return t <= hi
    if hi is None:
        return t >= lo
    return t == lo


def ensemble_members(lat, lon, tz, date_iso):
    """31-member GFS ensemble daily max for one date. Returns a list of temperatures."""
    url = ("https://ensemble-api.open-meteo.com/v1/ensemble"
           f"?latitude={lat}&longitude={lon}&daily=temperature_2m_max"
           f"&models=gfs025&timezone={tz.replace('/', '%2F')}"
           f"&start_date={date_iso}&end_date={date_iso}")
    d = get_json(url)
    daily = d.get("daily", {})
    out = []
    for k, v in daily.items():
        if k.startswith("temperature_2m_max") and isinstance(v, list) and v:
            if v[0] is not None:
                out.append(float(v[0]))
    return out


def deterministic_max(lat, lon, tz, date_iso):
    url = ("https://api.open-meteo.com/v1/forecast"
           f"?latitude={lat}&longitude={lon}&daily=temperature_2m_max"
           f"&timezone={tz.replace('/', '%2F')}&start_date={date_iso}&end_date={date_iso}")
    d = get_json(url)
    v = (d.get("daily", {}).get("temperature_2m_max") or [None])[0]
    return float(v) if v is not None else None


def market_brackets(ev):
    out = []
    for m in ev.get("markets") or []:
        label = m.get("groupItemTitle") or m.get("question") or ""
        br = parse_bracket(label)
        if br is None:
            continue
        try:
            bb = m.get("bestBid")
            ba = m.get("bestAsk")
            bb = float(bb) if bb is not None else None
            ba = float(ba) if ba is not None else None
        except (TypeError, ValueError):
            bb = ba = None
        mid = None
        if bb is not None and ba is not None:
            mid = (bb + ba) / 2.0
        elif ba is not None:
            mid = ba
        elif bb is not None:
            mid = bb
        try:
            last = float(m.get("lastTradePrice")) if m.get("lastTradePrice") is not None else None
        except (TypeError, ValueError):
            last = None
        out.append({"label": label, "br": br, "bid": bb, "ask": ba,
                    "mid": mid, "last": last,
                    "tokens": pmf._safe_json_list(m.get("clobTokenIds")),
                    "fee": float((m.get("feeSchedule") or {}).get("rate", 0.0) or 0.0)})
    return out


def resolved_bracket(ev):
    """For a settled market, the winning bracket (price pinned near 1)."""
    best, bestp = None, 0.0
    for m in ev.get("markets") or []:
        try:
            p = m.get("outcomePrices")
            p = json.loads(p) if isinstance(p, str) else p
            v = float(p[0]) if p and len(p) == 2 else None
        except Exception:
            v = None
        if v is None:
            try:
                v = float(m.get("lastTradePrice"))
            except (TypeError, ValueError):
                v = None
        if v is not None and v > bestp:
            bestp, best = v, parse_bracket(m.get("groupItemTitle") or "")
    return best if bestp > 0.9 else None


def find_temperature_events(closed):
    seen = {}
    for order in ("volume", "liquidity"):
        for pg in range(30):
            try:
                b = pmf._get(pmf.GAMMA_BASE, "/events",
                             {"closed": closed, "limit": 40, "offset": pg * 40,
                              "order": order, "ascending": "false"})
            except Exception:
                break
            if not b:
                break
            for e in b:
                if re.search(r"highest temperature", e.get("title") or "", re.I):
                    seen[e.get("slug")] = e
    return list(seen.values())
