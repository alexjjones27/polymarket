"""Backtest: does buying the leading side of Polymarket's recurring 5-minute
BTC Up/Down market, when it's priced around ~70% with ~2-3 minutes left in
the window, have a real edge -- or does the market already price it fairly
(or even overprice it, i.e. momentum reverses more than the price implies)?

Market mechanics (see https://polymarket.com/crypto/5M): a new market opens
every 5 minutes, resolves "Up" if Chainlink BTC/USD at window-end >= price
at window-start, else "Down". Series slug: btc-up-or-down-5m.

Method: for each historical resolved window, pull ~1-minute-fidelity CLOB
price history for the "Up" token, find whichever side (Up or Down) was
priced in a target band (default 0.65-0.75) at approximately
CHECK_SECONDS_BEFORE_END seconds before the window closed, and record
whether that side actually won. Aggregate per price bucket: realized win
rate vs. the bucket's implied probability is the edge (or lack of one).
This is the same methodology as src/football_favorite_bias.py /
tennis_favorite_bias.py / underdog_bias.py applied to a new market series.

Caveats this v1 does NOT model: bid/ask spread at execution time (real
fill price is worse than the last trade print used here), any platform
fee, and the fact that CLOB prices-history only updates on trades/price
changes -- for very thin windows the "closest prior observation" can be
somewhat stale relative to the exact check timestamp.
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import polymarket_final_pct as pmf

SERIES_SLUG = "btc-up-or-down-5m"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "btc_5m_momentum"
RAW_CACHE = Path(__file__).resolve().parents[1] / "data" / "raw" / "polymarket" / "btc_5m_events.json"

CHECK_SECONDS_BEFORE_END = 150  # "2-3 minutes left" -> check point at T-2.5min
LOOKBACK_DAYS = 14
BUCKET_WIDTH = 0.05  # 5-cent price buckets


def fetch_all_events(lookback_days: int) -> list[dict]:
    """Paginates by day (the series' own offset cap bites well before
    `lookback_days` worth of 5-min instances would fit in one query)."""
    if RAW_CACHE.exists():
        cached = json.loads(RAW_CACHE.read_text())
        if cached.get("lookback_days") == lookback_days:
            print(f"Using cached event list ({len(cached['events'])} events)")
            return cached["events"]

    now = pmf.pd.Timestamp.utcnow().floor("min")
    events = []
    for day_offset in range(lookback_days):
        day_end = now - pmf.pd.Timedelta(days=day_offset)
        day_start = day_end - pmf.pd.Timedelta(days=1)
        offset = 0
        while True:
            page = pmf._get(pmf.GAMMA_BASE, "/events", {
                "series_slug": SERIES_SLUG, "closed": "true", "limit": 100, "offset": offset,
                "end_date_min": day_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end_date_max": day_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            })
            if not page:
                break
            events.extend(page)
            if len(page) < 100:
                break
            offset += 100
        print(f"  day -{day_offset}: {len(events)} events so far", flush=True)

    RAW_CACHE.parent.mkdir(parents=True, exist_ok=True)
    RAW_CACHE.write_text(json.dumps({"lookback_days": lookback_days, "events": events}))
    return events


def analyze_event(event: dict) -> dict | None:
    markets = event.get("markets") or []
    if not markets:
        return None
    m = markets[0]
    outcomes = pmf._safe_json_list(m.get("outcomes"))
    outcome_prices = pmf._safe_json_list(m.get("outcomePrices"))
    token_ids = pmf._safe_json_list(m.get("clobTokenIds"))
    if outcomes != ["Up", "Down"] or len(outcome_prices) != 2 or len(token_ids) != 2:
        return None
    try:
        up_resolved = float(outcome_prices[0]) == 1.0
    except (ValueError, TypeError):
        return None

    end_s = int(pmf.pd.Timestamp(event["endDate"]).timestamp())
    check_s = end_s - CHECK_SECONDS_BEFORE_END

    df = pmf.fetch_price_series(token_ids[0], end_s - 3600, end_s + 60, fidelity=1)
    if df.empty:
        return None
    prior = df[df["t"] <= check_s]
    if prior.empty:
        return None
    p_up_at_check = float(prior.iloc[-1]["p"])
    staleness_s = check_s - int(prior.iloc[-1]["t"])
    if staleness_s > 180:  # observation too old to trust as "the price at T-2.5min"
        return None

    if p_up_at_check >= 0.5:
        leading_side, leading_price, won = "Up", p_up_at_check, up_resolved
    else:
        leading_side, leading_price, won = "Down", 1.0 - p_up_at_check, not up_resolved

    return {
        "slug": event.get("slug"), "end_s": end_s, "leading_side": leading_side,
        "leading_price": leading_price, "won": won, "staleness_s": staleness_s,
    }


def main():
    print(f"Fetching {LOOKBACK_DAYS} days of resolved {SERIES_SLUG} instances ...")
    events = fetch_all_events(LOOKBACK_DAYS)
    print(f"{len(events)} total resolved instances")

    print("Computing leading-side price at T-2.5min for each (fetching CLOB price history, cached) ...")
    rows = []
    with ThreadPoolExecutor(max_workers=24) as pool:
        futures = {pool.submit(analyze_event, e): e for e in events}
        for i, fut in enumerate(as_completed(futures)):
            try:
                r = fut.result()
            except Exception:
                r = None
            if r:
                rows.append(r)
            if (i + 1) % 500 == 0:
                print(f"  processed {i+1}/{len(events)} ...", flush=True)

    print(f"{len(rows)} usable observations (had a fresh-enough price point at T-2.5min)\n")

    df = pmf.pd.DataFrame(rows)
    df["bucket"] = (df["leading_price"] // BUCKET_WIDTH) * BUCKET_WIDTH

    print(f"{'bucket':>12} {'n':>6} {'wins':>6} {'win_rate':>9} {'implied_mid':>12} {'edge':>8} {'95% CI':>18}")
    summary_rows = []
    for bucket, g in df.groupby("bucket"):
        n = len(g)
        wins = int(g["won"].sum())
        win_rate = wins / n
        implied_mid = bucket + BUCKET_WIDTH / 2
        edge = win_rate - implied_mid
        se = (win_rate * (1 - win_rate) / n) ** 0.5 if n > 1 else float("nan")
        ci_lo, ci_hi = win_rate - 1.96 * se, win_rate + 1.96 * se
        print(f"{bucket:>11.2f}+ {n:>6} {wins:>6} {win_rate:>9.3f} {implied_mid:>12.3f} "
              f"{edge:>+8.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")
        summary_rows.append({"bucket": bucket, "n": n, "wins": wins, "win_rate": win_rate,
                              "implied_mid": implied_mid, "edge": edge, "ci_lo": ci_lo, "ci_hi": ci_hi})

    target = df[(df["leading_price"] >= 0.65) & (df["leading_price"] < 0.75)]
    if len(target):
        n, wins = len(target), int(target["won"].sum())
        win_rate = wins / n
        se = (win_rate * (1 - win_rate) / n) ** 0.5
        print(f"\n--- User's hypothesis: leading side priced 65-75% at T-2.5min ---")
        print(f"n={n}, wins={wins}, win_rate={win_rate:.3f}, implied~0.70, "
              f"edge={win_rate - 0.70:+.3f}, 95% CI=[{win_rate-1.96*se:.3f}, {win_rate+1.96*se:.3f}]")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "observations.csv", index=False)
    pmf.pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "bucket_summary.csv", index=False)
    print(f"\nSaved {RESULTS_DIR / 'observations.csv'} and {RESULTS_DIR / 'bucket_summary.csv'}")


if __name__ == "__main__":
    main()
