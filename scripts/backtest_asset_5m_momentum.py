"""Generic version of backtest_btc_5m_momentum.py / backtest_btc_5m_momentum_oos.py
/ the threshold-sweep analysis, parameterized by asset series slug, so the
SAME validation rigor applied to BTC (in-sample discovery + disjoint
out-of-sample confirmation + fee-adjusted net edge) can be applied to any
other 5-minute up/down market before trading it -- the edge should NOT be
assumed to transfer just because the market mechanics are identical.

Usage: python scripts/backtest_asset_5m_momentum.py eth
       python scripts/backtest_asset_5m_momentum.py sol
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import polymarket_final_pct as pmf

SECS_BEFORE = 30
THRESHOLDS = [0.85, 0.88, 0.90, 0.92, 0.93, 0.94, 0.95]
IN_SAMPLE_DAYS = (0, 14)
OOS_DAYS = (14, 28)


def fetch_events(series_slug: str, day_start: int, day_end: int, cache_name: str) -> list[dict]:
    cache_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "polymarket" / cache_name
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if cached.get("day_start") == day_start and cached.get("day_end") == day_end:
            print(f"  using cached {cache_name} ({len(cached['events'])} events)")
            return cached["events"]

    now = pmf.pd.Timestamp.utcnow().floor("min")
    events = []
    for day_offset in range(day_start, day_end):
        day_end_ts = now - pmf.pd.Timedelta(days=day_offset)
        day_start_ts = day_end_ts - pmf.pd.Timedelta(days=1)
        offset = 0
        while True:
            page = pmf._get(pmf.GAMMA_BASE, "/events", {
                "series_slug": series_slug, "closed": "true", "limit": 100, "offset": offset,
                "end_date_min": day_start_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end_date_max": day_end_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            })
            if not page:
                break
            events.extend(page)
            if len(page) < 100:
                break
            offset += 100
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"day_start": day_start, "day_end": day_end, "events": events}))
    print(f"  fetched {len(events)} events for {series_slug} days {day_start}-{day_end}")
    return events


def analyze_one(event: dict) -> dict | None:
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
    df = pmf.fetch_price_series(token_ids[0], end_s - 3600, end_s + 60, fidelity=1)
    if df.empty:
        return None
    check_s = end_s - SECS_BEFORE
    prior = df[df["t"] <= check_s]
    if prior.empty:
        return None
    if check_s - int(prior.iloc[-1]["t"]) > max(60, SECS_BEFORE):
        return None
    p_up = float(prior.iloc[-1]["p"])
    if p_up >= 0.5:
        leading_price, won = p_up, up_resolved
    else:
        leading_price, won = 1.0 - p_up, not up_resolved
    return {"leading_price": leading_price, "won": won}


def analyze_all(events: list[dict]) -> "pmf.pd.DataFrame":
    rows = []
    with ThreadPoolExecutor(max_workers=24) as pool:
        futures = {pool.submit(analyze_one, e): e for e in events}
        for i, fut in enumerate(as_completed(futures)):
            r = fut.result()
            if r:
                rows.append(r)
    return pmf.pd.DataFrame(rows)


def fee_pct_of_notional(price: float) -> float:
    return 0.07 * (1 - price)


def main():
    asset = sys.argv[1].lower()
    series_slug = f"{asset}-up-or-down-5m"
    print(f"=== {asset.upper()} 5-minute momentum edge: in-sample vs out-of-sample, T-{SECS_BEFORE}s ===")

    print("Fetching in-sample window (days 0-14) ...")
    is_events = fetch_events(series_slug, *IN_SAMPLE_DAYS, f"{asset}_5m_events.json")
    print("Fetching out-of-sample window (days 14-28) ...")
    oos_events = fetch_events(series_slug, *OOS_DAYS, f"{asset}_5m_events_oos.json")

    print("Analyzing in-sample ...")
    is_df = analyze_all(is_events)
    print("Analyzing out-of-sample ...")
    oos_df = analyze_all(oos_events)
    print(f"in-sample n={len(is_df)}, OOS n={len(oos_df)}\n")

    print(f"{'threshold':>10} | {'IS n':>6} {'IS edge':>8} {'IS net':>8} | {'OOS n':>6} {'OOS edge':>9} {'OOS net':>8}")
    for thr in THRESHOLDS:
        row = []
        for df in [is_df, oos_df]:
            g = df[df["leading_price"] >= thr]
            n = len(g)
            if n == 0:
                row.append((0, float("nan"), float("nan")))
                continue
            wr = g["won"].mean()
            implied = g["leading_price"].mean()
            edge = wr - implied
            fee = g["leading_price"].apply(fee_pct_of_notional).mean()
            row.append((n, edge, edge - fee))
        (n1, e1, net1), (n2, e2, net2) = row
        print(f"{thr:>10.2f} | {n1:>6} {e1:>+8.4f} {net1:>+8.4f} | {n2:>6} {e2:>+9.4f} {net2:>+8.4f}")

    out_dir = Path(__file__).resolve().parents[1] / "results" / "btc_5m_momentum"
    out_dir.mkdir(parents=True, exist_ok=True)
    is_df.to_csv(out_dir / f"{asset}_momentum_insample.csv", index=False)
    oos_df.to_csv(out_dir / f"{asset}_momentum_oos.csv", index=False)
    print(f"\nSaved to {out_dir}/{asset}_momentum_insample.csv and _oos.csv")


if __name__ == "__main__":
    main()
