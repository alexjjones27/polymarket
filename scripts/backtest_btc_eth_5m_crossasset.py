"""Tests the user's observed pattern ("BTC at 50%, ETH at 40% in the same
5-min window -- surely that's arbitrage") properly. It is NOT riskless
arbitrage: BTC and ETH are different assets, each resolving independently
on its own price path, so there is no model-free relationship forcing
their probabilities to match (unlike, say, Yes+No on the same event). The
real, testable question is whether a BTC-ETH probability spread carries
incremental information -- does the lower/higher-priced side tend to
catch up (spread closes) or diverge further, beyond what each asset's own
price already implies?

Method: for matched windows (same window_end_epoch exists in both the
btc-up-or-down-5m and eth-up-or-down-5m series -- confirmed to share the
same 5-minute grid), at each check-time compute p_btc_up and p_eth_up,
bucket by spread = p_btc_up - p_eth_up, and compare each asset's realized
win rate against its OWN implied probability within each spread bucket.
If a wide spread predicts a deviation from the asset's own price (e.g.
ETH outperforms its own 40% when BTC is at 50%, meaning ETH "catches up"
toward BTC), that is a real, exploitable lead-lag effect. If not, the two
markets are just pricing two different assets independently and
correctly, and the observed gaps are exactly what a well-calibrated
market for two imperfectly-correlated assets should look like.
"""
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import polymarket_final_pct as pmf
import backtest_btc_5m_momentum as base

ETH_SERIES_SLUG = "eth-up-or-down-5m"
ETH_RAW_CACHE = Path(__file__).resolve().parents[1] / "data" / "raw" / "polymarket" / "eth_5m_events.json"
CHECK_TIMES_S = [150, 90, 60, 30]
SPREAD_BUCKET_WIDTH = 0.05
MIN_BUCKET_N = 50


def fetch_eth_events(lookback_days: int) -> list[dict]:
    import json
    if ETH_RAW_CACHE.exists():
        cached = json.loads(ETH_RAW_CACHE.read_text())
        if cached.get("lookback_days") == lookback_days:
            print(f"Using cached ETH event list ({len(cached['events'])} events)")
            return cached["events"]

    now = pmf.pd.Timestamp.utcnow().floor("min")
    events = []
    for day_offset in range(lookback_days):
        day_end = now - pmf.pd.Timedelta(days=day_offset)
        day_start = day_end - pmf.pd.Timedelta(days=1)
        offset = 0
        while True:
            page = pmf._get(pmf.GAMMA_BASE, "/events", {
                "series_slug": ETH_SERIES_SLUG, "closed": "true", "limit": 100, "offset": offset,
                "end_date_min": day_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end_date_max": day_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            })
            if not page:
                break
            events.extend(page)
            if len(page) < 100:
                break
            offset += 100
        print(f"  ETH day -{day_offset}: {len(events)} events so far", flush=True)

    ETH_RAW_CACHE.parent.mkdir(parents=True, exist_ok=True)
    ETH_RAW_CACHE.write_text(json.dumps({"lookback_days": lookback_days, "events": events}))
    return events


def event_price_and_outcome(event: dict, secs_before: int) -> dict | None:
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
    check_s = end_s - secs_before
    prior = df[df["t"] <= check_s]
    if prior.empty:
        return None
    staleness = check_s - int(prior.iloc[-1]["t"])
    if staleness > max(60, secs_before):
        return None
    return {"window_end": end_s, "p_up": float(prior.iloc[-1]["p"]), "up_resolved": up_resolved}


def main():
    btc_events = base.fetch_all_events(base.LOOKBACK_DAYS)
    eth_events = fetch_eth_events(base.LOOKBACK_DAYS)
    print(f"BTC: {len(btc_events)} events, ETH: {len(eth_events)} events")

    for secs_before in CHECK_TIMES_S:
        print(f"\n=== check time: T-{secs_before}s ===")
        with ThreadPoolExecutor(max_workers=24) as pool:
            btc_futs = {pool.submit(event_price_and_outcome, e, secs_before): e for e in btc_events}
            btc_by_window = {}
            for fut in as_completed(btc_futs):
                r = fut.result()
                if r:
                    btc_by_window[r["window_end"]] = r

            eth_futs = {pool.submit(event_price_and_outcome, e, secs_before): e for e in eth_events}
            eth_by_window = {}
            for fut in as_completed(eth_futs):
                r = fut.result()
                if r:
                    eth_by_window[r["window_end"]] = r

        matched = []
        for w, b in btc_by_window.items():
            e = eth_by_window.get(w)
            if e:
                matched.append({
                    "window_end": w, "p_btc_up": b["p_up"], "p_eth_up": e["p_up"],
                    "btc_up_resolved": b["up_resolved"], "eth_up_resolved": e["up_resolved"],
                    "spread": b["p_up"] - e["p_up"],
                })
        df = pmf.pd.DataFrame(matched)
        print(f"{len(df)} matched (BTC, ETH) windows")
        if df.empty:
            continue

        corr = df["p_btc_up"].corr(df["p_eth_up"])
        outcome_corr = df["btc_up_resolved"].astype(int).corr(df["eth_up_resolved"].astype(int))
        print(f"corr(p_btc_up, p_eth_up) = {corr:.3f}   corr(btc_outcome, eth_outcome) = {outcome_corr:.3f}")

        df["spread_bucket"] = (df["spread"] / SPREAD_BUCKET_WIDTH).round() * SPREAD_BUCKET_WIDTH
        print(f"\n{'spread_bucket':>13} {'n':>5} | {'p_eth_up_avg':>12} {'eth_win_rate':>12} {'eth_edge':>9} | "
              f"{'p_btc_up_avg':>12} {'btc_win_rate':>12} {'btc_edge':>9}")
        for bucket, g in df.groupby("spread_bucket"):
            n = len(g)
            if n < MIN_BUCKET_N:
                continue
            eth_implied = g["p_eth_up"].mean()
            eth_actual = g["eth_up_resolved"].mean()
            btc_implied = g["p_btc_up"].mean()
            btc_actual = g["btc_up_resolved"].mean()
            print(f"{bucket:>+13.2f} {n:>5} | {eth_implied:>12.3f} {eth_actual:>12.3f} "
                  f"{eth_actual-eth_implied:>+9.3f} | {btc_implied:>12.3f} {btc_actual:>12.3f} "
                  f"{btc_actual-btc_implied:>+9.3f}")

        # Direct test of the user's literal example: BTC near 50%, ETH near 40% (or the mirror)
        lit = df[((df["p_btc_up"].between(0.45, 0.55)) & (df["p_eth_up"].between(0.35, 0.45))) |
                 ((df["p_btc_up"].between(0.45, 0.55)) & (df["p_eth_up"].between(0.55, 0.65)))]
        if len(lit) >= 20:
            eth_implied = lit["p_eth_up"].mean()
            eth_actual = lit["eth_up_resolved"].mean()
            n = len(lit)
            se = (eth_actual * (1 - eth_actual) / n) ** 0.5
            print(f"\nLiteral example (BTC~50%, ETH~40% or ~60%): n={n}, ETH implied={eth_implied:.3f}, "
                  f"ETH actual win rate={eth_actual:.3f}, edge={eth_actual-eth_implied:+.3f}, "
                  f"95% CI=[{eth_actual-1.96*se:.3f}, {eth_actual+1.96*se:.3f}]")

        out_dir = Path(__file__).resolve().parents[1] / "results" / "btc_5m_momentum"
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / f"btc_eth_crossasset_t{secs_before}.csv", index=False)


if __name__ == "__main__":
    main()
