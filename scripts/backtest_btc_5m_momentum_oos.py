"""Out-of-sample confirmation of the univariate BTC 95%+/<=90s finding
from backtest_btc_5m_sweep.py (this is the ONE finding from today's
session that hasn't been OOS-tested yet -- the cross-asset one was, and
failed). Reuses the disjoint days-14-to-28 BTC event cache and price
series already fetched for backtest_btc_eth_5m_oos.py -- no new network
calls needed, this is a pure re-analysis."""
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import polymarket_final_pct as pmf
import backtest_btc_5m_momentum as base

OOS_CACHE = Path(__file__).resolve().parents[1] / "data" / "raw" / "polymarket" / "btc_5m_events_oos.json"
CHECK_TIMES_S = [30, 60, 90, 120, 150]


def main():
    cached = json.loads(OOS_CACHE.read_text())
    events = cached["events"]
    print(f"OOS window: days {cached['day_start']}-{cached['day_end']} back, {len(events)} events")

    rows = []
    with ThreadPoolExecutor(max_workers=24) as pool:
        futures = {}
        for e in events:
            for secs in CHECK_TIMES_S:
                futures[pool.submit(analyze_one, e, secs)] = (e, secs)
        for fut in as_completed(futures):
            r = fut.result()
            if r:
                rows.append(r)

    df = pmf.pd.DataFrame(rows)
    print(f"{len(df)} total observations\n")
    print(f"{'secs_before':>11} {'n(>=0.95)':>10} {'win_rate':>9} {'implied':>8} {'edge':>8} {'z':>6}")
    for secs in CHECK_TIMES_S:
        g = df[(df["secs_before"] == secs) & (df["leading_price"] >= 0.95)]
        n = len(g)
        if n == 0:
            print(f"{secs:>11} {n:>10} -- no observations")
            continue
        wins = g["won"].sum()
        wr = wins / n
        se = (wr * (1 - wr) / n) ** 0.5 if n > 1 else float("nan")
        z = (wr - 0.975) / se if se > 0 else float("nan")
        print(f"{secs:>11} {n:>10} {wr:>9.3f} {0.975:>8.3f} {wr-0.975:>+8.3f} {z:>6.2f}")

    out = Path(__file__).resolve().parents[1] / "results" / "btc_5m_momentum" / "momentum_oos_observations.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")


def analyze_one(event, secs_before):
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
    if check_s - int(prior.iloc[-1]["t"]) > max(60, secs_before):
        return None
    p_up = float(prior.iloc[-1]["p"])
    if p_up >= 0.5:
        leading_price, won = p_up, up_resolved
    else:
        leading_price, won = 1.0 - p_up, not up_resolved
    return {"secs_before": secs_before, "leading_price": leading_price, "won": won}


if __name__ == "__main__":
    main()
