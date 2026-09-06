"""Backtest a fundamentally different, previously-untested version of the
BTC 5-min signal: instead of checking price at ONE fixed pre-close
snapshot (T-30s, as in backtest_btc_5m_momentum.py), this continuously
walks REAL per-trade data (data-api.polymarket.com/trades, not the
coarse 1-minute-bucketed prices-history) from START_MONITORING_S before
close through END_MONITORING_S after close, and finds the FIRST moment
either side's trade price crosses each threshold.

Discovered via manual inspection: many windows are still a coin-flip at
T-30s but resolve gradually over the following 1-5 minutes AFTER the
nominal close (real, continued trading -- not a data artifact), which
the snapshot-based backtest never examined at all. This tests whether
that gradual post-close price discovery is actually catchable and
profitable, not just the pre-close snapshot.

Uses trade PRINT price as a proxy for achievable execution price (same
simplifying assumption as the snapshot backtest); real execution would
pay the ask, typically a bit worse.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import polymarket_final_pct as pmf

START_MONITORING_S = -90   # start "watching" 90s before nominal close
END_MONITORING_S = 300     # keep watching up to 5 min after close
THRESHOLDS = [0.85, 0.88, 0.90, 0.92, 0.95]


def fetch_all_trades(cond_id: str) -> list[dict]:
    trades = []
    offset = 0
    while True:
        page = pmf._get(pmf.DATA_API_BASE, "/trades", {"market": cond_id, "limit": 500, "offset": offset})
        if not page:
            break
        trades.extend(page)
        if len(page) < 500:
            break
        offset += 500
        if offset > 3000:  # safety cap
            break
    return trades


def find_first_crossing(event: dict, threshold: float) -> dict | None:
    markets = event.get("markets") or []
    if not markets:
        return None
    m = markets[0]
    outcomes = pmf._safe_json_list(m.get("outcomes"))
    outcome_prices = pmf._safe_json_list(m.get("outcomePrices"))
    if outcomes != ["Up", "Down"] or len(outcome_prices) != 2:
        return None
    try:
        up_resolved = float(outcome_prices[0]) == 1.0
    except (ValueError, TypeError):
        return None
    cond_id = m.get("conditionId")
    if not cond_id:
        return None
    end_s = int(pmf.pd.Timestamp(event["endDate"]).timestamp())

    trades = fetch_all_trades(cond_id)
    if not trades:
        return None
    relevant = [t for t in trades
                if START_MONITORING_S <= (t["timestamp"] - end_s) <= END_MONITORING_S]
    relevant.sort(key=lambda t: t["timestamp"])

    for t in relevant:
        try:
            price = float(t["price"])
        except (ValueError, TypeError):
            continue
        side = t.get("outcome")
        if side not in ("Up", "Down"):
            continue
        if price >= threshold:
            won = up_resolved if side == "Up" else (not up_resolved)
            return {"secs_after_close": t["timestamp"] - end_s, "side": side,
                    "price": price, "won": won, "size": t.get("size")}
    return None


def analyze_sample(events: list[dict], threshold: float, max_workers: int = 16) -> list[dict]:
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(find_first_crossing, e, threshold): e for e in events}
        for fut in as_completed(futures):
            r = fut.result()
            if r:
                results.append(r)
    return results


def fee_pct(price: float) -> float:
    return 0.07 * (1 - price)


def main():
    import random
    is_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "polymarket" / "btc_5m_events.json"
    is_events = json.loads(is_path.read_text())["events"]
    random.seed(42)
    sample = random.sample(is_events, min(300, len(is_events)))
    print(f"Pilot: {len(sample)} randomly-sampled in-sample events, thresholds {THRESHOLDS}\n")

    for thr in THRESHOLDS:
        print(f"--- threshold {thr:.2f} ---")
        results = analyze_sample(sample, thr)
        n = len(results)
        if n == 0:
            print("  no crossings found")
            continue
        wins = sum(1 for r in results if r["won"])
        wr = wins / n
        avg_price = sum(r["price"] for r in results) / n
        avg_secs = sum(r["secs_after_close"] for r in results) / n
        fee = fee_pct(avg_price)
        edge = wr - avg_price
        print(f"  n={n}/{len(sample)} windows had a crossing, win_rate={wr:.3f}, avg_entry_price={avg_price:.3f}, "
              f"edge={edge:+.4f}, net={edge-fee:+.4f}, avg_secs_after_close={avg_secs:+.1f}")
        pre = sum(1 for r in results if r["secs_after_close"] < 0)
        print(f"  {pre}/{n} crossings happened BEFORE nominal close, {n-pre}/{n} AFTER")


if __name__ == "__main__":
    main()
