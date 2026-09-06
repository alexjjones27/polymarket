"""Direct comparison of PRICE_THRESHOLD=0.90 vs 0.94 as INDEPENDENT signals
(each threshold's own candidate-crossing-and-sustain detection, not just
filtering the other's confirmed prices), on the same 600-event samples
(300 in-sample + 300 OOS) -- trade data fetched once per event and reused
across both thresholds to avoid double the API calls.

Translates results into actual expected dollar profit/hour at the current
$8 stake, so "which is more profitable" has a concrete answer, not just a
net-edge percentage.
"""
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import polymarket_final_pct as pmf
import backtest_btc_5m_sustained as sust

SUSTAIN_S = 5
THRESHOLDS = [0.90, 0.92, 0.94]
STAKE_USD = 8.0
TOTAL_WINDOWS_PER_SAMPLE = 300  # each sample covers 300 randomly-drawn 5-min windows


def fee_pct(price: float) -> float:
    return 0.07 * (1 - price)


def prepare_sample(cache_name: str, label: str) -> list[dict]:
    path = Path(__file__).resolve().parents[1] / "data" / "raw" / "polymarket" / cache_name
    events = json.loads(path.read_text())["events"]
    random.seed(42)
    sample = random.sample(events, min(300, len(events)))
    print(f"{label}: fetching trades for {len(sample)} events (shared across all thresholds) ...")
    prepared = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(sust.prepare_event, e): e for e in sample}
        for fut in as_completed(futures):
            r = fut.result()
            if r:
                prepared.append(r)
    return prepared


def main():
    is_prepared = prepare_sample("btc_5m_events.json", "in_sample")
    oos_prepared = prepare_sample("btc_5m_events_oos.json", "oos")

    print(f"\n{'threshold':>10} | {'sample':>10} {'n_qualify':>10} {'win_rate':>9} {'net_edge':>9} "
          f"{'trades/hr':>10} {'$/hr':>8} {'$/day':>9}")
    for thr in THRESHOLDS:
        for label, prepared in [("in_sample", is_prepared), ("oos", oos_prepared)]:
            results = []
            for p in prepared:
                r = sust.find_sustained_crossing(p, thr, SUSTAIN_S)
                if r:
                    results.append(r)
            n = len(results)
            qualify_rate = n / TOTAL_WINDOWS_PER_SAMPLE
            if n == 0:
                print(f"{thr:>10.2f} | {label:>10} {'0':>10} {'--':>9} {'--':>9} {'--':>10} {'--':>8} {'--':>9}")
                continue
            wins = sum(1 for r in results if r["won"])
            win_rate = wins / n
            avg_price = sum(r["price"] for r in results) / n
            avg_fee = sum(fee_pct(r["price"]) for r in results) / n
            net_edge = (win_rate - avg_price) - avg_fee

            trades_per_hour = qualify_rate * 12  # 12 five-min windows per hour
            profit_per_trade = STAKE_USD * net_edge
            usd_per_hour = trades_per_hour * profit_per_trade
            usd_per_day = usd_per_hour * 24

            print(f"{thr:>10.2f} | {label:>10} {n:>10} {win_rate:>9.4f} {net_edge:>+9.4f} "
                  f"{trades_per_hour:>10.2f} {usd_per_hour:>+8.2f} {usd_per_day:>+9.2f}")


if __name__ == "__main__":
    main()
