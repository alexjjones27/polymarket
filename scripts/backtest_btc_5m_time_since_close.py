"""Tests the refined hypothesis: win rate depends on TIME REMAINING at entry
(proxied by secs_after_close -- more negative/smaller = confirmed earlier
relative to nominal close, more time left before true settlement for a
reversal to develop; larger positive = confirmed later, less time left),
not just the raw confirmed price level. Reuses the same 300-event samples
and sustained-crossing detection as backtest_btc_5m_price_bucket.py, this
time capturing secs_after_close alongside price.
"""
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import polymarket_final_pct as pmf
import backtest_btc_5m_sustained as sust

THRESHOLD = 0.90
SUSTAIN_S = 5


def run_sample(cache_name: str, label: str) -> list[dict]:
    path = Path(__file__).resolve().parents[1] / "data" / "raw" / "polymarket" / cache_name
    events = json.loads(path.read_text())["events"]
    random.seed(42)
    sample = random.sample(events, min(300, len(events)))

    prepared = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(sust.prepare_event, e): e for e in sample}
        for fut in as_completed(futures):
            r = fut.result()
            if r:
                prepared.append(r)

    results = []
    for p in prepared:
        r = sust.find_sustained_crossing(p, THRESHOLD, SUSTAIN_S)
        if r:
            results.append({"sample": label, "price": r["price"], "won": r["won"],
                             "secs_after_close": r["secs_after_close"]})
    print(f"{label}: {len(results)} confirmed crossings")
    return results


def main():
    all_results = run_sample("btc_5m_events.json", "in_sample") + \
                  run_sample("btc_5m_events_oos.json", "oos")

    df = pmf.pd.DataFrame(all_results)
    out = Path(__file__).resolve().parents[1] / "results" / "btc_5m_momentum" / "time_since_close_observations.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} observations to {out}\n")

    buckets = [-95, -30, 0, 30, 60, 90, 120, 180, 305]
    df["bucket"] = pmf.pd.cut(df["secs_after_close"], buckets)
    print("=== Win rate by secs_after_close (time relative to nominal close) ===")
    print(f"{'bucket':>20} {'n':>6} {'win_rate':>9} {'avg_price':>10}")
    for b, g in df.groupby("bucket", observed=True):
        if len(g) == 0:
            continue
        print(f"{str(b):>20} {len(g):>6} {g['won'].mean():>9.4f} {g['price'].mean():>10.4f}")

    print("\n=== Split at median secs_after_close ===")
    median = df["secs_after_close"].median()
    early = df[df["secs_after_close"] <= median]
    late = df[df["secs_after_close"] > median]
    print(f"  <= {median:.0f}s (n={len(early)}): win_rate={early['won'].mean():.4f}, avg_price={early['price'].mean():.4f}")
    print(f"  >  {median:.0f}s (n={len(late)}): win_rate={late['won'].mean():.4f}, avg_price={late['price'].mean():.4f}")

    print("\n=== Which matters more: controlling for price, does secs_after_close still predict win rate? ===")
    # crude control: within the 0.90-0.94 price band (where losses concentrated), split by time
    narrow = df[(df["price"] >= 0.90) & (df["price"] < 0.94)]
    med2 = narrow["secs_after_close"].median() if len(narrow) else None
    if med2 is not None:
        e2 = narrow[narrow["secs_after_close"] <= med2]
        l2 = narrow[narrow["secs_after_close"] > med2]
        print(f"  within price 0.90-0.94: early half (n={len(e2)}) win_rate={e2['won'].mean():.4f}, "
              f"late half (n={len(l2)}) win_rate={l2['won'].mean():.4f}")


if __name__ == "__main__":
    main()
