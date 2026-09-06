"""Tests the hypothesis from live trading (all 3 early losses landed at the
low end of confirmed entry prices, 0.90-0.92): does win rate at our LIVE
config (threshold=0.90, sustain=5s) actually vary by the CONFIRMED entry
price (the price after the 5s sustain wait, same as what execution pays),
not just by the raw 0.90 threshold used to detect the crossing?

Reuses the exact same sustained-crossing detection as
backtest_btc_5m_sustained.py, on the same 300-event random samples already
validated there (in-sample + OOS), but saves every individual confirmed
price + outcome instead of just aggregate stats, then buckets by price.
"""
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import polymarket_final_pct as pmf
import backtest_btc_5m_continuous as cont
import backtest_btc_5m_sustained as sust

THRESHOLD = 0.90
SUSTAIN_S = 5


def run_sample(cache_name: str, label: str) -> list[dict]:
    path = Path(__file__).resolve().parents[1] / "data" / "raw" / "polymarket" / cache_name
    events = json.loads(path.read_text())["events"]
    random.seed(42)
    sample = random.sample(events, min(300, len(events)))
    print(f"{label}: fetching trades for {len(sample)} events ...")

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
            results.append({"sample": label, "price": r["price"], "won": r["won"]})
    print(f"{label}: {len(results)} confirmed crossings")
    return results


def main():
    all_results = run_sample("btc_5m_events.json", "in_sample") + \
                  run_sample("btc_5m_events_oos.json", "oos")

    df = pmf.pd.DataFrame(all_results)
    out = Path(__file__).resolve().parents[1] / "results" / "btc_5m_momentum" / "price_bucket_observations.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} total observations to {out}\n")

    buckets = [0.90, 0.92, 0.94, 0.96, 0.98, 1.01]
    df["bucket"] = pmf.pd.cut(df["price"], buckets, right=False)
    print(f"{'bucket':>16} {'n':>6} {'win_rate':>9} {'avg_price':>10}")
    for b, g in df.groupby("bucket", observed=True):
        if len(g) == 0:
            continue
        print(f"{str(b):>16} {len(g):>6} {g['won'].mean():>9.4f} {g['price'].mean():>10.4f}")

    print("\nSplit at median price:")
    median = df["price"].median()
    low = df[df["price"] <= median]
    high = df[df["price"] > median]
    print(f"  <= {median:.3f} (n={len(low)}): win_rate={low['won'].mean():.4f}")
    print(f"  >  {median:.3f} (n={len(high)}): win_rate={high['won'].mean():.4f}")


if __name__ == "__main__":
    main()
