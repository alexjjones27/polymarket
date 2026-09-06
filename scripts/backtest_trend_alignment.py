"""Tests whether betting WITH vs AGAINST the recent multi-hour BTC trend
predicts win rate at our live config (threshold=0.90, sustain=5s) --
motivated by all 4 real live losses being counter-trend Down bets during
an up-trending session.

Trend for a given window = net direction of the prior LOOKBACK_WINDOWS
windows' own resolutions (cheap: uses each window's own outcomePrices,
already in the full cached event lists, no new fetches). Our bet is
"with_trend" if its side matches the majority of those prior resolutions.

Re-runs sustained-crossing detection on the same 300-event random samples
used elsewhere (this DOES require re-fetching trade data, ~minutes), this
time capturing side + window_end so trend context can be joined in.
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
LOOKBACK_OPTIONS = [12, 24, 36]  # windows = 1h, 2h, 3h


def build_resolution_map(cache_name: str) -> dict:
    """window_end -> resolved_up (bool), for every event in the full population."""
    path = Path(__file__).resolve().parents[1] / "data" / "raw" / "polymarket" / cache_name
    events = json.loads(path.read_text())["events"]
    res_map = {}
    for e in events:
        markets = e.get("markets") or []
        if not markets:
            continue
        m = markets[0]
        outcomes = pmf._safe_json_list(m.get("outcomes"))
        outcome_prices = pmf._safe_json_list(m.get("outcomePrices"))
        if outcomes != ["Up", "Down"] or len(outcome_prices) != 2:
            continue
        try:
            resolved_up = float(outcome_prices[0]) == 1.0
        except (ValueError, TypeError):
            continue
        end_s = int(pmf.pd.Timestamp(e["endDate"]).timestamp())
        res_map[end_s] = resolved_up
    return res_map


def trend_at(res_map: dict, window_end: int, lookback: int) -> str | None:
    """Looks at the `lookback` windows immediately preceding `window_end`
    (300s apart each) and returns 'Up' or 'Down' by majority, or None if
    not enough resolved history is available."""
    prior_ends = [window_end - 300 * i for i in range(1, lookback + 1)]
    known = [res_map[w] for w in prior_ends if w in res_map]
    if len(known) < lookback * 0.7:  # require most of the lookback to actually be present
        return None
    up_count = sum(known)
    down_count = len(known) - up_count
    if up_count == down_count:
        return None
    return "Up" if up_count > down_count else "Down"


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
            results.append({"sample": label, "window_end": p["end_s"], "side": r["side"],
                             "price": r["price"], "won": r["won"]})
    print(f"{label}: {len(results)} confirmed crossings")
    return results


def main():
    print("Building resolution maps (free -- reuses cached event outcomes) ...")
    res_map_is = build_resolution_map("btc_5m_events.json")
    res_map_oos = build_resolution_map("btc_5m_events_oos.json")
    print(f"in_sample: {len(res_map_is)} resolved windows, oos: {len(res_map_oos)} resolved windows\n")

    all_results = run_sample("btc_5m_events.json", "in_sample") + \
                  run_sample("btc_5m_events_oos.json", "oos")

    for r in all_results:
        res_map = res_map_is if r["sample"] == "in_sample" else res_map_oos
        for lb in LOOKBACK_OPTIONS:
            r[f"trend_{lb}"] = trend_at(res_map, r["window_end"], lb)

    df = pmf.pd.DataFrame(all_results)
    out = Path(__file__).resolve().parents[1] / "results" / "btc_5m_momentum" / "trend_alignment_observations.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} observations to {out}\n")

    for lb in LOOKBACK_OPTIONS:
        col = f"trend_{lb}"
        known = df[df[col].notna()].copy()
        known["with_trend"] = known["side"] == known[col]
        with_trend = known[known["with_trend"]]
        counter_trend = known[~known["with_trend"]]
        print(f"=== lookback={lb} windows ({lb*5}min) === ({len(known)}/{len(df)} had enough trend history)")
        if len(with_trend):
            print(f"  with-trend:    n={len(with_trend):>4}  win_rate={with_trend['won'].mean():.4f}")
        if len(counter_trend):
            print(f"  counter-trend: n={len(counter_trend):>4}  win_rate={counter_trend['won'].mean():.4f}")
        print()


if __name__ == "__main__":
    main()
