"""Corrected re-analysis: the original time sweep (backtest_btc_5m_sweep.py)
used a bucket-midpoint approximation for implied probability, which
overstated edges ~4-5x (discovered when properly validating the T-30s
finding). This redoes the full (check-time x threshold) grid with the
CORRECTED edge (actual mean price in each slice, not a bucket midpoint)
and the fee-adjusted net edge, on BOTH the in-sample and out-of-sample BTC
windows, so any real combination has to show up on both samples to count.
Reuses cached price series entirely -- zero new network calls.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import polymarket_final_pct as pmf

CHECK_TIMES_S = [30, 60, 80, 90, 110, 120, 150, 180, 210, 240, 270]
THRESHOLDS = [0.85, 0.88, 0.90, 0.92, 0.95]
MIN_N = 100


def load_events(cache_name: str) -> list[dict]:
    path = Path(__file__).resolve().parents[1] / "data" / "raw" / "polymarket" / cache_name
    return json.loads(path.read_text())["events"]


def analyze_one(event: dict, secs_before: int) -> dict | None:
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


def fee_pct(price: float) -> float:
    return 0.07 * (1 - price)


def build_df(events: list[dict]) -> "pmf.pd.DataFrame":
    rows = []
    with ThreadPoolExecutor(max_workers=24) as pool:
        futures = {}
        for e in events:
            for secs in CHECK_TIMES_S:
                futures[pool.submit(analyze_one, e, secs)] = None
        for fut in as_completed(futures):
            r = fut.result()
            if r:
                rows.append(r)
    return pmf.pd.DataFrame(rows)


def main():
    print("Loading cached BTC event lists ...")
    is_events = load_events("btc_5m_events.json")
    oos_events = load_events("btc_5m_events_oos.json")

    print("Building in-sample grid (reusing cached price series) ...")
    is_df = build_df(is_events)
    print("Building out-of-sample grid ...")
    oos_df = build_df(oos_events)
    print(f"in-sample rows={len(is_df)}, OOS rows={len(oos_df)}\n")

    print(f"{'secs_before':>11} {'threshold':>10} | {'IS n':>6} {'IS net':>8} | {'OOS n':>6} {'OOS net':>8}")
    results = []
    for secs in CHECK_TIMES_S:
        for thr in THRESHOLDS:
            row = []
            for df in [is_df, oos_df]:
                g = df[(df["secs_before"] == secs) & (df["leading_price"] >= thr)]
                n = len(g)
                if n < MIN_N:
                    row.append((n, float("nan")))
                    continue
                wr = g["won"].mean()
                implied = g["leading_price"].mean()
                fee = g["leading_price"].apply(fee_pct).mean()
                net = (wr - implied) - fee
                row.append((n, net))
            (n1, net1), (n2, net2) = row
            results.append({"secs_before": secs, "threshold": thr, "is_n": n1, "is_net": net1,
                             "oos_n": n2, "oos_net": net2})
            n1s = f"{n1:>6}" if n1 else f"{n1:>6}"
            net1s = f"{net1:>+8.4f}" if n1 >= MIN_N else f"{'--':>8}"
            net2s = f"{net2:>+8.4f}" if n2 >= MIN_N else f"{'--':>8}"
            print(f"{secs:>11} {thr:>10.2f} | {n1s} {net1s} | {n2:>6} {net2s}")

    out = Path(__file__).resolve().parents[1] / "results" / "btc_5m_momentum" / "time_threshold_grid.csv"
    pmf.pd.DataFrame(results).to_csv(out, index=False)
    print(f"\nSaved {out}")

    # Highlight: both IS and OOS net edge positive AND OOS n>=MIN_N
    print("\n=== Combinations positive on BOTH samples ===")
    for r in results:
        if r["is_n"] >= MIN_N and r["oos_n"] >= MIN_N and r["is_net"] > 0 and r["oos_net"] > 0:
            print(f"secs_before={r['secs_before']:>3}  threshold={r['threshold']:.2f}  "
                  f"IS net={r['is_net']:+.4f}  OOS net={r['oos_net']:+.4f}")


if __name__ == "__main__":
    main()
