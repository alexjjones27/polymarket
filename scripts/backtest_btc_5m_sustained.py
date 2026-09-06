"""Refined version of backtest_btc_5m_continuous.py: instead of triggering
on the very FIRST trade that touches a threshold (which backtest showed
underperforms -- it catches "head-fake" spikes that partially reverse
before the window settles), this requires the price to STAY at or above
threshold for SUSTAIN_S consecutive seconds before confirming an entry,
rejecting any candidate that dips back below threshold during that
confirmation window. Entry price is whatever's prevailing AT the
confirmation moment (SUSTAIN_S seconds after the initial touch), not the
original touch price -- a real system enforcing "sustained" necessarily
executes after that confirmation delay, at the then-current price.

Same underlying trade-level data and time range as backtest_btc_5m_continuous.py.
"""
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import polymarket_final_pct as pmf
import backtest_btc_5m_continuous as cont

THRESHOLDS = [0.85, 0.88, 0.90, 0.92, 0.95]
SUSTAIN_OPTIONS_S = [3, 5, 10]


def prepare_event(event: dict) -> dict | None:
    """Fetches trades ONCE per event; reused across every threshold/sustain
    combination to avoid redundant API calls."""
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
    trades = cont.fetch_all_trades(cond_id)
    if not trades:
        return None
    relevant = [t for t in trades
                if cont.START_MONITORING_S <= (t["timestamp"] - end_s) <= cont.END_MONITORING_S]
    relevant.sort(key=lambda t: t["timestamp"])
    by_side = {"Up": [], "Down": []}
    for t in relevant:
        side = t.get("outcome")
        if side not in by_side:
            continue
        try:
            price = float(t["price"])
        except (ValueError, TypeError):
            continue
        by_side[side].append((t["timestamp"], price, t.get("size")))
    return {"by_side": by_side, "up_resolved": up_resolved, "end_s": end_s}


def find_sustained_crossing(prepared: dict, threshold: float, sustain_s: int) -> dict | None:
    up_resolved = prepared["up_resolved"]
    by_side = prepared["by_side"]
    end_s = prepared["end_s"]
    for side, series in by_side.items():
        i = 0
        while i < len(series):
            ts, price, size = series[i]
            if price >= threshold:
                # candidate crossing at (ts, price) -- check it holds for sustain_s
                confirm_deadline = ts + sustain_s
                reverted = False
                confirm_price, confirm_ts = price, ts
                j = i + 1
                while j < len(series) and series[j][0] <= confirm_deadline:
                    if series[j][1] < threshold:
                        reverted = True
                        break
                    confirm_price, confirm_ts = series[j][1], series[j][0]
                    j += 1
                if reverted:
                    i = j + 1
                    continue
                won = up_resolved if side == "Up" else (not up_resolved)
                return {"secs_after_close": confirm_ts - end_s, "side": side,
                        "price": confirm_price, "won": won}
            i += 1
    return None


def analyze_sample(prepared_events, threshold, sustain_s):
    results = []
    for p in prepared_events:
        r = find_sustained_crossing(p, threshold, sustain_s)
        if r:
            results.append(r)
    return results


def main():
    cache_name = sys.argv[1] if len(sys.argv) > 1 else "btc_5m_events.json"
    is_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "polymarket" / cache_name
    is_events = json.loads(is_path.read_text())["events"]
    random.seed(42)
    sample = random.sample(is_events, min(300, len(is_events)))
    print(f"Pilot ({cache_name}): {len(sample)} events, thresholds {THRESHOLDS}, sustain options {SUSTAIN_OPTIONS_S}")
    print("Fetching trades once per event (reused across all threshold/sustain combos) ...")

    prepared_events = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(prepare_event, e): e for e in sample}
        for i, fut in enumerate(as_completed(futures)):
            r = fut.result()
            if r:
                prepared_events.append(r)
            if (i + 1) % 50 == 0:
                print(f"  prepared {i+1}/{len(sample)} ...", flush=True)
    print(f"{len(prepared_events)}/{len(sample)} events had usable trade data\n")

    for sustain_s in SUSTAIN_OPTIONS_S:
        print(f"=== sustain={sustain_s}s ===")
        for thr in THRESHOLDS:
            results = analyze_sample(prepared_events, thr, sustain_s)
            n = len(results)
            if n == 0:
                print(f"  threshold={thr:.2f}: no confirmed crossings")
                continue
            wins = sum(1 for r in results if r["won"])
            wr = wins / n
            avg_price = sum(r["price"] for r in results) / n
            fee = 0.07 * (1 - avg_price)
            edge = wr - avg_price
            print(f"  threshold={thr:.2f}: n={n}/{len(prepared_events)}, win_rate={wr:.3f}, "
                  f"avg_price={avg_price:.3f}, edge={edge:+.4f}, net={edge-fee:+.4f}")
        print()


if __name__ == "__main__":
    main()
