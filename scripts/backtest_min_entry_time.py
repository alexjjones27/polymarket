"""Tests the strongest live signal found so far: all 6 real losses occurred
at entries before T+180s (relative to nominal close); 31 trades entered at
T+180s or later had zero losses.

Mechanism: these windows keep trading for minutes after nominal close while
the settlement price is determined and disseminated. Early post-close there
is still genuine uncertainty; by T+180s+ the outcome is largely locked in.

This was NOT covered by earlier backtests -- find_sustained_crossing() takes
the FIRST qualifying crossing anywhere in [-90s, +300s], which historically
lands pre-close ~90% of the time. Here we impose a MINIMUM entry time: ignore
any crossing before min_entry_s, and take the first qualifying one at/after it.

Reports win rate AND net-of-fee edge, since later entries pay higher prices --
a lower loss rate is worthless if the edge shrinks proportionally.
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
MIN_ENTRY_OPTIONS = [None, 0, 60, 120, 180, 240]
STAKE_USD = 8.0


def find_crossing_after(prepared: dict, threshold: float, sustain_s: int,
                        min_entry_s) -> dict | None:
    """Same sustained-crossing logic as backtest_btc_5m_sustained, but only
    considers candidate crossings whose confirmation lands at/after
    min_entry_s (seconds relative to nominal close; None = no restriction)."""
    up_resolved = prepared["up_resolved"]
    end_s = prepared["end_s"]
    for side, series in prepared["by_side"].items():
        i = 0
        while i < len(series):
            ts, price, _size = series[i]
            if price >= threshold:
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
                rel = confirm_ts - end_s
                if min_entry_s is not None and rel < min_entry_s:
                    i += 1  # too early -- keep scanning for a later qualifying crossing
                    continue
                won = up_resolved if side == "Up" else (not up_resolved)
                return {"secs_after_close": rel, "side": side, "price": confirm_price, "won": won}
            i += 1
    return None


def prepare(cache_name: str, label: str) -> list[dict]:
    path = Path(__file__).resolve().parents[1] / "data" / "raw" / "polymarket" / cache_name
    events = json.loads(path.read_text())["events"]
    random.seed(42)
    sample = random.sample(events, min(300, len(events)))
    print(f"{label}: fetching trades for {len(sample)} events ...", flush=True)
    prepared = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(sust.prepare_event, e): e for e in sample}
        for fut in as_completed(futures):
            r = fut.result()
            if r:
                prepared.append(r)
    return prepared


def main():
    is_prep = prepare("btc_5m_events.json", "in_sample")
    oos_prep = prepare("btc_5m_events_oos.json", "oos")

    print(f"\n{'min_entry':>10} {'sample':>10} {'n':>5} {'win_rate':>9} {'avg_price':>10} "
          f"{'net_edge':>9} {'$/hr':>8}")
    for min_entry in MIN_ENTRY_OPTIONS:
        for label, prep in [("in_sample", is_prep), ("oos", oos_prep)]:
            results = [r for r in (find_crossing_after(p, THRESHOLD, SUSTAIN_S, min_entry)
                                    for p in prep) if r]
            n = len(results)
            if n == 0:
                continue
            wins = sum(1 for r in results if r["won"])
            wr = wins / n
            avg_price = sum(r["price"] for r in results) / n
            avg_fee = sum(0.07 * (1 - r["price"]) for r in results) / n
            net = (wr - avg_price) - avg_fee
            trades_hr = (n / 300) * 12
            usd_hr = trades_hr * STAKE_USD * net
            tag = "none" if min_entry is None else f"T+{min_entry}s"
            print(f"{tag:>10} {label:>10} {n:>5} {wr:>9.4f} {avg_price:>10.4f} "
                  f"{net:>+9.4f} {usd_hr:>+8.2f}")


if __name__ == "__main__":
    main()
