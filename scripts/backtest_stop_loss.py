"""Tests an emergency-exit (stop-loss) rule for the BTC 5-minute sustained-crossing
strategy, which currently buys once and holds to settlement with no position
management.

Motivation: forensics on all 11 live losses showed the collapses took 10-43s to
play out after entry, with trade prints continuing throughout (not a liquidity
freeze). So there may be time to sell back out before settlement, converting a
total loss into a partial one.

Two-part method, because neither sample alone can answer the question:

  1. COST (false triggers) -- from the historical in-sample/OOS backtests. These
     contain 584 qualifying trades but only 2 losses, so they cannot estimate the
     benefit of a stop; what they CAN measure is how often a winning position dips
     below the exit threshold and recovers, which is what the rule costs us.

  2. BENEFIT (real saves) -- replayed against the 11 actual live losses, using
     their real post-entry price paths. This is the only sample that contains the
     event the rule exists to protect against.

Realism constraints modelled explicitly (all pessimistic, since a stop-loss
backtest is very easy to flatter):
  - LATENCY_S: the live poller runs at 1s; we exit only at the first print at
    least LATENCY_S after the threshold is breached, not at the breach itself.
  - SLIPPAGE: we sell into the bid, but trade prints are last-trade. In a
    collapsing book the bid is below the last print, so exit proceeds are haircut.
  - Exit pays the taker fee again: shares * 0.07 * price * (1-price).
"""
import csv
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path

REPO = Path(r"D:\Finance\polymarket")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
import polymarket_final_pct as pmf
import backtest_btc_5m_sustained as sust

THRESHOLD = 0.94
SUSTAIN_S = 5
SAMPLE_N = 300
STAKE_USD = 5.0
LATENCY_S = 2.0            # poll interval + order round trip
SLIPPAGE = 0.03            # sell into the bid, not the last print
EXIT_LEVELS = [None, 0.85, 0.80, 0.70, 0.60, 0.50, 0.40]


def taker_fee(shares: float, price: float) -> float:
    return shares * 0.07 * price * (1.0 - price)


def simulate(series, entry_ts, entry_price, won, exit_level):
    """Returns (pnl, stopped_out). series is [(ts, price, size)] for OUR side."""
    shares = STAKE_USD / entry_price
    cost = shares * entry_price + taker_fee(shares, entry_price)

    if exit_level is not None:
        after = [(ts, p) for ts, p, _ in series if ts > entry_ts]
        breach_ts = next((ts for ts, p in after if p < exit_level), None)
        if breach_ts is not None:
            # exit at the first print at least LATENCY_S after the breach
            fill = next(((ts, p) for ts, p in after if ts >= breach_ts + LATENCY_S), None)
            if fill is None:
                fill = next(((ts, p) for ts, p in reversed(after) if ts >= breach_ts), None)
            if fill is not None:
                exit_price = max(0.001, fill[1] * (1.0 - SLIPPAGE))
                proceeds = shares * exit_price - taker_fee(shares, exit_price)
                return proceeds - cost, True

    payout = shares if won else 0.0
    return payout - cost, False


def prepare(cache_name):
    path = REPO / "data" / "raw" / "polymarket" / cache_name
    events = json.loads(path.read_text())["events"]
    random.seed(42)
    sample = random.sample(events, min(SAMPLE_N, len(events)))
    prepared = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futs = {pool.submit(sust.prepare_event, e): e for e in sample}
        for fut in as_completed(futs):
            r = fut.result()
            if r:
                prepared.append(r)
    return prepared


def run_historical(prepared, label):
    print(f"\n=== {label} (n={len(prepared)} windows) ===")
    print(f"{'exit_at':>8} {'trades':>7} {'stops':>6} {'false_stops':>12} {'true_stops':>11} "
          f"{'total_pnl':>10} {'vs_baseline':>12}")
    baseline = None
    for lvl in EXIT_LEVELS:
        total, stops, false_stops, true_stops, n = 0.0, 0, 0, 0, 0
        for p in prepared:
            r = sust.find_sustained_crossing(p, THRESHOLD, SUSTAIN_S)
            if not r:
                continue
            entry_ts = p["end_s"] + r["secs_after_close"]
            series = p["by_side"][r["side"]]
            pnl, stopped = simulate(series, entry_ts, r["price"], r["won"], lvl)
            total += pnl
            n += 1
            if stopped:
                stops += 1
                if r["won"]:
                    false_stops += 1
                else:
                    true_stops += 1
        if baseline is None:
            baseline = total
        tag = "none" if lvl is None else f"{lvl:.2f}"
        print(f"{tag:>8} {n:>7} {stops:>6} {false_stops:>12} {true_stops:>11} "
              f"{total:>+10.2f} {total - baseline:>+12.2f}")


def fetch_all_trades(cond_id):
    out, offset = [], 0
    while True:
        page = pmf._get(pmf.DATA_API_BASE, "/trades",
                        {"market": cond_id, "limit": 500, "offset": offset})
        if not page:
            break
        out.extend(page)
        if len(page) < 500:
            break
        offset += 500
        if offset > 2000:
            break
    return out


def load_live_loss(row):
    window_end = int(row["window_end"])
    side = row["side"]
    entry_dt = datetime.strptime(row["trade_time"], "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=timezone(timedelta(hours=1)))
    entry_ts = entry_dt.timestamp()
    res = pmf._get(pmf.GAMMA_BASE, "/events", {"slug": f"btc-updown-5m-{window_end}"})
    if not res:
        return None
    trades = fetch_all_trades(res[0]["markets"][0].get("conditionId"))
    series = []
    for t in sorted(trades, key=lambda x: x["timestamp"]):
        try:
            p = float(t["price"])
        except (ValueError, TypeError):
            continue
        sz = float(t.get("size", 0) or 0)
        if t.get("outcome") == side:
            series.append((t["timestamp"], p, sz))
        elif t.get("outcome") in ("Up", "Down"):
            series.append((t["timestamp"], 1.0 - p, sz))
    if not series:
        return None
    return {"window": window_end, "series": series, "entry_ts": entry_ts,
            "entry_price": float(row["ask_price"])}


def run_live_losses():
    rows = list(csv.DictReader(open(REPO / "results" / "btc_5m_live" / "trade_log.csv")))
    losses = [r for r in rows if r.get("resolved_won") == "False"]
    print(f"\n=== REAL LIVE LOSSES (n={len(losses)}) -- benefit side ===")
    loaded = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(load_live_loss, r): r for r in losses}
        for fut in as_completed(futs):
            r = fut.result()
            if r:
                loaded.append(r)
    print(f"{'exit_at':>8} {'saved':>6} {'total_pnl':>10} {'avg_loss':>10}  (baseline = hold to $0)")
    for lvl in EXIT_LEVELS:
        total, saved = 0.0, 0
        for L in loaded:
            pnl, stopped = simulate(L["series"], L["entry_ts"], L["entry_price"], False, lvl)
            total += pnl
            if stopped:
                saved += 1
        tag = "none" if lvl is None else f"{lvl:.2f}"
        print(f"{tag:>8} {saved:>6} {total:>+10.2f} {total / len(loaded):>+10.2f}")


def main():
    is_prep = prepare("btc_5m_events.json")
    oos_prep = prepare("btc_5m_events_oos.json")
    run_historical(is_prep, "IN-SAMPLE -- cost of false triggers")
    run_historical(oos_prep, "OOS -- cost of false triggers")
    run_live_losses()


if __name__ == "__main__":
    main()
