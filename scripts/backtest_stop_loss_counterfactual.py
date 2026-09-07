"""What would our actual live P&L have been if the stop-loss had existed from day one?

backtest_stop_loss.py replayed only the LOSSES, which measures the benefit but not
the cost. This applies the rule to EVERY settled live trade, so winners that dipped
below the stop level and recovered are charged as false stops -- the honest version.

Rule replicated as configured live: exit once price holds below STOP_LOSS_LEVEL for
STOP_SUSTAIN_S, with LATENCY_S before the fill and a slippage haircut.

Caveat worth stating: the live rule reads the order book's BID, but the only history
available for past trades is trade PRINTS, which sit above the bid. A print-based
simulation therefore triggers slightly LATER than the real rule would, so these
numbers are approximate -- most likely conservative on both the saves and the false
stops. Book snapshots are being collected now so future analysis will not need this
approximation.
"""
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path

REPO = Path(r"D:\Finance\polymarket")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
import polymarket_final_pct as pmf

STOP_LEVEL = 0.70
STOP_SUSTAIN_S = 2.0
LATENCY_S = 2.0
SLIPPAGE = 0.03
FEE_RATE = 0.07
MID_FIX = datetime(2026, 9, 7, 9, 55, 55, tzinfo=timezone(timedelta(hours=1)))


def fee(shares, price):
    return shares * FEE_RATE * price * (1.0 - price)


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


def analyse(row):
    window_end = int(row["window_end"])
    side = row["side"]
    won = row["resolved_won"] == "True"
    shares = float(row["size"])
    cost = float(row["cost_usd"])
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
        if t.get("outcome") == side:
            series.append((t["timestamp"], p))
        elif t.get("outcome") in ("Up", "Down"):
            series.append((t["timestamp"], 1.0 - p))
    after = [(ts, p) for ts, p in series if ts > entry_ts]

    actual = (shares - cost) if won else -cost

    # simulate the stop
    breach_start = None
    confirmed = None
    for ts, p in after:
        if p < STOP_LEVEL:
            if breach_start is None:
                breach_start = ts
            if ts - breach_start >= STOP_SUSTAIN_S:
                confirmed = ts
                break
        else:
            breach_start = None
    stopped = False
    counter = actual
    if confirmed is not None:
        fill = next(((ts, p) for ts, p in after if ts >= confirmed + LATENCY_S), None)
        if fill is None:
            fill = next(((ts, p) for ts, p in reversed(after) if ts >= confirmed), None)
        if fill is not None:
            exit_px = max(0.001, fill[1] * (1.0 - SLIPPAGE))
            proceeds = shares * exit_px - fee(shares, exit_px)
            counter = proceeds - cost
            stopped = True

    return {"window": window_end, "won": won, "actual": actual, "counter": counter,
            "stopped": stopped, "entry": float(row["ask_price"]),
            "post_fix": entry_dt >= MID_FIX}


def main():
    rows = [r for r in csv.DictReader(open(REPO / "results" / "btc_5m_live" / "trade_log.csv"))
            if r.get("resolved_won") in ("True", "False")]
    print(f"replaying stop-loss over {len(rows)} settled live trades ...", flush=True)

    out = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        futs = {pool.submit(analyse, r): r for r in rows}
        for fut in as_completed(futs):
            r = fut.result()
            if r:
                out.append(r)

    def report(label, grp):
        if not grp:
            return
        a = sum(r["actual"] for r in grp)
        c = sum(r["counter"] for r in grp)
        st = [r for r in grp if r["stopped"]]
        false_stops = [r for r in st if r["won"]]
        true_stops = [r for r in st if not r["won"]]
        saved = sum(r["counter"] - r["actual"] for r in true_stops)
        lost = sum(r["counter"] - r["actual"] for r in false_stops)
        print(f"\n=== {label} (n={len(grp)}) ===")
        print(f"  actual P&L                 ${a:+.2f}")
        print(f"  with stop-loss             ${c:+.2f}")
        print(f"  difference                 ${c-a:+.2f}")
        print(f"  stop-outs                  {len(st)} of {len(grp)} = {len(st)/len(grp):.1%}")
        print(f"    true stops (real losses) {len(true_stops)}  saved ${saved:+.2f}")
        print(f"    false stops (would win)  {len(false_stops)}  cost  ${lost:+.2f}")

    report("ALL LIVE TRADES", out)
    report("PRE-FIX ONLY (ask-triggered era)", [r for r in out if not r["post_fix"]])
    report("POST-FIX ONLY (mid-triggered era)", [r for r in out if r["post_fix"]])


if __name__ == "__main__":
    main()
