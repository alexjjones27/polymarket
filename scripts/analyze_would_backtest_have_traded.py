"""Second half of the ask-vs-print diagnosis.

analyze_ask_vs_print.py established that we pay ~2.46pp above the prevailing print.
That is an execution cost. But if an inflated ask ALSO triggers entries the
print-based backtest would never have taken, the damage is worse than overpaying:
it is a selection effect that steers us into precisely the windows where the ask
is most detached from consensus.

This applies the backtest's own rule to our live entry moments: were the trade
PRINTS actually holding at/above PRICE_THRESHOLD for SUSTAIN_S before we bought?
If a large share of our trades -- and especially of our losses -- fail that test,
then the live bot is trading a different and worse population of windows than the
one the +2.72% OOS edge was measured on, and the backtest never validated what we
are actually doing.
"""
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path

REPO = Path(r"D:\Finance\polymarket")
sys.path.insert(0, str(REPO / "src"))
import polymarket_final_pct as pmf

THRESHOLD = 0.94
SUSTAIN_S = 5


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
    if not series:
        return None

    # the backtest's condition, evaluated at our entry moment: every print in the
    # SUSTAIN_S seconds before entry at/above threshold (and at least one print)
    win = [p for ts, p in series if 0 <= entry_ts - ts <= SUSTAIN_S]
    if not win:
        qualifies = False
        reason = "no prints in sustain window"
    elif min(win) < THRESHOLD:
        qualifies = False
        reason = f"print dipped to {min(win):.3f}"
    else:
        qualifies = True
        reason = ""

    return {"window": window_end, "won": row["resolved_won"] == "True",
            "fill": float(row["ask_price"]), "qualifies": qualifies,
            "reason": reason, "n_prints": len(win),
            "min_print": min(win) if win else None}


def main():
    rows = list(csv.DictReader(open(REPO / "results" / "btc_5m_live" / "trade_log.csv")))
    settled = [r for r in rows if r.get("resolved_won") in ("True", "False")]
    print(f"checking {len(settled)} live trades against the backtest's print rule ...",
          flush=True)

    out = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        futs = {pool.submit(analyse, r): r for r in settled}
        for fut in as_completed(futs):
            r = fut.result()
            if r:
                out.append(r)

    q = [r for r in out if r["qualifies"]]
    nq = [r for r in out if not r["qualifies"]]
    print(f"\n=== WOULD THE BACKTEST HAVE TAKEN OUR TRADES? (n={len(out)}) ===")
    print(f"  prints also sustained >= {THRESHOLD}:  {len(q):3d} = {len(q)/len(out):.1%}")
    print(f"  prints did NOT qualify:          {len(nq):3d} = {len(nq)/len(out):.1%}")
    print("  -> the second group is entries the ask created and the backtest never saw")

    print(f"\n=== WIN RATE OF EACH GROUP ===")
    for label, grp in [("backtest WOULD have traded", q), ("ask-only entries", nq)]:
        if not grp:
            continue
        w = sum(1 for r in grp if r["won"])
        print(f"  {label:28s} n={len(grp):3d}  wins={w:3d}  win_rate={w/len(grp):7.2%}")

    print(f"\n=== WHERE DID THE 11 LOSSES COME FROM? ===")
    losses = [r for r in out if not r["won"]]
    lq = sum(1 for r in losses if r["qualifies"])
    print(f"  losses the backtest would also have taken: {lq}/{len(losses)}")
    print(f"  losses that were ask-only entries:         {len(losses)-lq}/{len(losses)}")
    for r in sorted(losses, key=lambda x: x["window"]):
        tag = "backtest too" if r["qualifies"] else f"ASK-ONLY ({r['reason']})"
        print(f"    {r['window']} fill={r['fill']:.3f}  {tag}")


if __name__ == "__main__":
    main()
