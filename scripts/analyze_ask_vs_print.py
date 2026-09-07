"""Why does the backtest show an edge the live system does not?

Leading hypothesis: the backtest and the live bot are not detecting the same
signal, because they read different prices.

  backtest -- find_sustained_crossing() scans TRADE PRINTS. It fires when actual
              executed trades hold at/above the threshold, and it records the
              print price as the entry price.
  live     -- poll_window() reads the BEST ASK in the order book. It fires when
              the ask holds at/above the threshold, and pays the ask.

Near close these books can be very asymmetric: bid 0.85 / ask 0.97 with prints
landing around 0.88-0.92. In that window the backtest sees 0.88-0.92, stays below
a 0.94 threshold and does NOT trade -- while the live bot sees 0.97, fires, and
buys at a price far above what anyone is actually paying.

If that is what is happening it is a selection effect, not a pricing nuisance: the
live bot would be systematically entering exactly the windows where the ask is most
inflated relative to consensus, i.e. the worst ones, which the backtest never
records at all. It predicts our fills sit ABOVE contemporaneous prints.

This measures that gap directly on every live trade we have.
"""
import csv
import statistics as st
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path

REPO = Path(r"D:\Finance\polymarket")
sys.path.insert(0, str(REPO / "src"))
import polymarket_final_pct as pmf

WINDOW_S = 10  # how far either side of entry to sample prints


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
    fill = float(row["ask_price"])
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

    before = [p for ts, p in series if 0 <= entry_ts - ts <= WINDOW_S]
    around = [p for ts, p in series if abs(ts - entry_ts) <= WINDOW_S]
    if not before or not around:
        return None

    return {
        "window": window_end, "won": row["resolved_won"] == "True",
        "fill": fill,
        "print_before": st.median(before),
        "print_around": st.median(around),
        "gap_before": fill - st.median(before),
        "gap_around": fill - st.median(around),
        "n_before": len(before),
    }


def main():
    rows = list(csv.DictReader(open(REPO / "results" / "btc_5m_live" / "trade_log.csv")))
    settled = [r for r in rows if r.get("resolved_won") in ("True", "False")]
    print(f"analysing {len(settled)} live trades ...", flush=True)

    out = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        futs = {pool.submit(analyse, r): r for r in settled}
        for fut in as_completed(futs):
            r = fut.result()
            if r:
                out.append(r)

    gaps_b = [r["gap_before"] for r in out]
    gaps_a = [r["gap_around"] for r in out]
    print(f"\n=== DID WE PAY ABOVE THE MARKET? (n={len(out)}) ===")
    print(f"  mean fill price                    {st.mean([r['fill'] for r in out]):.4f}")
    print(f"  mean median-print in 10s BEFORE    {st.mean([r['print_before'] for r in out]):.4f}")
    print(f"  mean gap (fill - print_before)     {st.mean(gaps_b):+.4f}  "
          f"median {st.median(gaps_b):+.4f}")
    print(f"  mean gap (fill - print_around)     {st.mean(gaps_a):+.4f}  "
          f"median {st.median(gaps_a):+.4f}")
    paid_above = sum(1 for g in gaps_b if g > 0)
    print(f"  trades where we paid ABOVE prints  {paid_above}/{len(out)} = {paid_above/len(out):.1%}")

    print(f"\n=== GAP BY OUTCOME ===")
    for label, sel in [("WINS", True), ("LOSSES", False)]:
        grp = [r for r in out if r["won"] is sel]
        if not grp:
            continue
        g = [r["gap_before"] for r in grp]
        print(f"  {label:7s} n={len(grp):3d}  mean_fill={st.mean([r['fill'] for r in grp]):.4f}  "
              f"mean_print={st.mean([r['print_before'] for r in grp]):.4f}  "
              f"mean_gap={st.mean(g):+.4f}")

    print(f"\n=== IMPLIED vs REALISED (the edge itself) ===")
    n = len(out)
    wins = sum(1 for r in out if r["won"])
    mean_fill = st.mean([r["fill"] for r in out])
    mean_print = st.mean([r["print_before"] for r in out])
    print(f"  we paid a price implying   {mean_fill:.2%} win probability")
    print(f"  the market was printing    {mean_print:.2%}")
    print(f"  we actually won            {wins/n:.2%}  ({wins}/{n})")
    print(f"  edge vs our fill price     {wins/n - mean_fill:+.2%}")
    print(f"  edge vs the print price    {wins/n - mean_print:+.2%}")


if __name__ == "__main__":
    main()
