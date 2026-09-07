"""Open-ended search for exploitable edge, plus the requested entry-threshold sweep.

Three analyses, deliberately not restricted to the strategy we already run:

1. CALIBRATION CURVE -- the most informative single test. At fixed times before
   close, sample BOTH sides of every window and bucket by price. Exactly one side
   of each window wins, so this covers the whole 0-1 range with no selection at
   all. If the market is well calibrated everywhere, no entry rule at any price
   can profit, and everything else here is doomed. If some band is systematically
   mispriced, that band IS the edge, whichever direction it points.

2. THRESHOLD SWEEP -- the sustained-crossing rule at thresholds from 0.55 to 0.97,
   answering directly what a 0.70 entry would have returned. Lower thresholds fire
   earlier and more often, and pay far more per win, but their break-even win rate
   is also far lower AND they pay much higher fees.

3. LONGSHOT TEST -- the mirror of what we do: buy the CHEAP side. Favourite-longshot
   bias would make these systematically overpriced; if instead they are underpriced
   in these fast markets, that is an edge pointing the opposite way to our strategy.

Fee model is the real one, fee_per_share = 0.07 * price * (1-price), not the
linearised version used in some earlier scripts. This matters enormously here: fees
peak at price 0.50 (0.0175/share) and are ~4x higher at 0.70 than at 0.94, which is
exactly the region a lower threshold would push us into.

Net edge per share = win_rate - price - fee. Break-even win rate = price + fee.
"""
import json
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scipy import stats as sps

REPO = Path(r"D:\Finance\polymarket")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
import backtest_btc_5m_sustained as sust

SAMPLE_N = 600
SUSTAIN_S = 5
STAKE_USD = 5.0
THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.94, 0.97]
SNAPSHOT_OFFSETS = [-60, -30]        # seconds relative to true close
CAL_BUCKETS = [(lo / 20, (lo + 1) / 20) for lo in range(20)]   # 0.00-0.05 ... 0.95-1.00


def fee_per_share(price):
    return 0.07 * price * (1.0 - price)


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


def price_at(series, end_s, offset):
    """Last traded price at or before end_s+offset."""
    target = end_s + offset
    last = None
    for ts, p, _ in series:
        if ts <= target:
            last = p
        else:
            break
    return last


def calibration(prepared, label):
    print(f"\n=== 1. CALIBRATION CURVE -- {label} ===")
    print("   every window contributes both sides; exactly one wins. no selection.")
    for offset in SNAPSHOT_OFFSETS:
        buckets = defaultdict(lambda: [0, 0])  # bucket -> [n, wins]
        for p in prepared:
            for side, series in p["by_side"].items():
                px = price_at(series, p["end_s"], offset)
                if px is None:
                    continue
                won = p["up_resolved"] if side == "Up" else (not p["up_resolved"])
                for lo, hi in CAL_BUCKETS:
                    if lo <= px < hi:
                        buckets[(lo, hi)][0] += 1
                        buckets[(lo, hi)][1] += 1 if won else 0
                        break
        print(f"\n  at close{offset:+d}s:")
        print(f"  {'price band':>14} {'n':>5} {'implied':>8} {'actual':>8} {'diff':>8} "
              f"{'fee':>7} {'net_edge':>9} {'p':>7}")
        for (lo, hi), (n, w) in sorted(buckets.items()):
            if n < 25:
                continue
            mid = (lo + hi) / 2
            actual = w / n
            f = fee_per_share(mid)
            net = actual - mid - f
            pv = sps.binomtest(w, n, min(0.999, mid + f), alternative="two-sided").pvalue
            flag = ""
            if pv < 0.01 and net > 0:
                flag = "  <-- BUY EDGE"
            elif pv < 0.01 and net < 0:
                flag = "  <-- overpriced"
            print(f"  [{lo:.2f},{hi:.2f}) {n:>5} {mid:>8.3f} {actual:>8.3f} "
                  f"{actual-mid:>+8.3f} {f:>7.4f} {net:>+9.4f} {pv:>7.4f}{flag}")


def threshold_sweep(prepared, label):
    print(f"\n=== 2. ENTRY THRESHOLD SWEEP -- {label} ===")
    print(f"  {'thresh':>7} {'n':>5} {'rate':>6} {'win_rate':>9} {'avg_px':>7} {'breakeven':>10} "
          f"{'fee':>7} {'edge/sh':>8} {'ret_on_stake':>13} {'total_pnl':>10} {'$/hr':>7}")
    for th in THRESHOLDS:
        res = []
        for p in prepared:
            r = sust.find_sustained_crossing(p, th, SUSTAIN_S)
            if r:
                res.append(r)
        n = len(res)
        if n == 0:
            continue
        wins = sum(1 for r in res if r["won"])
        wr = wins / n
        avg_px = sum(r["price"] for r in res) / n
        avg_fee = sum(fee_per_share(r["price"]) for r in res) / n
        # exact per-trade P&L at a fixed dollar stake
        pnl = 0.0
        for r in res:
            px = r["price"]
            shares = STAKE_USD / px
            pnl += (shares - STAKE_USD - shares * fee_per_share(px)) if r["won"] \
                else (-STAKE_USD - shares * fee_per_share(px))
        edge_sh = wr - avg_px - avg_fee
        ret = pnl / (n * STAKE_USD)
        trades_hr = (n / len(prepared)) * 12
        print(f"  {th:>7.2f} {n:>5} {n/len(prepared):>5.0%} {wr:>8.2%} {avg_px:>7.3f} "
              f"{avg_px+avg_fee:>10.3f} {avg_fee:>7.4f} {edge_sh:>+8.4f} {ret:>+12.2%} "
              f"{pnl:>+10.2f} {trades_hr*STAKE_USD*edge_sh/avg_px:>+7.2f}")


def longshot(prepared, label):
    print(f"\n=== 3. LONGSHOT TEST (buy the CHEAP side) -- {label} ===")
    print(f"  {'buy below':>10} {'n':>5} {'win_rate':>9} {'avg_px':>7} {'breakeven':>10} "
          f"{'edge/sh':>8} {'total_pnl':>10}")
    for cap in [0.05, 0.10, 0.15, 0.20, 0.30]:
        res = []
        for p in prepared:
            for side, series in p["by_side"].items():
                px = price_at(series, p["end_s"], -60)
                if px is None or px > cap or px <= 0.005:
                    continue
                won = p["up_resolved"] if side == "Up" else (not p["up_resolved"])
                res.append((px, won))
        n = len(res)
        if n < 10:
            continue
        wins = sum(1 for _, w in res if w)
        wr = wins / n
        avg_px = sum(px for px, _ in res) / n
        avg_fee = sum(fee_per_share(px) for px, _ in res) / n
        pnl = 0.0
        for px, w in res:
            sh = STAKE_USD / px
            pnl += (sh - STAKE_USD - sh * fee_per_share(px)) if w else (-STAKE_USD - sh * fee_per_share(px))
        print(f"  {cap:>10.2f} {n:>5} {wr:>8.2%} {avg_px:>7.3f} {avg_px+avg_fee:>10.3f} "
              f"{wr-avg_px-avg_fee:>+8.4f} {pnl:>+10.2f}")


def main():
    for cache, label in [("btc_5m_events.json", "IN-SAMPLE"),
                         ("btc_5m_events_oos.json", "OOS")]:
        prep = prepare(cache)
        print(f"\n{'#'*70}\n# {label}: {len(prep)} windows\n{'#'*70}")
        calibration(prep, label)
        threshold_sweep(prep, label)
        longshot(prep, label)


if __name__ == "__main__":
    main()
