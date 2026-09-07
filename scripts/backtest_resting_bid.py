"""Tests the user-proposed idea: instead of taking the ask at ~0.95, rest a BUY
order below the market (e.g. 0.85) and let the wicks come to us.

The appeal is real. Filling at 0.85 instead of 0.95 changes the payoff from about
+5% to +17.6% per win, and a resting order is a MAKER order, so we would earn the
rebate instead of paying the ~0.07*p*(1-p) taker fee we currently pay on every trade.

The danger is equally real and is the classic market-making problem: a resting bid
only fills when someone is willing to sell into it, which is precisely when the
market is repricing against us. We already have a hint that this is what happens --
of our live trades, the 22 that filled BELOW their limit (i.e. swept a moving book)
won only 81.8% versus 92.7% for exact fills, and returned -7.76% versus -3.90%.

So the whole question is empirical: conditional on the price wicking down to LEVEL,
how often does it recover and settle at 1.00? Break-even is a win rate equal to
LEVEL itself (buy at 0.85, need >85% to profit), which is a high bar.

Method: identify the favoured side the same way the strategy does (first sustained
crossing of PRICE_THRESHOLD on prints), then assume a resting bid at LEVEL from that
moment on. It fills if any later print trades at or below LEVEL. Outcome is whether
that side ultimately won.
"""
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scipy import stats as sps

REPO = Path(r"D:\Finance\polymarket")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
import backtest_btc_5m_sustained as sust

THRESHOLD = 0.94
SUSTAIN_S = 5
SAMPLE_N = 400
STAKE_USD = 5.0
LEVELS = [0.90, 0.85, 0.80, 0.75, 0.70]


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


def run(prepared, label):
    print(f"\n=== {label} (n={len(prepared)} windows) ===")
    print(f"{'bid_at':>7} {'signals':>8} {'filled':>7} {'fill_rate':>10} {'wins':>6} "
          f"{'win_rate':>9} {'breakeven':>10} {'pnl':>9} {'per_fill':>9}")
    for lvl in LEVELS:
        signals = filled = wins = 0
        pnl = 0.0
        for p in prepared:
            r = sust.find_sustained_crossing(p, THRESHOLD, SUSTAIN_S)
            if not r:
                continue
            signals += 1
            entry_ts = p["end_s"] + r["secs_after_close"]
            series = p["by_side"][r["side"]]
            after = [(ts, pr) for ts, pr, _ in series if ts > entry_ts]
            # a resting bid at lvl fills once someone trades down to it
            if not any(pr <= lvl for _, pr in after):
                continue
            filled += 1
            shares = STAKE_USD / lvl
            if r["won"]:
                wins += 1
                pnl += shares - STAKE_USD
            else:
                pnl -= STAKE_USD
        if not filled:
            print(f"{lvl:>7.2f} {signals:>8} {filled:>7} {'--':>10}")
            continue
        wr = wins / filled
        print(f"{lvl:>7.2f} {signals:>8} {filled:>7} {filled/signals:>9.1%} {wins:>6} "
              f"{wr:>8.2%} {lvl:>9.2%} {pnl:>+9.2f} {pnl/filled:>+9.3f}")


def main():
    is_prep = prepare("btc_5m_events.json")
    oos_prep = prepare("btc_5m_events_oos.json")
    run(is_prep, "IN-SAMPLE")
    run(oos_prep, "OOS")

    print("\n=== interpretation ===")
    print("win_rate must exceed breakeven (= the bid level) for the idea to make money,")
    print("before even accounting for the maker rebate that would improve it slightly.")


if __name__ == "__main__":
    main()
