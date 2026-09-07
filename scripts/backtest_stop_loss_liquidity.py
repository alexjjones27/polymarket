"""The stop-loss simulation assumes we can sell at a print price minus slippage.
That is worthless if no counterparty exists: to exit we must hit a bid, and in a
collapsing one-sided book there may be none.

Trade prints are proof of a match, so this checks, for each real live loss, how
much size actually traded on our side in the seconds after the stop would have
fired -- i.e. whether a counterparty demonstrably existed at those prices, and
whether our ~5 shares would have been a small or large fraction of it.
"""
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(r"D:\Finance\polymarket")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from backtest_stop_loss import load_live_loss, STAKE_USD

EXIT_LEVEL = 0.70
LATENCY_S = 2.0
WINDOW_S = 15.0  # how long after the breach we look for exit liquidity


def main():
    rows = list(csv.DictReader(open(REPO / "results" / "btc_5m_live" / "trade_log.csv")))
    losses = [r for r in rows if r.get("resolved_won") == "False"]
    loaded = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(load_live_loss, r): r for r in losses}
        for fut in as_completed(futs):
            r = fut.result()
            if r:
                loaded.append(r)
    loaded.sort(key=lambda x: x["window"])

    print(f"exit level {EXIT_LEVEL}, latency {LATENCY_S}s, liquidity window {WINDOW_S}s")
    print(f"{'window':>12} {'our_shares':>11} {'prints':>7} {'shares_traded':>14} "
          f"{'our_%_of_vol':>13} {'px_range':>16}")
    for L in loaded:
        our_shares = STAKE_USD / L["entry_price"]
        after = [(ts, p, sz) for ts, p, sz in L["series"] if ts > L["entry_ts"]]
        breach = next((ts for ts, p, _ in after if p < EXIT_LEVEL), None)
        if breach is None:
            print(f"{L['window']:>12}  never breached")
            continue
        win = [(ts, p, sz) for ts, p, sz in after
               if breach + LATENCY_S <= ts <= breach + LATENCY_S + WINDOW_S]
        if not win:
            print(f"{L['window']:>12} {our_shares:>11.1f} {0:>7} {0.0:>14.1f} "
                  f"{'NO LIQUIDITY':>13}")
            continue
        vol = sum(sz for _, _, sz in win)
        pxs = [p for _, p, _ in win]
        pct = our_shares / vol * 100 if vol else float("inf")
        print(f"{L['window']:>12} {our_shares:>11.1f} {len(win):>7} {vol:>14.1f} "
              f"{pct:>12.1f}% {min(pxs):.3f}-{max(pxs):.3f}")


if __name__ == "__main__":
    main()
