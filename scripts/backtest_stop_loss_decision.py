"""Decision analysis for the stop-loss rule, combining the two halves measured by
backtest_stop_loss.py:

  cost/trade   -- from 575 historical pre-close entries (well powered)
  saving/loss  -- from the 11 real live losses (the only sample containing the event)

The rule is worth running iff  true_loss_rate > cost_per_trade / saving_per_loss.
That break-even is the number that matters, because the true loss rate is the one
parameter we genuinely do not know: the historical backtest says 0.34% (2/584),
live says 8.5% (11/129), and those cannot both describe the same process.

Also prints the per-loss realised exit for the chosen level, to confirm the fast
collapses are not being handed unrealistically good fills by the simulation.
"""
import csv
import sys
from pathlib import Path

from scipy import stats as sps

REPO = Path(r"D:\Finance\polymarket")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from backtest_stop_loss import simulate, load_live_loss, STAKE_USD

# cost per trade, averaged over in-sample and OOS pre-close buckets
# (from backtest_stop_loss_regime.py)
COST_PER_TRADE = {
    0.85: (0.1085 + 0.0754) / 2,
    0.80: (0.1086 + 0.0576) / 2,
    0.70: (0.0931 + 0.0428) / 2,
    0.60: (0.0864 + 0.0283) / 2,
    0.50: (0.0705 + 0.0229) / 2,
    0.40: (0.0607 + 0.0239) / 2,
}
# average realised loss with stop, from backtest_stop_loss.py live-loss replay
AVG_LOSS_WITH_STOP = {0.85: 2.06, 0.80: 2.12, 0.70: 2.21, 0.60: 2.85, 0.50: 3.24, 0.40: 3.46}
AVG_LOSS_NO_STOP = 5.02

LIVE_ALL = (11, 129)
LIVE_POSTFIX = (3, 49)
HISTORICAL = (2, 584)


def ci(k, n):
    lo, hi = sps.beta.ppf(0.025, k, n - k + 1), sps.beta.ppf(0.975, k + 1, n - k)
    return (0.0 if k == 0 else lo), hi


def main():
    print("=== BREAK-EVEN LOSS RATE BY EXIT LEVEL ===")
    print(f"{'exit_at':>8} {'cost/trade':>11} {'saved/loss':>11} {'breakeven':>11}")
    best = None
    for lvl in sorted(COST_PER_TRADE, reverse=True):
        cost = COST_PER_TRADE[lvl]
        saved = AVG_LOSS_NO_STOP - AVG_LOSS_WITH_STOP[lvl]
        be = cost / saved
        if best is None or be < best[1]:
            best = (lvl, be)
        print(f"{lvl:>8.2f} {cost:>11.4f} {saved:>11.2f} {be:>10.2%}")
    lvl, be = best
    print(f"\nlowest break-even: exit at {lvl:.2f}, needs true loss rate > {be:.2%}")

    print("\n=== IS OUR LOSS RATE ABOVE THAT? (95% Clopper-Pearson) ===")
    for name, (k, n) in [("live all", LIVE_ALL), ("live post-fix", LIVE_POSTFIX),
                         ("historical backtest", HISTORICAL)]:
        lo, hi = ci(k, n)
        verdict = "ABOVE break-even" if lo > be else ("below" if hi < be else "straddles break-even")
        print(f"  {name:22s} {k:3d}/{n:4d} = {k/n:6.2%}   95% CI [{lo:.2%}, {hi:.2%}]   {verdict}")

    k, n = LIVE_ALL
    p_hist = HISTORICAL[0] / HISTORICAL[1]
    p_val = sps.binomtest(k, n, p_hist, alternative="greater").pvalue
    print(f"\n  P(>={k} losses in {n} trades | historical rate {p_hist:.3%}) = {p_val:.2e}")
    print("  -> the historical loss rate cannot describe the live process.")

    print(f"\n=== EXPECTED NET EFFECT at exit {lvl:.2f} ===")
    saved = AVG_LOSS_NO_STOP - AVG_LOSS_WITH_STOP[lvl]
    cost = COST_PER_TRADE[lvl]
    for name, (k2, n2) in [("live all", LIVE_ALL), ("live post-fix", LIVE_POSTFIX),
                           ("historical", HISTORICAL)]:
        rate = k2 / n2
        net = rate * saved - cost
        print(f"  if true rate = {rate:6.2%} ({name:14s}): "
              f"{net:+.4f}/trade  = {net*10:+.2f}/hr at 10 trades/hr")

    print(f"\n=== PER-LOSS REALISED EXIT at {lvl:.2f} (sanity: no free lunch on fast collapses) ===")
    rows = list(csv.DictReader(open(REPO / "results" / "btc_5m_live" / "trade_log.csv")))
    losses = [r for r in rows if r.get("resolved_won") == "False"]
    print(f"{'window':>12} {'entry':>7} {'pnl_no_stop':>12} {'pnl_stop':>10} {'saved':>8}")
    tot_saved = 0.0
    for r in losses:
        L = load_live_loss(r)
        if not L:
            continue
        base, _ = simulate(L["series"], L["entry_ts"], L["entry_price"], False, None)
        stop, was = simulate(L["series"], L["entry_ts"], L["entry_price"], False, lvl)
        tot_saved += stop - base
        print(f"{L['window']:>12} {L['entry_price']:>7.2f} {base:>+12.2f} {stop:>+10.2f} "
              f"{stop-base:>+8.2f}{'' if was else '  (NOT stopped)'}")
    print(f"{'TOTAL':>12} {'':>7} {'':>12} {'':>10} {tot_saved:>+8.2f}")


if __name__ == "__main__":
    main()
