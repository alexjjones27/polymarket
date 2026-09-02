"""Walk-forward backtest: buy YES on Polymarket politics markets, Kelly-sized
against an empirically-calibrated LOCAL probability estimate for the
market's own entry price (see politics_kelly_calibration.py -- specifically
local_calibrated_p_yes -- for why this, and not trusting the market's own
price as truth, is what generates the edge Kelly sizes against). Population
and entries from scripts/build_politics_kelly_population.py
(results/polymarket_final_pct/politics_kelly_entries.csv).

Mechanics mirror run_kelly_backtest.py's own event-loop simulation (entry/
resolve events over a shared bankroll, walk-forward belief updates, per-
trade/per-bucket/aggregate risk caps, the same FRACTIONS sweep and
START_BANKROLL) -- reused deliberately rather than reinvented.

An earlier version of this script calibrated against fixed 10%-wide price
buckets (one posterior mean per bucket). Discarded after inspecting the
first backtest run: comparing one bucket-average estimate against each
member's own EXACT entry price is a real discretization bug (see
local_calibrated_p_yes's docstring) that systematically selects the
worst-performing market within each bucket, not just adds noise -- a likely
cause of the wild, non-monotonic swings the bucket version showed across
the fraction sweep (adjacent Kelly fractions producing +9.7% and -45% CAGR
back to back). Price buckets are kept only for the whole-sample calibration
DIAGNOSTIC below (a coarse summary table is still a fine way to eyeball
calibration), never for sizing a trade.
"""
import csv
import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from politics_kelly_calibration import BUCKET_WIDTH, KERNEL_BANDWIDTH, kelly_fraction, local_calibrated_p_yes, price_bucket
import polymarket_final_pct as pmf

REPO = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO / "results" / "polymarket_final_pct"
ENTRIES_PATH = RESULTS_DIR / "politics_kelly_entries.csv"

START_BANKROLL = 10_000.0
FRACTIONS = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]
MAX_POS_PCT = 0.03       # no single trade risks more than 3% of current equity
AGG_CAP_PCT = 0.50       # no more than 50% of equity committed across all open positions at once
BUCKET_CAP_PCT = 0.25    # no more than 25% of equity committed to any one (still 10%-wide) price band at once -- a risk/diversification cap, unrelated to how calibration itself is computed
DEFAULT_PRIOR_STRENGTH = 20.0  # pseudo-observations of weight behind "assume this exact price is already correct"
PRIOR_STRENGTH_SWEEP = [5.0, 20.0, 50.0]
DEFAULT_BANDWIDTH = KERNEL_BANDWIDTH
BANDWIDTH_SWEEP = [0.02, 0.05, 0.10]
FEE_CATEGORY = "politics"

# A market's own first-trade notional is a real, measured ceiling on what
# could plausibly have been filled at that exact price -- median across
# this population is $9.20, 88% under $50 (checked directly against the
# cached trade tapes). Sizing purely off a growing bankroll (up to
# MAX_POS_PCT of it) would silently assume fills many multiples larger than
# the entire real market activity being modeled. 1.0x means "at most the
# one observed trade's own size" -- the most conservative, defensible
# reading, not "up to some assumed multiple of it."
DEFAULT_LIQUIDITY_CAP_MULTIPLE = 1.0
BANKROLL_SWEEP = [10_000.0, 1_000.0]


def parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_entries(path: Path = ENTRIES_PATH) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["entry_dt"] = parse_dt(r["entry_time"])
        r["resolve_dt"] = parse_dt(r["resolution_time"])
        r["yes_price"] = float(r["yes_price"])
        r["first_trade_notional"] = float(r["first_trade_notional"])
        r["resolved_yes"] = r["resolved_yes"] == "True"
        r["n_trades"] = int(r["n_trades"])
    return rows


def run_sim(entries: list[dict], fraction: float, fee_mode: str, prior_strength: float = DEFAULT_PRIOR_STRENGTH,
            bandwidth: float = DEFAULT_BANDWIDTH,
            max_pos_pct: float = MAX_POS_PCT, agg_cap_pct: float = AGG_CAP_PCT, bucket_cap_pct: float = BUCKET_CAP_PCT,
            liquidity_cap_multiple: float = DEFAULT_LIQUIDITY_CAP_MULTIPLE, start_bankroll: float = START_BANKROLL,
            track_trades: bool = False) -> dict:
    events = []
    for r in entries:
        resolve_key_t = max(r["resolve_dt"], r["entry_dt"])  # never resolve before entry (same defensive clamp as run_kelly_backtest.py)
        events.append((r["entry_dt"], 0, "entry", r))
        events.append((resolve_key_t, 1, "resolve", r))
    events.sort(key=lambda e: (e[0], e[1]))

    # Growing list of (entry_price, resolved_yes) for markets already
    # resolved by this point in the walk-forward simulation -- what
    # local_calibrated_p_yes weighs by proximity to a new entry's price.
    resolved_history: list[tuple[float, bool]] = []

    cash = start_bankroll
    committed: dict[int, float] = {}
    committed_by_bucket: dict[int, float] = {}
    equity_series = []
    n_taken = n_skip_noedge = n_skip_capital = n_wins_taken = n_losses_taken = 0
    trade_records = []

    for t, order, kind, r in events:
        if kind == "entry":
            price = r["yes_price"]
            fee_frac = 0.0 if fee_mode == "maker" else pmf.taker_fee_frac_of_notional(price, FEE_CATEGORY)
            bucket = price_bucket(price)  # risk-cap bucketing only, not calibration -- see module docstring
            p = local_calibrated_p_yes(price, resolved_history, prior_strength, bandwidth)
            equity = cash + sum(committed.values())
            f_kelly = kelly_fraction(p, price, fee_frac)
            if f_kelly <= 0:
                n_skip_noedge += 1
                continue
            desired = f_kelly * fraction * equity
            desired = min(desired, max_pos_pct * equity)
            total_committed = sum(committed.values())
            bucket_committed = committed_by_bucket.get(bucket, 0.0)
            room_agg = agg_cap_pct * equity - total_committed
            room_bucket = bucket_cap_pct * equity - bucket_committed
            liquidity_cap = r["first_trade_notional"] * liquidity_cap_multiple
            stake = min(desired, room_agg, room_bucket, cash, liquidity_cap)
            if stake <= 1e-6:
                n_skip_capital += 1
                continue
            cash -= stake
            committed[id(r)] = stake
            committed_by_bucket[bucket] = committed_by_bucket.get(bucket, 0.0) + stake
            r["_stake"] = stake
            r["_price"] = price
            r["_fee_frac"] = fee_frac
            r["_bucket"] = bucket
            r["_p_hat_at_entry"] = p
            n_taken += 1
            if track_trades:
                r["_entry_equity"] = equity
        else:
            # The outcome is public information regardless of whether we had
            # capital in this specific market -- every resolved entry feeds
            # the calibration history, same convention as run_kelly_backtest.py's
            # own bucket_flips/bucket_n update.
            resolved_history.append((r["yes_price"], r["resolved_yes"]))

            key = id(r)
            if key not in committed:
                equity_series.append((t, cash + sum(committed.values())))
                continue
            stake = committed.pop(key)
            committed_by_bucket[r["_bucket"]] -= stake
            price, fee_frac = r["_price"], r["_fee_frac"]
            b = (1.0 - price) / price - fee_frac
            L = 1.0 + fee_frac
            if r["resolved_yes"]:
                pnl = stake * b
                n_wins_taken += 1
            else:
                pnl = -stake * L
                n_losses_taken += 1
            cash += stake + pnl
            if track_trades:
                trade_records.append({
                    "t": r["resolution_time"][:10], "bucket": r["_bucket"], "yes_price": price,
                    "p_hat_at_entry": round(r["_p_hat_at_entry"], 4), "stake": round(stake, 2),
                    "stake_pct_of_equity": round(stake / r["_entry_equity"] * 100, 3),
                    "pnl": round(pnl, 2), "won": r["resolved_yes"], "question": r["question"],
                })
        equity_series.append((t, cash + sum(committed.values())))

    final_equity = cash + sum(committed.values())
    if not equity_series:
        return None

    start_day = equity_series[0][0].date()
    end_day = equity_series[-1][0].date()
    by_day = {}
    for t, eq in equity_series:
        by_day[t.date()] = eq
    daily = []
    d = start_day
    last_eq = start_bankroll
    while d <= end_day:
        if d in by_day:
            last_eq = by_day[d]
        daily.append((d, last_eq))
        d += timedelta(days=1)

    span_days = (end_day - start_day).days
    cagr = (final_equity / start_bankroll) ** (365.0 / span_days) - 1 if span_days > 0 and final_equity > 0 else float("nan")

    peak = -math.inf
    max_dd = 0.0
    for _, eq in daily:
        peak = max(peak, eq)
        dd = (peak - eq) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    rets = []
    for i in range(1, len(daily)):
        e0, e1 = daily[i - 1][1], daily[i][1]
        if e0 > 0:
            rets.append(e1 / e0 - 1)
    mean_r = sum(rets) / len(rets) if rets else 0
    var_r = sum((x - mean_r) ** 2 for x in rets) / len(rets) if rets else 0
    std_r = math.sqrt(var_r)
    sharpe = (mean_r / std_r * math.sqrt(365)) if std_r > 0 else float("nan")

    return {
        "fraction": fraction, "fee_mode": fee_mode, "prior_strength": prior_strength, "bandwidth": bandwidth,
        "start_bankroll": start_bankroll, "liquidity_cap_multiple": liquidity_cap_multiple,
        "final_equity": round(final_equity, 2),
        "total_return_pct": round((final_equity / start_bankroll - 1) * 100, 2),
        "cagr_pct": round(cagr * 100, 2) if not math.isnan(cagr) else None,
        "max_drawdown_pct": round(max_dd * 100, 2),
        "sharpe": round(sharpe, 2) if not math.isnan(sharpe) else None,
        "n_taken": n_taken, "n_skip_noedge": n_skip_noedge, "n_skip_capital": n_skip_capital,
        "n_wins_taken": n_wins_taken, "n_losses_taken": n_losses_taken,
        "span_days": span_days,
        "daily_series": [(d.isoformat(), round(eq, 2)) for d, eq in daily],
        "trade_records": trade_records if track_trades else None,
    }


def bucket_calibration_report(entries: list[dict]) -> dict:
    """Whole-sample (not walk-forward -- this is a diagnostic, not a
    trading signal) empirical YES rate per price bucket, so calibration can
    be inspected directly rather than only inferred from the backtest's P&L."""
    buckets: dict[int, dict] = {}
    for r in entries:
        b = price_bucket(r["yes_price"], BUCKET_WIDTH)
        agg = buckets.setdefault(b, {"n": 0, "wins": 0})
        agg["n"] += 1
        agg["wins"] += 1 if r["resolved_yes"] else 0
    out = {}
    for b in sorted(buckets):
        n, wins = buckets[b]["n"], buckets[b]["wins"]
        lo, hi = b * BUCKET_WIDTH, (b + 1) * BUCKET_WIDTH
        out[f"{lo:.0%}-{hi:.0%}"] = {"n": n, "empirical_yes_rate": round(wins / n, 4) if n else None,
                                      "bucket_midpoint": round((lo + hi) / 2, 4)}
    return out


def main():
    entries = load_entries()
    print(f"[politics-kelly] {len(entries)} politics market entries loaded")
    print(f"[politics-kelly] span: {min(r['entry_dt'] for r in entries).date()} to {max(r['resolve_dt'] for r in entries).date()}")

    calibration = bucket_calibration_report(entries)
    print("\n=== Whole-sample calibration by entry-price bucket (diagnostic, not walk-forward) ===")
    for band, stats in calibration.items():
        print(f"  {band:<10} n={stats['n']:<6} empirical_yes_rate={stats['empirical_yes_rate']}  (bucket implies ~{stats['bucket_midpoint']:.0%})")

    results = {"maker": {}, "taker": {}}
    print(f"\n=== Fraction sweep (prior_strength={DEFAULT_PRIOR_STRENGTH}) ===")
    for fee_mode in ["maker", "taker"]:
        for frac in FRACTIONS:
            res = run_sim([dict(r) for r in entries], frac, fee_mode)
            results[fee_mode][str(frac)] = res
            print(f"  {fee_mode:<6} frac={frac:<8} final=${res['final_equity']:>12,.0f}  "
                  f"CAGR={res['cagr_pct']:>7.1f}%  MaxDD={res['max_drawdown_pct']:>6.1f}%  Sharpe={res['sharpe']}  "
                  f"taken={res['n_taken']:<6} skip_noedge={res['n_skip_noedge']:<6} skip_cap={res['n_skip_capital']}")

    print(f"\n=== Prior-strength sensitivity (maker fees, quarter-Kelly, bandwidth={DEFAULT_BANDWIDTH}) ===")
    prior_sweep = {}
    for ps in PRIOR_STRENGTH_SWEEP:
        res = run_sim([dict(r) for r in entries], 0.25, "maker", prior_strength=ps)
        prior_sweep[str(ps)] = res
        print(f"  prior_strength={ps:<6} final=${res['final_equity']:>12,.0f}  CAGR={res['cagr_pct']:>7.1f}%  "
              f"MaxDD={res['max_drawdown_pct']:>6.1f}%  Sharpe={res['sharpe']}  taken={res['n_taken']}")
    results["prior_strength_sweep"] = prior_sweep

    print(f"\n=== Kernel-bandwidth sensitivity (maker fees, quarter-Kelly, prior_strength={DEFAULT_PRIOR_STRENGTH}) ===")
    bandwidth_sweep = {}
    for bw in BANDWIDTH_SWEEP:
        res = run_sim([dict(r) for r in entries], 0.25, "maker", bandwidth=bw)
        bandwidth_sweep[str(bw)] = res
        print(f"  bandwidth={bw:<6} final=${res['final_equity']:>12,.0f}  CAGR={res['cagr_pct']:>7.1f}%  "
              f"MaxDD={res['max_drawdown_pct']:>6.1f}%  Sharpe={res['sharpe']}  taken={res['n_taken']}")
    results["bandwidth_sweep"] = bandwidth_sweep

    print(f"\n=== Liquidity-cap stress test (maker fees, quarter-Kelly): what if stakes weren't capped by the "
          f"market's own first-trade size? ===")
    liq_capped = run_sim([dict(r) for r in entries], 0.25, "maker", liquidity_cap_multiple=1.0)
    liq_uncapped = run_sim([dict(r) for r in entries], 0.25, "maker", liquidity_cap_multiple=float("inf"))
    print(f"  capped (1.0x first-trade size): final=${liq_capped['final_equity']:>12,.0f}  CAGR={liq_capped['cagr_pct']:.1f}%  taken={liq_capped['n_taken']}")
    print(f"  uncapped (bankroll-sized only):  final=${liq_uncapped['final_equity']:>12,.0f}  CAGR={liq_uncapped['cagr_pct']:.1f}%  taken={liq_uncapped['n_taken']}")
    results["liquidity_cap_stress_test"] = {"capped": liq_capped, "uncapped": liq_uncapped}

    print(f"\n=== Bankroll sweep (quarter-Kelly, maker fees, liquidity-capped) ===")
    bankroll_sweep = {}
    for bankroll in BANKROLL_SWEEP:
        res = run_sim([dict(r) for r in entries], 0.25, "maker", start_bankroll=bankroll, track_trades=True)
        bankroll_sweep[str(bankroll)] = res
        print(f"  bankroll=${bankroll:<10,.0f} final=${res['final_equity']:>12,.0f}  total_return={res['total_return_pct']:>7.1f}%  "
              f"CAGR={res['cagr_pct']:>7.1f}%  MaxDD={res['max_drawdown_pct']:>6.1f}%  Sharpe={res['sharpe']}  "
              f"taken={res['n_taken']:<6} skip_cap={res['n_skip_capital']}")
    results["bankroll_sweep"] = bankroll_sweep
    results["recommended"] = bankroll_sweep[str(START_BANKROLL)]
    results["calibration_report"] = calibration
    results["n_entries"] = len(entries)

    out_path = RESULTS_DIR / "politics_kelly_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, default=str)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
