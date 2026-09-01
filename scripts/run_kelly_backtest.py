import csv, re, json, math, os
from datetime import datetime, timedelta

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "results", "polymarket_final_pct")


def parse(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


EXACT_SCORE_RE = re.compile(r'^Exact Score:', re.I)
WEATHER_RE = re.compile(r'highest temperature.*(be between|be \d)', re.I)


def load(fn):
    with open(os.path.join(DATA_DIR, fn), newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    # A rare data quirk in the lower-threshold sweep files: one market has
    # `won` populated but an empty resolution_time (endDate parsing miss
    # upstream). Dropped defensively rather than crashing -- trades_maker.csv
    # itself has zero such rows, so this is a no-op there.
    rows = [r for r in rows if r.get('entry_time') and r.get('resolution_time')]
    for r in rows:
        r['entry_dt'] = parse(r['entry_time'])
        r['resolve_dt'] = parse(r['resolution_time'])
        r['entry_price'] = float(r['entry_price'])
        r['won'] = r['won'] == 'True'
        r['fee_frac'] = float(r['fee_frac'])
        r['depth_capped'] = r['depth_capped'] == 'True'
        r['cap_shares'] = float(r['cap_shares']) if r['cap_shares'] not in ('', None) else None
        r['excluded'] = bool(EXACT_SCORE_RE.search(r['question'])) or bool(WEATHER_RE.search(r['question']))
    return rows


def flip_counts_by(trades: list[dict], key_field: str) -> dict[str, tuple[int, int]]:
    """(n_flips, n_trades) per distinct value of `key_field`, over tradeable
    (non-excluded) rows only. Used by the live scanners (scan_live_signals.py,
    scan_live_signals_70.py) to derive their Beta-prior flip counts directly
    from a committed trades CSV at run time, instead of a hand-copied
    snapshot that silently goes stale the next time this backtest is
    re-run."""
    out: dict[str, tuple[int, int]] = {}
    for r in trades:
        if r["excluded"]:
            continue
        k = r[key_field]
        flips, n = out.get(k, (0, 0))
        out[k] = (flips + (0 if r["won"] else 1), n + 1)
    return out


PRIOR_A, PRIOR_B = 1.0, 300.0
MAX_POS_PCT = 0.03
AGG_CAP_PCT = 0.50
CAT_CAP_PCT = 0.25
START_BANKROLL = 10000.0
FRACTIONS = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]


def run_sim(all_trades, fraction, max_pos_pct=MAX_POS_PCT, agg_cap_pct=AGG_CAP_PCT, cat_cap_pct=CAT_CAP_PCT,
            track_trades=False, prior_a=PRIOR_A, prior_b=PRIOR_B):
    tradeable = [r for r in all_trades if not r['excluded']]
    events = []
    for r in tradeable:
        # Never let a resolve event sort before its own entry event: ~54 trades in the
        # source data have a resolution_time a few seconds earlier than entry_time (a
        # snapshot-granularity artifact on very fast-resolving markets like "Up or Down"
        # crypto minute bets). Left unclamped, that silently leaks committed capital that
        # never gets returned to the bankroll for the rest of the simulation.
        resolve_key_t = max(r['resolve_dt'], r['entry_dt'])
        events.append((r['entry_dt'], 0, 'entry', r))
        events.append((resolve_key_t, 1, 'resolve', r))
    events.sort(key=lambda e: (e[0], e[1]))

    bucket_flips = {}
    bucket_n = {}

    def qh(bucket):
        k = bucket_flips.get(bucket, 0)
        n = bucket_n.get(bucket, 0)
        return (prior_a + k) / (prior_a + prior_b + n)

    cash = START_BANKROLL
    committed = {}
    committed_by_bucket = {}
    equity_series = []
    n_taken = 0
    n_skip_noedge = 0
    n_skip_capital = 0
    n_flips_taken = 0
    n_wins_taken = 0
    trade_records = []

    for t, order, kind, r in events:
        if kind == 'entry':
            bucket = r['report_bucket']
            price = r['entry_price']
            L = 1.0 + r['fee_frac']
            b = (1.0 - price) / price - r['fee_frac']
            equity = cash + sum(committed.values())
            if b <= 0:
                n_skip_noedge += 1
                continue
            q = qh(bucket)
            p = 1.0 - q
            f_kelly = (p * b - q * L) / (b * L)
            if f_kelly <= 0:
                n_skip_noedge += 1
                continue
            desired = f_kelly * fraction * equity
            desired = min(desired, max_pos_pct * equity)
            if r['depth_capped'] and r['cap_shares'] is not None:
                desired = min(desired, r['cap_shares'] * price)
            total_committed = sum(committed.values())
            bucket_committed = committed_by_bucket.get(bucket, 0.0)
            room_agg = agg_cap_pct * equity - total_committed
            room_cat = cat_cap_pct * equity - bucket_committed
            stake = min(desired, room_agg, room_cat, cash)
            if stake <= 1e-6:
                n_skip_capital += 1
                continue
            cash -= stake
            committed[id(r)] = stake
            committed_by_bucket[bucket] = committed_by_bucket.get(bucket, 0.0) + stake
            r['_stake'] = stake
            r['_b'] = b
            r['_L'] = L
            n_taken += 1
            if track_trades:
                r['_entry_equity'] = equity
        else:
            # Update the walk-forward flip-rate belief from every resolved tradeable
            # market, whether or not we had capital in that specific trade -- the
            # outcome is public information regardless of whether we sized a position.
            bucket = r['report_bucket']
            bucket_n[bucket] = bucket_n.get(bucket, 0) + 1
            if not r['won']:
                bucket_flips[bucket] = bucket_flips.get(bucket, 0) + 1

            key = id(r)
            if key not in committed:
                equity_series.append((t, cash + sum(committed.values())))
                continue
            stake = committed.pop(key)
            committed_by_bucket[bucket] -= stake
            if r['won']:
                pnl = stake * r['_b']
                n_wins_taken += 1
            else:
                pnl = -stake * r['_L']
                n_flips_taken += 1
            cash += stake + pnl
            if track_trades:
                trade_records.append({
                    "t": r['resolution_time'][:10], "bucket": bucket, "stake": round(stake, 2),
                    "stake_pct_of_equity": round(stake / r['_entry_equity'] * 100, 3),
                    "pnl": round(pnl, 2), "won": r['won'], "question": r['question'],
                    "depth_capped": r['depth_capped'],
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
    last_eq = START_BANKROLL
    while d <= end_day:
        if d in by_day:
            last_eq = by_day[d]
        daily.append((d, last_eq))
        d += timedelta(days=1)

    span_days = (end_day - start_day).days
    cagr = (final_equity / START_BANKROLL) ** (365.0 / span_days) - 1 if span_days > 0 else float('nan')

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
    sharpe = (mean_r / std_r * math.sqrt(365)) if std_r > 0 else float('nan')

    return {
        "fraction": fraction,
        "final_equity": round(final_equity, 2),
        "total_return_pct": round((final_equity / START_BANKROLL - 1) * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "sharpe": round(sharpe, 2) if not math.isnan(sharpe) else None,
        "n_taken": n_taken, "n_skip_noedge": n_skip_noedge, "n_skip_capital": n_skip_capital,
        "n_wins_taken": n_wins_taken, "n_flips_taken": n_flips_taken,
        "daily_series": [(d.isoformat(), round(eq, 2)) for d, eq in daily],
        "span_days": span_days,
        "trade_records": trade_records if track_trades else None,
    }


def main():
    maker_trades = load("trades_maker.csv")
    taker_trades = load("trades_taker.csv")

    n_total = len(maker_trades)
    n_excluded = sum(1 for r in maker_trades if r['excluded'])
    flips_excluded = sum(1 for r in maker_trades if r['excluded'] and not r['won'])
    flips_tradeable = sum(1 for r in maker_trades if not r['excluded'] and not r['won'])
    print(f"total={n_total} excluded={n_excluded} (flips in excluded={flips_excluded}) tradeable_flips={flips_tradeable}")

    results = {"maker": {}, "taker": {}}
    for frac in FRACTIONS:
        res_m = run_sim([dict(r) for r in maker_trades], frac)
        res_t = run_sim([dict(r) for r in taker_trades], frac)
        results["maker"][str(frac)] = res_m
        results["taker"][str(frac)] = res_t
        print(f"frac={frac:<8} MAKER final=${res_m['final_equity']:>12,.0f}  CAGR={res_m['cagr_pct']:>7.1f}%  MaxDD={res_m['max_drawdown_pct']:>6.1f}%  Sharpe={res_m['sharpe']}  taken={res_m['n_taken']} skip_cap={res_m['n_skip_capital']} flips_taken={res_m['n_flips_taken']}")

    print("\n--- max-position-cap sweep (fraction fixed at 0.25, the recommended quarter-Kelly) ---")
    pos_sweep = {}
    for max_pos in [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]:
        res = run_sim([dict(r) for r in maker_trades], 0.25, max_pos_pct=max_pos)
        pos_sweep[str(max_pos)] = res
        print(f"max_pos={max_pos:<6} final=${res['final_equity']:>12,.0f}  CAGR={res['cagr_pct']:>7.1f}%  MaxDD={res['max_drawdown_pct']:>6.1f}%  Sharpe={res['sharpe']}")
    results["position_cap_sweep"] = pos_sweep

    print("\n--- stress test: naive Kelly with NO hard risk caps (max_pos=100%, agg=100%, cat=100%) ---")
    naive = run_sim([dict(r) for r in maker_trades], 1.0, max_pos_pct=1.0, agg_cap_pct=1.0, cat_cap_pct=1.0)
    print(f"naive uncapped full-Kelly: final=${naive['final_equity']:>12,.0f}  CAGR={naive['cagr_pct']:.1f}%  MaxDD={naive['max_drawdown_pct']:.1f}%  Sharpe={naive['sharpe']}  skip_cap={naive['n_skip_capital']}")
    results["naive_uncapped_stress"] = naive

    print("\n--- recommended configuration, trade-level detail (quarter-Kelly, standard caps) ---")
    recommended = run_sim([dict(r) for r in maker_trades], 0.25, track_trades=True)
    print(f"recommended: final=${recommended['final_equity']:>12,.0f}  CAGR={recommended['cagr_pct']:.1f}%  MaxDD={recommended['max_drawdown_pct']:.1f}%  Sharpe={recommended['sharpe']}  n_trades={recommended['n_taken']}")
    results["recommended"] = recommended

    out_path = os.path.join(DATA_DIR, "kelly_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f)
    print("saved", out_path)


if __name__ == "__main__":
    main()
