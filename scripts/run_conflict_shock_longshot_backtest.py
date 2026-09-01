"""Tests a specific refinement of the favorite-longshot-bias question: does
buying the LONGSHOT side pay off in markets exposed to a sudden, discrete
geopolitical/military shock (a strike, invasion, ceasefire break, coup)
that can flip a near-certain "no" into "yes" overnight -- as opposed to the
general population, where run_longshot_buy_backtest.py already found buying
longshots is decisively negative EV (-84.9% ROI, 0.13% realized win rate)?

Motivating anecdote: a real Polymarket market on whether the US and Iran
would break a ceasefire by a given date traded at ~99.1% "No" and then
flipped to "Yes" within the window after a US strike -- exactly the "flip"
event type this project's flip-rate analysis already treats as the whole
point of the Final-1% strategy's tail risk, just viewed from the other side
of the trade.

Population: NOT the general stratified sample -- a keyword-targeted filter
over the FULL resolved-market census for questions about strikes,
invasions, ceasefires, coups, and related discrete conflict actions (
"filtering out the unrealistic ones" per the task: this keeps markets whose
underlying event is a specific, dateable military/geopolitical action, not
routine sports/weather/price variance or vague geopolitical commentary).
Runs the exact same no-lookahead crossing detection, simulate_trade, and
flip analysis as the Final-1% backtest, just on this different population,
so results are directly comparable.

Same honesty standard as the rest of this project: this is a narrow,
low-volume market category, so the sample of markets that actually cross
the threshold will be small and the resulting confidence intervals will be
wide. That is the actual finding to take seriously, not a precise flip-rate
point estimate.
"""
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import polymarket_final_pct as pmf

RESULTS_DIR = pmf.RESULTS_DIR
THRESHOLD = 0.99
N_CONSECUTIVE = 3

# Discrete, dateable military/geopolitical actions -- a strike, invasion,
# ceasefire break, coup, missile launch. Deliberately narrower than the
# general "politics" bucket (which also catches routine elections) and
# narrower than a bare "war" keyword (which would catch trade-war/price-war
# usage): every phrase here names a specific, sudden action a government or
# military can take, not a general state of affairs or an election outcome.
CONFLICT_RE = re.compile(
    r"\b(ceasefire|cease-fire|cease fire|invade|invasion|military strike|airstrike|air strike|"
    r"strikes? on|attack on|bomb(ing)?|troops? (deploy|enter|cross)|missile launch|"
    r"nuclear (attack|strike)|coup|assassinat|armistice|war powers|"
    r"peace (deal|treaty|agreement)|invades?)\b",
    re.IGNORECASE,
)


def filter_conflict_markets(census: list[dict]) -> list[dict]:
    df_filtered = [
        m for m in census
        if CONFLICT_RE.search(str(m.get("question", "")))
        and pmf._to_epoch_s(m.get("startDate") or m.get("createdAt")) is not None
        and pmf._to_epoch_s(m.get("startDate") or m.get("createdAt")) >= pmf._to_epoch_s(pmf.CLOB_LAUNCH_CUTOFF)
        and pmf._safe_json_list(m.get("clobTokenIds"))
    ]
    return df_filtered


def mirror_longshot_trade(trade: dict) -> dict:
    """Buying the complementary (longshot) side of this same crossing --
    same approximation run_longshot_buy_backtest.py uses and documents:
    entry price ~= 1 - original_entry_price, wins iff the original flipped.
    Reasonable given original entries sit at >=0.99, so the true complement
    price is within a cent or two of this."""
    longshot_price = max(1e-4, 1.0 - trade["entry_price"])
    shares = trade["notional"] / longshot_price  # same $ notional as the original trade
    won = not trade["won"]
    payout = shares * 1.0 if won else 0.0
    return {
        "market_id": trade["market_id"], "question": trade["question"],
        "longshot_entry_price": longshot_price, "shares": shares,
        "notional": trade["notional"], "won": won, "payout": payout,
        "pnl": payout - trade["notional"],
    }


def main():
    print("Fetching (cached) census of resolved markets ...")
    census = pmf.fetch_resolved_markets_census()
    print(f"  census size: {len(census):,}")

    conflict_markets = filter_conflict_markets(census)
    print(f"  {len(conflict_markets):,} markets match the conflict/shock keyword filter "
          f"(post CLOB-cutoff, with CLOB token ids)")

    signal_cfg = pmf.SignalConfig(threshold=THRESHOLD, n_consecutive=N_CONSECUTIVE)
    cfg = pmf.BacktestConfig(signal=signal_cfg, position_notional=100.0,
                              gas=pmf.GasAssumptions(relayer_sponsored=True))
    fill = pmf.FillAssumptions(fill_type="maker")

    print(f"Running crossing detection on {len(conflict_markets)} markets ...")
    all_crossings = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(pmf.find_market_crossings, m, signal_cfg): m for m in conflict_markets}
        for fut in as_completed(futures):
            market = futures[fut]
            for crossing in fut.result():
                all_crossings.append((market, crossing))

    print(f"  {len(all_crossings)} crossings found across {len(conflict_markets)} markets")

    trades = [pmf.simulate_trade(crossing, market, fill, cfg, cap_shares=None) for market, crossing in all_crossings]
    trades_df = pmf.pd.DataFrame(trades)

    if trades_df.empty:
        print("No crossings found in this population at all -- nothing to report.")
        return

    metrics = pmf.compute_metrics(trades_df, "pnl_net")
    print(f"\n=== Favorite side (buy >=99%, same mechanic as Final-1%) ===")
    wilson = metrics["flip_rate_wilson_95"]
    cp = metrics["flip_rate_clopper_pearson_95"]
    print(f"n_trades={metrics['n_trades']}  n_flips={metrics['n_flips']}  "
          f"flip_rate={metrics['flip_rate']*100:.2f}%  "
          f"wilson_95=({wilson[0]*100:.2f}%, {wilson[1]*100:.2f}%)  "
          f"clopper_pearson_95=({cp[0]*100:.2f}%, {cp[1]*100:.2f}%)")
    print(f"total_pnl_net=${metrics['total_pnl']:.2f}  total_notional=${metrics['total_notional']:.2f}  "
          f"total_return={metrics['total_return']*100:.3f}%")

    markets_by_id = {str(m["id"]): m for m in conflict_markets}
    flips_df = pmf.analyze_flips(trades_df, markets_by_id)
    print(f"\n=== Every flip found (the actual 'shock' events) ===")
    if flips_df.empty:
        print("  (none)")
    else:
        for _, r in flips_df.iterrows():
            print(f"  {str(r['entry_time'])[:10]}  entry=${r['entry_price']:.4f}  "
                  f"held {r['holding_days']:.2f}d  {r['question']}")

    print(f"\n=== Mirror: buying the LONGSHOT side of every crossing instead ===")
    mirrors = [mirror_longshot_trade(t) for t in trades]
    n_wins = sum(1 for m in mirrors if m["won"])
    total_notional = sum(m["notional"] for m in mirrors)
    total_pnl = sum(m["pnl"] for m in mirrors)
    avg_entry = sum(m["longshot_entry_price"] for m in mirrors) / len(mirrors) if mirrors else float("nan")
    print(f"n_trades={len(mirrors)}  n_wins={n_wins}  win_rate={n_wins/len(mirrors)*100:.2f}%  "
          f"avg_entry_price(~breakeven_rate_needed)={avg_entry*100:.3f}%")
    print(f"total_staked=${total_notional:.2f}  total_pnl=${total_pnl:.2f}  "
          f"roi={total_pnl/total_notional*100:.2f}%" if total_notional else "no trades")
    if n_wins:
        print("  Winning longshot trades:")
        for m in mirrors:
            if m["won"]:
                print(f"    entry=${m['longshot_entry_price']:.4f}  payout=${m['payout']:,.2f}  {m['question']}")

    out = {
        "n_census": len(census),
        "n_conflict_markets": len(conflict_markets),
        "n_crossings": len(all_crossings),
        "favorite_side_metrics": {k: v for k, v in metrics.items()},
        "flips": flips_df.to_dict("records") if not flips_df.empty else [],
        "longshot_mirror": {
            "n_trades": len(mirrors), "n_wins": n_wins,
            "win_rate_pct": round(n_wins / len(mirrors) * 100, 3) if mirrors else None,
            "avg_entry_price_pct": round(avg_entry * 100, 4) if mirrors else None,
            "total_staked": round(total_notional, 2), "total_pnl": round(total_pnl, 2),
            "roi_pct": round(total_pnl / total_notional * 100, 3) if total_notional else None,
            "trades": mirrors,
        },
    }
    out_path = RESULTS_DIR / "conflict_shock_longshot_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
