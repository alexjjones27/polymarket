"""Same momentum/reversal backtest as run_momentum_backtest.py, applied to
a different market family: MLB game moneylines, to answer "does this
generalize beyond crypto." Reuses simulate_market and run_variant
unchanged -- only the population and the live-window definition differ.

Population: MLB moneyline markets ("Team A vs. Team B", the plain
two-team event market, not its Nth-inning-winner or spread siblings which
share the same event). Live window: Polymarket lists these markets for
pre-game trading days in advance, but a live-play momentum signal should
only look at the live window -- approximated here as the LAST 4 hours
before the game's closedTime (an MLB game runs ~3 hours; 4 hours gives
margin for extras/delays), not the market's full multi-day listing
lifetime. This is a real, disclosed methodological choice, not the
market's true first-pitch time (which Gamma does not expose directly).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import polymarket_final_pct as pmf
from run_momentum_backtest import run_variant, HOLDING_PERIODS_S

REPO = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO / "results" / "polymarket_final_pct"
LIVE_WINDOW_S = 4 * 3600


def fetch_mlb_population(max_events=2000):
    cache_path = REPO / "data" / "raw" / "polymarket" / "momentum_mlb_population.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    print("[momentum-mlb] paging /events?series_slug=mlb&closed=true ...")
    all_events = []
    for offset in range(0, pmf.GAMMA_OFFSET_CAP, 100):
        page = pmf._get(pmf.GAMMA_BASE, "/events", {
            "closed": "true", "limit": 100, "offset": offset, "series_slug": "mlb",
            "order": "startDate", "ascending": "false",
        })
        if not page:
            break
        all_events.extend(page)
        if len(all_events) >= max_events:
            break

    population = []
    for e in all_events:
        for mk in e.get("markets", []):
            q = mk.get("question", "")
            # only the plain "Team A vs. Team B" moneyline market -- its
            # siblings (Nth-inning winner/tied, spread, O/U) use different
            # phrasing ("X to win the 2nd inning?", "Spread: X (-1.5)",
            # "X vs Y: O/U 8.5") that never matches this exact pattern
            if not pmf.re.search(r"^[\w .'-]+ vs\.? [\w .'-]+$", q.strip()):
                continue
            outcomes = pmf._safe_json_list(mk.get("outcomes"))
            token_ids = pmf._safe_json_list(mk.get("clobTokenIds"))
            if len(outcomes) != 2 or len(token_ids) != 2:
                continue
            idx = pmf.resolved_outcome_index(mk)
            if idx is None:
                continue
            closed_time = mk.get("closedTime")
            if not closed_time:
                continue
            end_s = pmf._to_epoch_s(closed_time)
            if end_s is None:
                continue
            start_s = end_s - LIVE_WINDOW_S
            if start_s < pmf._to_epoch_s(pmf.CLOB_LAUNCH_CUTOFF):
                continue
            population.append({
                "conditionId": mk["conditionId"], "question": q, "token_id": token_ids[0],
                "asset": "mlb", "start_s": start_s, "resolution_time": pmf.pd.Timestamp(closed_time, tz="UTC").isoformat(),
                "resolved_yes": outcomes[idx] == outcomes[0],
            })

    print(f"[momentum-mlb] {len(all_events)} events scanned, {len(population)} plain moneyline markets found")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(population))
    return population


def main():
    population = fetch_mlb_population()
    print(f"[momentum-mlb] {len(population)} MLB moneyline markets")
    sample = population  # small enough population that a full run is affordable

    results = {}
    t0 = time.time()
    for direction in ["momentum", "reversal"]:
        for holding_s in [15 * 60, 30 * 60]:
            key = f"{direction}_{holding_s//60}min"
            print(f"[momentum-mlb] running {key} (no trailing stop, the crypto backtest's best config) ...", flush=True)
            r = run_variant(sample, direction, holding_s, trailing_stop_delta=1.0)
            results[key] = r
            print(f"  {key}: n={r['n_trades']}  win_rate={r.get('win_rate')}  "
                  f"total_pnl=${r.get('total_pnl')}  mean_ret={r.get('mean_return_frac')}  "
                  f"({time.time()-t0:.0f}s elapsed)", flush=True)

    out_path = RESULTS_DIR / "momentum_backtest_mlb_results.json"
    with open(out_path, "w") as f:
        json.dump({"n_population": len(population), "variants": results}, f, indent=2, default=str)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
