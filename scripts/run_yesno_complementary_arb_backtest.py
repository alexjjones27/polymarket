"""Backtest: complementary YES/NO (two-outcome) arbitrage.

Anomaly #1 from the documented-anomalies list: every Polymarket two-outcome
market settles its pair to exactly $1 via the underlying CTF condition,
NegRisk grouping or not -- this applies to literally every binary market on
the platform (Yes/No, Over/Under, TeamA/TeamB), not just the explicit
NegRisk baskets scripts/run_negrisk_arb_full_backtest.py already covers.
Two mechanical conditions:
  ask_0 + ask_1 < 1   -> buy both outcomes, redeem the pair for a
                          guaranteed $1 regardless of which one resolves
  bid_0 + bid_1 > 1   -> mint a complementary pair for $1 (CTF split),
                          sell both for more than $1

This backtest tests the buy-both condition historically, using the
project's established price-snapshot proxy (`/prices-history`, the last
observed CLOB snapshot, not a true historical ask -- the same disclosed
limitation as every other backtest here, since no historical order-book
depth exists for a resolved market). The mint-and-sell condition needs
historical bid data this proxy can't supply; scan_negrisk_arb_live.py's
sibling live scanner covers a live version of both sides instead, where a
real order book exists.

Population: every two-outcome resolved market in the full census, not a
NegRisk-only or Yes/No-labeled-only subset, post-CLOB-cutoff, cleanly
resolved. Stratified sample by quarter x category, same method and same
target size discipline as the Final-1% population, for comparability with
the rest of this project's results.
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import polymarket_final_pct as pmf

REPO = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO / "results" / "polymarket_final_pct"

SAMPLE_TARGET = 3000
FIDELITY_MIN = 60          # 1h bars, this project's established "coarse full-lifetime" convention
N_CONSECUTIVE = 2          # persistence filter, same as the Final-1% and NegRisk signals
FETCH_WORKERS = 24
STALENESS_TOLERANCE_S = 20 * 60  # both sides' last print must be within 20min of each other to count as simultaneous


def _slim_two_outcome_meta(m: dict) -> dict | None:
    """None unless m is a cleanly-resolved, post-CLOB, exactly-two-outcome
    market with both CLOB token ids present -- kept minimal (not the full
    Gamma dict) so streaming the ~750k-market census doesn't repeat the
    documented ~9GB in-memory risk."""
    outcomes = pmf._safe_json_list(m.get("outcomes"))
    if len(outcomes) != 2:
        return None
    token_ids = pmf._safe_json_list(m.get("clobTokenIds"))
    if len(token_ids) != 2:
        return None
    cid = m.get("conditionId")
    if not cid:
        return None
    idx = pmf.resolved_outcome_index(m)
    if idx is None:
        return None
    res_ts = pmf._resolution_timestamp(m)
    if res_ts is None:
        return None
    start_ts = pmf._to_epoch_s(m.get("startDate") or m.get("createdAt"))
    if start_ts is None or start_ts < pmf._to_epoch_s(pmf.CLOB_LAUNCH_CUTOFF):
        return None
    events = m.get("events")
    return {
        "conditionId": cid, "question": m.get("question", ""), "slug": m.get("slug", ""),
        "events": events if isinstance(events, list) else [],
        "outcomes": outcomes, "token_ids": token_ids,
        "endDate": m.get("endDate"), "startDate": m.get("startDate"), "createdAt": m.get("createdAt"),
        "resolution_time": res_ts.isoformat(), "resolved_outcome": outcomes[idx],
    }


def stream_two_outcome_population(cache_dir: Path = pmf.GAMMA_CACHE_DIR) -> list[dict]:
    leaf_files = sorted(cache_dir.glob("leaf_*.json"))
    print(f"[yesno-arb] streaming {len(leaf_files)} cached census leaf files (no new network calls) ...")
    metas: dict[str, dict] = {}
    for i, path in enumerate(leaf_files):
        raw = json.loads(path.read_text())
        for m in raw:
            meta = _slim_two_outcome_meta(m)
            if meta is not None:
                metas[meta["conditionId"]] = meta
        del raw
        if (i + 1) % 100 == 0:
            print(f"  [yesno-arb] {i + 1}/{len(leaf_files)} leaf files, {len(metas)} two-outcome markets so far", flush=True)
    return list(metas.values())


def fetch_arb_windows(meta: dict) -> dict | None:
    """Fetches both outcome tokens' full price history and checks for a
    persistent (N_CONSECUTIVE straight snapshots) buy-both gap. None on a
    market with no usable overlapping price data for either token."""
    start_s = pmf._to_epoch_s(meta["startDate"] or meta["createdAt"])
    end_s = int(pmf.pd.Timestamp(meta["resolution_time"]).timestamp())
    if start_s is None or end_s <= start_s:
        return None

    series = {}
    for tok in meta["token_ids"]:
        try:
            series[tok] = pmf.fetch_price_series(tok, start_s, end_s, fidelity=FIDELITY_MIN)
        except Exception:
            series[tok] = pmf.pd.DataFrame(columns=["t", "p"])

    tok0, tok1 = meta["token_ids"]
    if series[tok0].empty or series[tok1].empty:
        return {"conditionId": meta["conditionId"], "question": meta["question"][:120], "usable": False}

    # merge_asof with a tight tolerance, NOT concat+ffill: the two tokens
    # trade independently, so a naive union-then-forward-fill pairs a fresh
    # print on one side with a stale (sometimes hours-old) one on the
    # other, and their sum drifting from 1 in that case is just two
    # asynchronous last-trade prints, not a real simultaneously-executable
    # gap. Confirmed live on a thin FDA-approval market: the two sides'
    # RAW prints never actually cross, but ffill-then-sum manufactured an
    # apparent 48%-profit gap out of prints roughly a day apart. Requiring
    # both sides within STALENESS_TOLERANCE_S of each other is the direct
    # fix, same discipline as this project's other artifact checks.
    d0 = series[tok0].rename(columns={"p": "p0"})[["t", "p0"]].sort_values("t")
    d1 = series[tok1].rename(columns={"p": "p1"})[["t", "p1"]].sort_values("t")
    combined = pmf.pd.merge_asof(d0, d1, on="t", direction="nearest", tolerance=STALENESS_TOLERANCE_S)
    combined = combined.dropna(how="any").set_index("t").sort_index()
    if combined.empty:
        return {"conditionId": meta["conditionId"], "question": meta["question"][:120], "usable": False}

    category = pmf.classify_report_bucket({"question": meta["question"], "slug": meta["slug"], "events": meta["events"]})
    fee = pmf.taker_fee_frac_of_notional
    # taker_fee_frac_of_notional returns a fraction of notional, so cost per share = price * (1 + fee_frac)
    cost = combined["p0"] * (1 + fee(combined["p0"], category)) + combined["p1"] * (1 + fee(combined["p1"], category))
    profit_frac = 1.0 - cost

    qualifies = profit_frac > 0
    run = 0
    hit_idx = None
    idx_list = qualifies.index.tolist()
    for i, ok in enumerate(qualifies.tolist()):
        run = run + 1 if ok else 0
        if run >= N_CONSECUTIVE:
            hit_idx = idx_list[i - N_CONSECUTIVE + 1]
            break

    result = {
        "conditionId": meta["conditionId"], "question": meta["question"][:120], "category": category,
        "usable": True, "n_grid_points": len(combined), "max_profit_frac": float(profit_frac.max()),
        "arb_found": hit_idx is not None,
    }
    if hit_idx is not None:
        result.update({
            "entry_time": pmf.pd.Timestamp(hit_idx, unit="s", tz="UTC").isoformat(),
            "pair_sum": float(combined.loc[hit_idx].sum()),
            "profit_frac": float(profit_frac.loc[hit_idx]),
        })
    return result


def main():
    sample_cache = REPO / "data" / "raw" / "polymarket" / "yesno_arb_sample.json"
    if sample_cache.exists():
        sample = json.loads(sample_cache.read_text())
        print(f"[yesno-arb] loaded cached stratified sample: {len(sample)} markets ({sample_cache})")
    else:
        population = stream_two_outcome_population()
        print(f"[yesno-arb] {len(population)} cleanly-resolved, two-outcome, post-CLOB markets in the census")
        sample = pmf.stratified_sample_markets(population, SAMPLE_TARGET)
        print(f"[yesno-arb] stratified sample: {len(sample)} markets (quarter x category)")
        sample_cache.parent.mkdir(parents=True, exist_ok=True)
        sample_cache.write_text(json.dumps(sample, default=str))

    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        futures = {pool.submit(fetch_arb_windows, m): m for m in sample}
        for i, fut in enumerate(as_completed(futures)):
            r = fut.result()
            if r is not None:
                results.append(r)
            if (i + 1) % 250 == 0:
                print(f"  [yesno-arb] {i + 1}/{len(sample)} processed ({time.time()-t0:.0f}s elapsed)", flush=True)

    usable = [r for r in results if r["usable"]]
    hits = sorted([r for r in usable if r["arb_found"]], key=lambda r: -r["profit_frac"])

    by_category = {}
    for r in usable:
        by_category.setdefault(r["category"], {"n": 0, "n_arb": 0})
        by_category[r["category"]]["n"] += 1
        if r["arb_found"]:
            by_category[r["category"]]["n_arb"] += 1

    summary = {
        "n_population": len(population),
        "n_sampled": len(sample),
        "n_usable": len(usable),
        "n_arb_windows_found": len(hits),
        "pct_markets_with_arb": round(len(hits) / len(usable) * 100, 3) if usable else None,
        "mean_profit_frac_pct": round(sum(r["profit_frac"] for r in hits) / len(hits) * 100, 4) if hits else None,
        "median_profit_frac_pct": round(sorted(r["profit_frac"] for r in hits)[len(hits)//2] * 100, 4) if hits else None,
        "by_category": by_category,
        "top_hits": hits[:40],
        "all_usable": usable,
    }

    print(f"\n=== {len(hits)}/{len(usable)} usable markets ({summary['pct_markets_with_arb']}%) showed a persistent buy-both gap ===")
    if hits:
        print(f"Mean profit at the gap: {summary['mean_profit_frac_pct']}%  |  median: {summary['median_profit_frac_pct']}%")
    for r in hits[:15]:
        print(f"  {r['question'][:50]:50s} {r['category']:12s} sum={r['pair_sum']:.4f}  profit={r['profit_frac']*100:.3f}%")

    out_path = RESULTS_DIR / "yesno_complementary_arb_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
