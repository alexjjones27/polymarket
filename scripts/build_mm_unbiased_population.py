"""Builds an unbiased market population for market-making research, fixing a
selection-bias problem in every MM analysis so far.

run_mm_proxy_backtest.py and its Q3/volume/resolution deep dives all reused
trades_maker.csv -- but that file is the FINAL-1% STRATEGY's own population:
every market in it had some outcome cross the 99%+ threshold at some point in
its life (that's the entire premise of the Final-1% longshot strategy). A
market-making desk doesn't only quote in markets that eventually go to a
near-certain extreme -- most of what it would actually trade never does. All
of the pace/volume/resolution-proximity conclusions reached so far were
therefore validated on a population that's structurally unrepresentative of
what a real MM book would look like, which is a more fundamental problem than
overfitting the filter thresholds (see run_mm_walkforward_validation.py for
that).

Fix: draw a fresh, independent, stratified-random sample (by resolution
quarter x report category, proportional allocation -- no probability/
extremity filter of any kind) from the COMPLETE, uncurated census of resolved
markets already cached on disk by the Final-1% strategy's own census crawler,
then fetch each sampled market's trade tape (new network calls, cached going
forward exactly like everywhere else in this codebase).

Memory note: the raw census is ~800k+ Gamma market objects (~4.6GB of JSON on
disk) -- materializing that many FULL market dicts in one process (as
polymarket_final_pct.fetch_resolved_markets_census() does) pushed this
environment to ~9GB+ resident and risked OOM (observed directly: killed at
9.2GB and climbing on a 15GB box with no swap). This script instead streams
each cached leaf file one at a time and immediately reduces every market to
the handful of fields actually needed for stratification and later lookup,
discarding the rest -- the slim population fits comfortably in memory.
"""
import csv
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import polymarket_final_pct as pmf

REPO = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO / "results" / "polymarket_final_pct"
OUT_PATH = RESULTS_DIR / "mm_unbiased_population.csv"

N_TARGET = 1500
SEED = 20260902  # independent of the Final-1% strategy's own sampling seed (42)
FETCH_WORKERS = 16


def _slim_market(m: dict):
    """Reduces one full Gamma market dict to only what stratification and
    later lookup need. classify_report_bucket/resolved_outcome_index/
    _resolution_timestamp are all called HERE, while the full dict (with its
    `events`/`outcomePrices` fields) is still in hand -- the slim record
    that survives never needs them again."""
    cid = m.get("conditionId")
    if not cid:
        return None
    outcome_idx = pmf.resolved_outcome_index(m)
    if outcome_idx is None:
        return None  # not cleanly resolved (e.g. voided/ambiguous)
    return {
        "conditionId": cid,
        "question": m.get("question") or "",
        "report_bucket": pmf.classify_report_bucket(m),
        "startDate": m.get("startDate"),
        "createdAt": m.get("createdAt"),
        "endDate": m.get("endDate"),
        "resolution_time": str(pmf._resolution_timestamp(m)),
    }


def stream_slim_census(cache_dir: Path = pmf.GAMMA_CACHE_DIR) -> list[dict]:
    """Streams every cached leaf_*.json file one at a time, slimming each
    market immediately and discarding the raw (heavy) list before moving to
    the next file -- peak memory is one leaf file's worth of full dicts plus
    the ever-growing slim list, not the full census twice over."""
    leaf_files = sorted(cache_dir.glob("leaf_*.json"))
    print(f"[unbiased-pop] streaming {len(leaf_files)} cached leaf files (no new network calls for this step)...")
    slim: list[dict] = []
    for i, path in enumerate(leaf_files):
        raw = json.loads(path.read_text())
        for m in raw:
            s = _slim_market(m)
            if s is not None:
                slim.append(s)
        del raw
        if (i + 1) % 50 == 0:
            print(f"  [unbiased-pop] {i + 1}/{len(leaf_files)} leaf files processed, {len(slim)} slim records so far", flush=True)
    return slim


def stratified_sample_slim(slim_census: list[dict], n_target: int, seed: int) -> list[dict]:
    """Same algorithm as polymarket_final_pct.stratified_sample_markets
    (proportional allocation by resolution-quarter x report-category, with
    per-group seeding + shuffle-then-slice for subset-stability -- see that
    function's own docstring for why both of those matter), adapted to slim
    records that already carry a precomputed report_bucket instead of the
    full `events` field the original function recomputes it from."""
    df = pd.DataFrame(slim_census)
    df["end_ts"] = df["endDate"].apply(pmf._to_epoch_s)
    df["start_ts"] = df.apply(lambda r: pmf._to_epoch_s(r.get("startDate") or r.get("createdAt")), axis=1)
    df = df.dropna(subset=["end_ts", "start_ts"])
    df = df[df["start_ts"] >= pmf._to_epoch_s(pmf.CLOB_LAUNCH_CUTOFF)]
    df["quarter"] = pd.to_datetime(df["end_ts"], unit="s").dt.to_period("Q").astype(str)

    frac = min(1.0, n_target / max(len(df), 1))
    parts = []
    for (quarter, bucket), grp in df.groupby(["quarter", "report_bucket"]):
        k = max(1, round(len(grp) * frac)) if len(grp) else 0
        k = min(k, len(grp))
        if k <= 0:
            continue
        group_seed = int.from_bytes(hashlib.sha256(f"{seed}|{quarter}|{bucket}".encode()).digest()[:8], "big")
        rng = np.random.default_rng(group_seed)
        shuffled = rng.permutation(grp.index.to_numpy())
        idx = shuffled[:k]
        parts.append(grp.loc[idx])
    sampled = pd.concat(parts) if parts else df.iloc[0:0]
    return sampled.to_dict("records")


def _fetch_one(cid: str):
    return cid, len(pmf.fetch_market_trades(cid))


def main():
    slim_census = stream_slim_census()
    print(f"[unbiased-pop] {len(slim_census)} cleanly-resolved markets in the slim census "
          f"(uncurated -- no probability/extremity filter)")

    sample = stratified_sample_slim(slim_census, N_TARGET, seed=SEED)
    print(f"[unbiased-pop] stratified sample: {len(sample)} markets "
          f"(quarter x report_bucket, proportional -- independent of the Final-1% strategy's own selection)")

    meta_by_cid = {
        r["conditionId"]: {
            "condition_id": r["conditionId"], "question": r["question"],
            "report_bucket": r["report_bucket"], "resolution_time": r["resolution_time"],
        }
        for r in sample
    }

    n_already_cached = sum(1 for cid in meta_by_cid if pmf._trades_cache_path(cid).exists())
    print(f"[unbiased-pop] {n_already_cached} already have a cached trade tape "
          f"(from prior runs / overlap with the Final-1% population); fetching the rest with {FETCH_WORKERS} workers...")

    n_trades_by_cid = {}
    n_done = 0
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        futures = {pool.submit(_fetch_one, cid): cid for cid in meta_by_cid}
        for fut in as_completed(futures):
            cid = futures[fut]
            try:
                _, n_trades = fut.result()
                n_trades_by_cid[cid] = n_trades
            except Exception as exc:
                print(f"  [unbiased-pop] WARNING: fetch failed for {cid}: {exc}")
                n_trades_by_cid[cid] = 0
            n_done += 1
            if n_done % 100 == 0:
                print(f"  [unbiased-pop] {n_done}/{len(meta_by_cid)} markets processed ...", flush=True)

    rows = []
    for cid, meta in meta_by_cid.items():
        rows.append({**meta, "n_trades": n_trades_by_cid.get(cid, 0)})
    rows.sort(key=lambda r: r["resolution_time"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_with_trades = sum(1 for r in rows if r["n_trades"] > 0)
    print(f"\n[unbiased-pop] wrote {len(rows)} markets to {OUT_PATH} "
          f"({n_with_trades} with at least one real trade)")
    print(f"[unbiased-pop] report_bucket mix: "
          + ", ".join(f"{b}={sum(1 for r in rows if r['report_bucket'] == b)}" for b in pmf.REPORT_BUCKETS))


if __name__ == "__main__":
    main()
