"""Builds the entry dataset for the politics-only, price-calibrated Kelly
backtest (scripts/run_politics_kelly_backtest.py): one row per resolved
Polymarket politics market, with an entry price and the resolved outcome.

Population: every resolved market Polymarket itself tags "politics"
(`tag_slug=politics`, confirmed live against Gamma's API -- see
fetch_resolved_markets_census's `extra_params`), from the same CLOB-era
cutoff (2022-09-01) used throughout this codebase, deduplicated, with a
valid resolved_outcome_index. No probability/extremity filter of any kind
-- unlike the Final-1% strategy's own population, this one deliberately
spans the WHOLE 0-100% range, since the strategy under test bets across the
full price spectrum, not just the near-100% tail.

Entry rule, stated plainly because it's a real modeling choice: the
market's own FIRST real trade (chronologically -- fetch_market_trades does
NOT return trades in timestamp order, confirmed live, so this sorts before
taking the first one), price normalized onto the YES side (yes_price =
price if outcome=="Yes" else 1-price). This is the simplest, most
literal reading of "look at what the market says," and needs no extra API
surface beyond what's already used elsewhere in this codebase. The real
tradeoff, disclosed rather than hidden: a market's very first trade can be
thin/noisy (wide effective spread, one illiquid fill) relative to a price
observed once real trading volume has built up -- flagged here, not
smoothed over.

Markets with zero real trades (never actually traded on the CLOB) are
skipped -- there is no entry price to assign them.
"""
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import polymarket_final_pct as pmf

REPO = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO / "results" / "polymarket_final_pct"
POLITICS_CENSUS_CACHE = pmf.DATA_DIR / "gamma_politics"

FETCH_WORKERS = 24


def fetch_politics_census() -> list[dict]:
    print("[politics-kelly] fetching politics-tagged resolved market census ...")
    markets = pmf.fetch_resolved_markets_census(cache_dir=POLITICS_CENSUS_CACHE, extra_params={"tag_slug": "politics"})
    print(f"[politics-kelly] {len(markets)} politics-tagged markets in the census")
    return markets


def resolve_market_meta(m: dict) -> dict | None:
    """condition_id, question, resolution_time, resolved_outcome_index for
    one market -- None if any required field can't be determined (mirrors
    the same defensive pattern stratified_sample_markets/_slim_market use
    elsewhere in this codebase)."""
    cid = m.get("conditionId")
    if not cid:
        return None
    idx = pmf.resolved_outcome_index(m)
    if idx is None:
        return None
    res_ts = pmf._resolution_timestamp(m)
    if res_ts is None:
        return None
    outcomes = pmf._safe_json_list(m.get("outcomes"))
    if idx >= len(outcomes):
        return None
    return {
        "condition_id": cid,
        "question": m.get("question", ""),
        "resolution_time": res_ts.isoformat(),
        "resolved_outcome": outcomes[idx],
    }


def fetch_entry(meta: dict) -> dict | None:
    """First real (chronologically-sorted) trade for this market, price
    normalized onto the YES side. None if the market has no usable trades."""
    raw = pmf.fetch_market_trades(meta["condition_id"])
    if not raw:
        return None
    valid = []
    for t in raw:
        try:
            price = float(t["price"])
            ts = float(t.get("timestamp", 0))
            outcome = t["outcome"]
        except (KeyError, ValueError, TypeError):
            continue
        if price <= 0 or price >= 1 or outcome not in ("Yes", "No"):
            continue
        valid.append({"price": price, "timestamp": ts, "outcome": outcome})
    if not valid:
        return None
    valid.sort(key=lambda t: t["timestamp"])
    first = valid[0]
    yes_price = first["price"] if first["outcome"] == "Yes" else 1.0 - first["price"]
    return {
        "condition_id": meta["condition_id"],
        "question": meta["question"][:120],
        "entry_time": pmf.pd.Timestamp(first["timestamp"], unit="s", tz="UTC").isoformat(),
        "yes_price": round(yes_price, 4),
        "resolution_time": meta["resolution_time"],
        "resolved_yes": meta["resolved_outcome"] == "Yes",
        "n_trades": len(valid),
    }


def main():
    census = fetch_politics_census()

    metas = []
    n_no_condition_id = n_no_outcome_idx = n_no_res_time = 0
    for m in census:
        meta = resolve_market_meta(m)
        if meta is None:
            if not m.get("conditionId"):
                n_no_condition_id += 1
            elif pmf.resolved_outcome_index(m) is None:
                n_no_outcome_idx += 1
            else:
                n_no_res_time += 1
            continue
        metas.append(meta)
    print(f"[politics-kelly] {len(metas)} markets with a resolvable outcome "
          f"(dropped: {n_no_condition_id} no condition_id, {n_no_outcome_idx} no resolved index, {n_no_res_time} no resolution time)")

    print(f"[politics-kelly] fetching trade tapes for entry prices ({FETCH_WORKERS} workers) ...")
    entries = []
    n_no_trades = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        futures = {pool.submit(fetch_entry, meta): meta for meta in metas}
        for i, fut in enumerate(as_completed(futures)):
            entry = fut.result()
            if entry is None:
                n_no_trades += 1
            else:
                entries.append(entry)
            if (i + 1) % 500 == 0:
                print(f"  [politics-kelly] {i + 1}/{len(metas)} processed ({time.time() - t0:.0f}s elapsed) ...", flush=True)

    print(f"[politics-kelly] {len(entries)} markets with a usable entry price ({n_no_trades} had zero real trades)")

    entries.sort(key=lambda r: r["entry_time"])
    out_path = RESULTS_DIR / "politics_kelly_entries.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["condition_id", "question", "entry_time", "yes_price",
                                           "resolution_time", "resolved_yes", "n_trades"])
        w.writeheader()
        w.writerows(entries)
    print(f"saved {out_path} ({len(entries)} rows)")


if __name__ == "__main__":
    main()
