"""Builds the entry dataset for the politics-only, price-calibrated Kelly
backtest (scripts/run_politics_kelly_backtest.py): one row per resolved
Polymarket politics market, with an entry price and the resolved outcome.

Population: every resolved market classify_report_bucket calls "politics"
(the same keyword classifier used throughout this codebase, applied
client-side), from the full resolved-market census
(scripts/_fetch_full_census_to_disk.py), from the same CLOB-era cutoff
(2022-09-01) used everywhere else, with a valid resolved_outcome_index. No
probability/extremity filter of any kind -- unlike the Final-1% strategy's
own population, this one deliberately spans the WHOLE 0-100% range, since
the strategy under test bets across the full price spectrum, not just the
near-100% tail.

An earlier version of this script filtered server-side via Gamma's
`tag_slug=politics` query param. Confirmed live, and the hard way: that
filter stops restricting results once pagination passes a low offset
within a date bucket (a request for tag_slug=politics at offset>=2000
returns MLB and IPL cricket markets) -- not safe for a bulk crawl, so
population is now built by streaming the already-cached full census
leaf files and classifying client-side instead, same memory-safe pattern
build_mm_unbiased_population.py already established for exactly this
"844k-market census, don't materialize it all at once" problem.

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

FETCH_WORKERS = 24


def _slim_politics_meta(m: dict) -> dict | None:
    """Reduces one full Gamma market dict to what this script needs, IF it's
    a cleanly-resolved politics market -- called while the full dict (with
    its `events`/`outcomes`/`outcomePrices` fields) is still in hand, same
    discipline as build_mm_unbiased_population.py's own _slim_market. None
    for anything not politics, not cleanly resolved, or missing a usable
    field -- mirrors resolve_market_meta's old defensive checks."""
    if pmf.classify_report_bucket(m) != "politics":
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
    outcomes = pmf._safe_json_list(m.get("outcomes"))
    if idx >= len(outcomes):
        return None
    return {
        "condition_id": cid,
        "question": m.get("question", ""),
        "resolution_time": res_ts.isoformat(),
        "resolved_outcome": outcomes[idx],
    }


def stream_politics_population(cache_dir: Path = pmf.GAMMA_CACHE_DIR) -> list[dict]:
    """Streams every cached leaf_*.json file one at a time, slimming and
    classifying each market immediately and discarding the raw (heavy) list
    before moving to the next file -- same memory-safe pattern as
    build_mm_unbiased_population.py's stream_slim_census, applied here with
    a politics filter instead of keeping every category."""
    leaf_files = sorted(cache_dir.glob("leaf_*.json"))
    print(f"[politics-kelly] streaming {len(leaf_files)} cached census leaf files "
          f"(no new network calls for this step)...")
    metas: dict[str, dict] = {}  # condition_id -> meta, de-duplicating across leaves as we go
    for i, path in enumerate(leaf_files):
        raw = json.loads(path.read_text())
        for m in raw:
            meta = _slim_politics_meta(m)
            if meta is not None:
                metas[meta["condition_id"]] = meta
        del raw
        if (i + 1) % 50 == 0:
            print(f"  [politics-kelly] {i + 1}/{len(leaf_files)} leaf files processed, "
                  f"{len(metas)} politics markets so far", flush=True)
    return list(metas.values())


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
    metas = stream_politics_population()
    print(f"[politics-kelly] {len(metas)} cleanly-resolved politics markets "
          f"(post-CLOB-cutoff, classify_report_bucket=='politics', deduplicated)")

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
