"""One-off utility: populates pmf.GAMMA_CACHE_DIR's leaf_*.json cache files
WITHOUT ever holding the full census in memory at once -- unlike calling
fetch_resolved_markets_census() directly, which accumulates every fetched
market into one `all_markets` list before returning (documented memory risk
in build_mm_unbiased_population.py: ~9GB+ resident for the full ~844k-market
census on a 15GB box). Each leaf is fetched, written to disk by
_fetch_leaf_bucket itself, and its return value is discarded here rather
than collected -- downstream consumers (build_mm_unbiased_population.py's
own stream_slim_census, or the politics-population equivalent) read the
cached leaf files back one at a time instead.

No tag filter: confirmed live that Gamma's tag_slug param stops filtering
once pagination passes a low offset within a bucket (returns MLB/cricket
markets at offset>=2000 even with tag_slug=politics) -- not reliable for a
bulk crawl, so this fetches the same complete, uncurated census every other
census-dependent script in this repo already relies on, and category
filtering happens client-side via classify_report_bucket instead.
"""
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import polymarket_final_pct as pmf


def main():
    date_min = pmf.CLOB_LAUNCH_CUTOFF
    date_max = pmf.pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
    leaves = pmf._plan_leaf_buckets(date_min, date_max, pmf.GAMMA_CACHE_DIR)
    print(f"[census-to-disk] {len(leaves)} leaf buckets planned")

    n_done = 0
    n_markets = 0
    with ThreadPoolExecutor(max_workers=pmf._CENSUS_WORKERS) as pool:
        futures = {pool.submit(pmf._fetch_leaf_bucket, lo, hi, pmf.GAMMA_CACHE_DIR): (lo, hi) for lo, hi in leaves}
        for fut in as_completed(futures):
            try:
                n_markets += len(fut.result())  # length only -- never binds the batch itself to a name
            except Exception as exc:
                lo, hi = futures[fut]
                print(f"  [census-to-disk] FAILED {lo}..{hi}: {exc}")
            n_done += 1
            if n_done % 25 == 0:
                print(f"  [census-to-disk] {n_done}/{len(leaves)} leaves done, {n_markets} markets seen so far", flush=True)

    print(f"[census-to-disk] done: {n_done} leaves, {n_markets} markets total (all written to {pmf.GAMMA_CACHE_DIR})")


if __name__ == "__main__":
    main()
