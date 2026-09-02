"""One-off utility: warms the on-disk trade-tape cache (data/raw/polymarket/trades/)
for every market in the unbiased MM population (mm_unbiased_population.csv), in
parallel. Not part of the research pipeline itself -- every downstream script
(run_mm_walkforward_validation.py, run_mm_proxy_advanced.py, run_mm_proxy_v3.py)
already calls the same cached fetch_market_trades and works fine without this,
this just avoids doing ~1,300 sequential network round-trips inside a single-
threaded research script.
"""
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import polymarket_final_pct as pmf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_mm_walkforward_validation as wf

WORKERS = 16


def main():
    meta = wf.load_population_meta()
    cids = list(meta.keys())
    print(f"[prefetch] {len(cids)} markets to warm")
    done = 0
    errors = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(pmf.fetch_market_trades, cid): cid for cid in cids}
        for fut in as_completed(futures):
            cid = futures[fut]
            try:
                fut.result()
            except Exception as exc:
                errors += 1
                print(f"  [prefetch] FAILED {cid}: {exc}", flush=True)
            done += 1
            if done % 100 == 0:
                print(f"  [prefetch] {done}/{len(cids)} done ({errors} errors)", flush=True)
    print(f"[prefetch] complete: {done} processed, {errors} errors")


if __name__ == "__main__":
    main()
