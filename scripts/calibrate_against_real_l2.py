"""Real L2 order-book calibration check, using the one genuinely free,
no-account, independently-verified historical L2 source found for Polymarket:
PolyOrderbooks' open Zenodo archive (CC BY 4.0, DOI 10.5281/zenodo.22084114,
https://doi.org/10.5281/zenodo.22084114). Six other providers were checked
(PolyOrderbooks' own live API, PMData, PolyHistorical, PolymarketData.co,
Telonex, Probalytics, PolyData) -- all are real, live services, but every one
of them gates even its "free" tier behind an API key / account signup, which
was not created here without asking first.

SCOPE, stated plainly because it is the single most important caveat: this
dataset covers crypto Up/Down markets ONLY (BTC/ETH/SOL/XRP/BNB/DOGE/HYPE/ZEC),
three contract lengths (5m/15m/4h), captured 2026-08-21 to 2026-08-24 -- a
4-day window, ~800 markets, no politics/sports/other. It CANNOT re-run or
validate the existing 1,331-market walk-forward backtest, which spans a
completely different (and mostly non-crypto) population from 2023-2026. What
it CAN do, and all this script attempts: check whether the MM proxy model's
core assumptions -- a flat half-spread, a flat $25 per-trade notional cap, a
15-second markout/reaction window -- are in the right ballpark for AT LEAST
this slice, against real L2 book state instead of inferring everything from
trade prints. The dataset is order-book SNAPSHOTS, not executed trades: there
is no fill/execution data here, so fill_share cannot be validated this way,
only spread and depth.

Downloads (once) to data/raw/polyorderbooks_l2/ -- kept out of git, same
convention as every other external cache in this repo (see .gitignore).
"""
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO / "data" / "raw" / "polyorderbooks_l2"
RESULTS_DIR = REPO / "results" / "polymarket_final_pct"

ZENODO_RECORD = 22084977  # version DOI 10.5281/zenodo.22084115 == this record, resolved from the concept DOI 10.5281/zenodo.22084114
FILES = ["updown_5m", "updown_15m", "updown_4h"]

# The MM proxy model's own assumptions (run_mm_proxy_backtest.py), reused
# here by value rather than imported, since this script has no dependency on
# that module's trade-print machinery -- only the numbers matter for comparison.
MODEL_BASE_HALF_SPREAD = 0.01     # -> $0.02 assumed full spread
MODEL_HALF_SPREADS_GRID = [0.005, 0.01, 0.02]  # -> $0.01 / $0.02 / $0.04 full spread
MODEL_MAX_NOTIONAL_PER_TRADE = 25.0
MODEL_MARKOUT_WINDOW_SECONDS = 15


def download_if_missing() -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in FILES:
        path = CACHE_DIR / f"{name}.parquet"
        if not path.exists() or path.stat().st_size < 1024:
            # Zenodo intermittently 504s on a cold request (observed directly
            # while building this script) -- retry with backoff, same pattern
            # as _request_json in src/polymarket_final_pct.py. A failed
            # attempt is removed rather than left as a tiny, truncated
            # "cached" file that a later run would wrongly treat as complete.
            url = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{name}.parquet?download=1"
            last_err = None
            for attempt in range(5):
                try:
                    print(f"[l2-calibration] downloading {url} -> {path} (attempt {attempt + 1})")
                    urllib.request.urlretrieve(url, path)
                    if path.stat().st_size >= 1024:
                        last_err = None
                        break
                    last_err = RuntimeError(f"downloaded file suspiciously small: {path.stat().st_size} bytes")
                except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                    last_err = exc
                path.unlink(missing_ok=True)
                time.sleep(min(2 ** attempt, 20))
            if last_err is not None:
                raise RuntimeError(f"failed to download {url} after 5 attempts: {last_err}")
        paths[name] = path
    return paths


def load(paths: dict) -> dict:
    return {name: pd.read_parquet(path) for name, path in paths.items()}


def capture_cadence(df: pd.DataFrame) -> dict:
    ordered = df.sort_values(["market_slug", "token_id", "captured_at"])
    gaps = ordered.groupby(["market_slug", "token_id"])["captured_at"].diff().dt.total_seconds().dropna()
    return {"median_s": round(gaps.median(), 2), "p10_s": round(gaps.quantile(0.1), 2), "p90_s": round(gaps.quantile(0.9), 2)}


def clean_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a real, two-sided, non-crossed book -- see the README's own
    documented caveats (empty sides and crossed quotes are real market/
    collector-staleness phenomena, disclosed and flagged, not removed from
    the source data; this analysis filters them out only where a genuine
    two-sided spread/depth number is what's being measured)."""
    return df[~df.crossed & (df.bid_prices.str.len() > 0) & (df.ask_prices.str.len() > 0)]


def spread_stats(df: pd.DataFrame) -> dict:
    clean = clean_rows(df)
    spread = clean.best_ask - clean.best_bid
    return {
        "n_clean_rows": len(clean), "n_total_rows": len(df),
        "median": round(spread.median(), 4), "mean": round(spread.mean(), 4),
        "p25": round(spread.quantile(0.25), 4), "p75": round(spread.quantile(0.75), 4),
    }


def touch_depth_stats(df: pd.DataFrame) -> dict:
    clean = clean_rows(df).copy()
    bid_depth_usd = clean.bid_sizes.str[0] * clean.best_bid
    ask_depth_usd = clean.ask_sizes.str[0] * clean.best_ask
    touch = pd.concat([bid_depth_usd, ask_depth_usd])
    return {
        "median_usd": round(touch.median(), 2), "p10_usd": round(touch.quantile(0.1), 2),
        "p25_usd": round(touch.quantile(0.25), 2), "p75_usd": round(touch.quantile(0.75), 2),
        "pct_below_model_notional_cap": round((touch < MODEL_MAX_NOTIONAL_PER_TRADE).mean() * 100, 1),
    }


def empty_and_crossed_by_minute_to_close(df: pd.DataFrame) -> dict:
    d = df.copy()
    d["min_to_close"] = (d["seconds_to_close"] // 60).clip(lower=0, upper=10)
    d["empty"] = (d.bid_prices.str.len() == 0) | (d.ask_prices.str.len() == 0)
    g = d.groupby("min_to_close").agg(empty_rate=("empty", "mean"), crossed_rate=("crossed", "mean"), n=("empty", "size"))
    return {
        str(int(k)) + ("+ (rest of tape)" if k == 10 else ""): {
            "empty_rate_pct": round(v["empty_rate"] * 100, 1),
            "crossed_rate_pct": round(v["crossed_rate"] * 100, 1),
            "n_rows": int(v["n"]),
        }
        for k, v in g.iterrows()
    }


def main():
    paths = download_if_missing()
    data = load(paths)

    out = {"dataset": {"doi": "10.5281/zenodo.22084114", "record": ZENODO_RECORD,
                        "capture_window": "2026-08-21 to 2026-08-24", "coins": "BTC,ETH,SOL,XRP,BNB,DOGE,HYPE,ZEC"},
           "model_assumptions": {
               "base_half_spread": MODEL_BASE_HALF_SPREAD, "implied_base_full_spread": MODEL_BASE_HALF_SPREAD * 2,
               "half_spread_grid": MODEL_HALF_SPREADS_GRID,
               "implied_full_spread_grid": [h * 2 for h in MODEL_HALF_SPREADS_GRID],
               "max_notional_per_trade": MODEL_MAX_NOTIONAL_PER_TRADE,
               "markout_window_seconds": MODEL_MARKOUT_WINDOW_SECONDS,
           },
           "by_contract_length": {}}

    print(f"=== Real L2 calibration check against PolyOrderbooks' Zenodo archive ===")
    print(f"Model assumes: base full spread ${MODEL_BASE_HALF_SPREAD * 2}, "
          f"grid {[round(h * 2, 3) for h in MODEL_HALF_SPREADS_GRID]}, "
          f"notional cap ${MODEL_MAX_NOTIONAL_PER_TRADE}, markout window {MODEL_MARKOUT_WINDOW_SECONDS}s\n")

    for name, df in data.items():
        cadence = capture_cadence(df)
        spread = spread_stats(df)
        depth = touch_depth_stats(df)
        by_minute = empty_and_crossed_by_minute_to_close(df)
        out["by_contract_length"][name] = {
            "n_markets": int(df.market_slug.nunique()), "n_rows": len(df),
            "capture_cadence_seconds": cadence, "real_full_spread": spread,
            "real_touch_depth_usd": depth, "empty_and_crossed_by_minute_to_close": by_minute,
        }
        print(f"--- {name} ({df.market_slug.nunique()} markets, {len(df):,} rows) ---")
        print(f"  capture cadence: median {cadence['median_s']}s (p10-p90: {cadence['p10_s']}-{cadence['p90_s']}s)")
        print(f"  real full spread: median ${spread['median']}  mean ${spread['mean']}  "
              f"[p25 ${spread['p25']}, p75 ${spread['p75']}]  vs. model's assumed ${MODEL_BASE_HALF_SPREAD * 2} base / "
              f"{[round(h * 2, 3) for h in MODEL_HALF_SPREADS_GRID]} grid")
        print(f"  touch depth: median ${depth['median_usd']}  [p25 ${depth['p25_usd']}, p75 ${depth['p75_usd']}]  "
              f"-- {depth['pct_below_model_notional_cap']}% of touch quotes are BELOW the model's ${MODEL_MAX_NOTIONAL_PER_TRADE} cap")
        print(f"  minute-to-close=0: {by_minute['0']['empty_rate_pct']}% empty-sided, "
              f"{by_minute['0']['crossed_rate_pct']}% crossed (n={by_minute['0']['n_rows']:,})")
        print()

    out_path = RESULTS_DIR / "l2_calibration_check.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
