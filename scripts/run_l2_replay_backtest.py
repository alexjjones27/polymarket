"""Runs l2_market_pnl (l2_replay_backtest.py) over every market backfilled by
scripts/backfill_polyorderbooks_l2.py -- the first MM backtest in this
project built on real order-book state instead of trade-print inference.
Reports the same best-case/markout/gap shape as every prior MM script here,
for direct comparison against the trade-print model's own numbers on the
SAME market family (crypto Up/Down, since that's this data's only coverage).

Both outcome tokens ("Yes"/"No") of each market are backtested and summed --
they are two independent, separately-quoted order books (confirmed live:
each token has its own price ladder), not a single symmetric one.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from l2_replay_backtest import l2_market_pnl, load_touch_series

REPO = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO / "data" / "raw" / "polyorderbooks_l2_live"
MANIFEST_PATH = CACHE_DIR / "_manifest.json"
RESULTS_DIR = REPO / "results" / "polymarket_final_pct"

HALF_SPREADS = [0.005, 0.01, 0.02]
FILL_SHARES = [0.05, 0.15, 0.30]
BASE_HALF_SPREAD = 0.01
BASE_FILL_SHARE = 0.15


def main():
    if not MANIFEST_PATH.exists():
        print(f"No manifest at {MANIFEST_PATH} -- run scripts/backfill_polyorderbooks_l2.py first.")
        return
    manifest = json.loads(MANIFEST_PATH.read_text())
    print(f"[l2-replay] {len(manifest)} markets in the backfilled manifest")

    per_market = []
    n_skipped_no_data = 0
    for slug, meta in manifest.items():
        cache_file = CACHE_DIR / meta["cache_file"]
        if not cache_file.exists():
            n_skipped_no_data += 1
            continue
        raw = json.loads(cache_file.read_text())
        token_labels = list(raw.get("data", {}).keys())
        if not token_labels:
            n_skipped_no_data += 1
            continue
        market_result = {"slug": slug, "contract_length": meta["contract_length"], "coin": meta["coin"],
                          "pnl_best_case": 0.0, "pnl_with_markout": 0.0, "n_captured": 0, "n_quotable_ticks": 0}
        for label in token_labels:
            touches = load_touch_series(cache_file, label)
            r = l2_market_pnl(touches, BASE_HALF_SPREAD, BASE_FILL_SHARE)
            market_result["pnl_best_case"] += r["pnl_best_case"]
            market_result["pnl_with_markout"] += r["pnl_with_markout"]
            market_result["n_captured"] += r["n_captured"]
            market_result["n_quotable_ticks"] += r["n_quotable_ticks"]
        per_market.append(market_result)

    print(f"[l2-replay] {len(per_market)} markets with cached data ({n_skipped_no_data} skipped, no data)")

    by_length = {}
    for r in per_market:
        b = by_length.setdefault(r["contract_length"], {"n_markets": 0, "best_case": 0.0, "markout": 0.0, "n_captured": 0})
        b["n_markets"] += 1
        b["best_case"] += r["pnl_best_case"]
        b["markout"] += r["pnl_with_markout"]
        b["n_captured"] += r["n_captured"]

    total_best = sum(r["pnl_best_case"] for r in per_market)
    total_markout = sum(r["pnl_with_markout"] for r in per_market)
    total_captured = sum(r["n_captured"] for r in per_market)
    gap = (1 - total_markout / total_best) * 100 if total_best else None

    print(f"\n=== Real order-book replay backtest, base config (half_spread=${BASE_HALF_SPREAD}, "
          f"fill_share={BASE_FILL_SHARE:.0%}) ===")
    print(f"{len(per_market)} markets, {total_captured} fills captured")
    print(f"best_case=${total_best:,.2f}  markout=${total_markout:,.2f}  "
          f"gap={f'{gap:.1f}%' if gap is not None else 'n/a'}")
    print("\nBy contract length:")
    for length, b in by_length.items():
        g = (1 - b["markout"] / b["best_case"]) * 100 if b["best_case"] else None
        print(f"  {length:<6} n_markets={b['n_markets']:<4} n_captured={b['n_captured']:<6} "
              f"best_case=${b['best_case']:>10,.2f}  markout=${b['markout']:>10,.2f}  "
              f"gap={f'{g:.1f}%' if g is not None else 'n/a'}")

    print("\nSensitivity grid (total markout PnL, $):")
    sensitivity = []
    for hs in HALF_SPREADS:
        for fs in FILL_SHARES:
            total = 0.0
            n_cap = 0
            for slug, meta in manifest.items():
                cache_file = CACHE_DIR / meta["cache_file"]
                if not cache_file.exists():
                    continue
                raw = json.loads(cache_file.read_text())
                for label in raw.get("data", {}).keys():
                    touches = load_touch_series(cache_file, label)
                    r = l2_market_pnl(touches, hs, fs)
                    total += r["pnl_with_markout"]
                    n_cap += r["n_captured"]
            sensitivity.append({"half_spread": hs, "fill_share": fs, "total_markout": round(total, 2), "n_captured": n_cap})
            print(f"  half_spread=${hs:<6} fill_share={fs:<6.0%}  total_markout=${total:>10,.2f}  n_captured={n_cap}")

    out = {
        "n_markets": len(per_market), "n_skipped_no_data": n_skipped_no_data,
        "base_config": {"half_spread": BASE_HALF_SPREAD, "fill_share": BASE_FILL_SHARE},
        "total_best_case": round(total_best, 2), "total_markout": round(total_markout, 2),
        "total_captured": total_captured, "gap_pct": round(gap, 1) if gap is not None else None,
        "by_contract_length": by_length, "sensitivity_grid": sensitivity,
        "per_market": [{"slug": r["slug"], "contract_length": r["contract_length"], "coin": r["coin"],
                         "pnl_best_case": round(r["pnl_best_case"], 4), "pnl_with_markout": round(r["pnl_with_markout"], 4),
                         "n_captured": r["n_captured"]} for r in per_market],
    }
    out_path = RESULTS_DIR / "l2_replay_backtest_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
