"""Backfills real L2 order-book history for recently-resolved Polymarket
crypto Up/Down markets from PolyOrderbooks' live API (src/polyorderbooks_client.py),
into a PERMANENT, git-committed cache -- unlike every other external cache in
this repo, this one is not re-fetchable later: the free tier's retention is
7 days (confirmed live via /v1/usage), so anything not captured within that
window is gone for good. That is also why this cache is NOT gitignored (see
.gitignore's comment on data/raw/polyorderbooks_l2_live/).

Scoped to 5-minute and 15-minute BTC/ETH/SOL/XRP/BNB/DOGE/HYPE/ZEC Up/Down
markets: cheap per-market (a handful of paginated /books calls each, given
the 60 req/min free-tier ceiling) and high-frequency (many distinct resolved
markets available at any time), unlike 4-hour contracts, whose genuinely
several-hour-long active window would burn the request budget on far fewer
markets. Not attempted here -- a natural follow-up with a longer time budget
or a paid tier's higher rate limit.

Real trading activity is concentrated in a window close to each market's
end_date, empirically confirmed live (a market with end_date 12:15:00 showed
real book activity from ~12:06 to ~12:16) -- NOT from its `start_date`, which
just reflects when the listing was created and can be a full day earlier.
Fetch windows below are sized from that observation, not assumed.
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import polyorderbooks_client as poc

REPO = Path(__file__).resolve().parents[1]
CACHE_DIR = poc.BOOKS_CACHE_DIR
MANIFEST_PATH = CACHE_DIR / "_manifest.json"

CONTRACT_LENGTHS = {
    "5m": {"search": "updown-5m", "pre_seconds": 1200, "post_seconds": 180},
    "15m": {"search": "updown-15m", "pre_seconds": 2700, "post_seconds": 180},
}
COINS = ["btc", "eth", "sol", "xrp", "bnb", "doge", "hype", "zec"]


def _epoch(iso_ts: str) -> int:
    return int(datetime.fromisoformat(iso_ts.replace("Z", "+00:00")).timestamp())


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, default=str))


def main(max_markets_per_contract_length: int = 150, time_budget_seconds: float = 900):
    manifest = load_manifest()
    usage = poc.get_usage()
    print(f"[backfill] plan={usage['plan']} max_history_days={usage['limits']['max_history_days']} "
          f"requests_remaining={usage['limits']['requests_remaining']}")

    start_time = time.monotonic()
    n_fetched = 0
    n_skipped_cached = 0
    n_skipped_unresolved = 0
    n_errors = 0

    for coin in COINS:
        for length, cfg in CONTRACT_LENGTHS.items():
            if time.monotonic() - start_time > time_budget_seconds:
                print(f"[backfill] time budget ({time_budget_seconds}s) reached, stopping")
                _print_summary(n_fetched, n_skipped_cached, n_skipped_unresolved, n_errors)
                save_manifest(manifest)
                return
            search = f"{coin}-{cfg['search']}"
            try:
                markets = list(poc.iter_markets(search=search, include_closed=True,
                                                 max_markets=max_markets_per_contract_length))
            except Exception as exc:
                print(f"  [backfill] FAILED listing {search}: {exc}")
                n_errors += 1
                continue
            print(f"[backfill] {search}: {len(markets)} markets found")
            for m in markets:
                slug = m["slug"]
                if slug in manifest:
                    n_skipped_cached += 1
                    continue
                if not m.get("is_resolved"):
                    n_skipped_unresolved += 1
                    continue
                if time.monotonic() - start_time > time_budget_seconds:
                    print(f"[backfill] time budget reached mid-contract-length, stopping")
                    _print_summary(n_fetched, n_skipped_cached, n_skipped_unresolved, n_errors)
                    save_manifest(manifest)
                    return
                end_ts = _epoch(m.get("closed_at") or m["end_date"])
                start_ts = end_ts - cfg["pre_seconds"]
                fetch_end_ts = end_ts + cfg["post_seconds"]
                try:
                    path = poc.fetch_and_cache_market_books(slug, start_ts, fetch_end_ts)
                    n_rows = sum(len(v) for v in json.loads(path.read_text()).get("data", {}).values())
                    manifest[slug] = {
                        "contract_length": length, "coin": coin, "end_date": m["end_date"],
                        "winner": m.get("winner"), "volume": m.get("volume"), "n_rows": n_rows,
                        "cache_file": path.name,
                    }
                    n_fetched += 1
                    if n_fetched % 10 == 0:
                        print(f"  [backfill] {n_fetched} markets fetched so far "
                              f"({time.monotonic() - start_time:.0f}s elapsed) ...", flush=True)
                        save_manifest(manifest)
                except Exception as exc:
                    print(f"  [backfill] FAILED {slug}: {exc}")
                    n_errors += 1

    save_manifest(manifest)
    _print_summary(n_fetched, n_skipped_cached, n_skipped_unresolved, n_errors)


def _print_summary(n_fetched, n_skipped_cached, n_skipped_unresolved, n_errors):
    print(f"\n[backfill] done: {n_fetched} newly fetched, {n_skipped_cached} already cached, "
          f"{n_skipped_unresolved} not yet resolved (skipped), {n_errors} errors")
    print(f"[backfill] manifest saved to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
