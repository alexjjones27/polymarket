"""Thin client for PolyOrderbooks' live REST API (https://api.polyorderbooks.com),
the one third-party Polymarket L2 data vendor whose free tier was verified to
work end-to-end (see docs/mm_strategy_methodology.md Section 11). Unlike
Polymarket's own `/book` endpoint (404s the instant a market resolves), this
archive retains full-depth L2 order-book snapshots for resolved markets --
but only for `max_history_days` (7 on the free "starter" plan, confirmed live
via /v1/usage), so data not fetched and cached within that window is gone.

Coverage is crypto markets only (confirmed live: searches for "president",
"NFL", "election", "senate" return zero or crypto-conference-adjacent
results, never real politics/sports) -- this does not reach the rest of the
MM strategy's population, only its crypto_price / Up-Down slice.

Requires POLYORDERBOOKS_API_KEY in .env (see .env.example). The key is never
logged or printed by this module.
"""
import datetime
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

API_BASE = "https://api.polyorderbooks.com"
BOOKS_CACHE_DIR = REPO_ROOT / "data" / "raw" / "polyorderbooks_l2_live"
MAX_BOOKS_PAGE_LIMIT = 200  # server-enforced ceiling, confirmed live (limit=500 -> 400 "limit must be <= 200")

# Free "starter" plan: 60 requests/minute (confirmed via /v1/usage). No
# per-response rate-limit headers are exposed, so pacing is done client-side
# with a fixed minimum gap between requests rather than reading remaining
# quota back from each response.
MIN_SECONDS_BETWEEN_REQUESTS = 1.1  # ~54/min, a safety margin under the 60/min ceiling
_last_request_time = [0.0]


def _get(path: str, params: dict) -> dict:
    api_key = os.environ.get("POLYORDERBOOKS_API_KEY")
    if not api_key:
        raise RuntimeError("POLYORDERBOOKS_API_KEY not set in .env")
    elapsed = time.monotonic() - _last_request_time[0]
    if elapsed < MIN_SECONDS_BETWEEN_REQUESTS:
        time.sleep(MIN_SECONDS_BETWEEN_REQUESTS - elapsed)
    url = f"{API_BASE}{path}?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    req = urllib.request.Request(url, headers={"X-API-Key": api_key})
    last_err = None
    for attempt in range(5):
        try:
            _last_request_time[0] = time.monotonic()
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504):
                last_err = exc
                time.sleep(min(2 ** attempt, 20))
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_err = exc
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"GET {url} failed after 5 retries: {last_err}")


def get_usage() -> dict:
    """Plan/quota info -- never logs the key itself, just what it's entitled
    to (max_history_days is the load-bearing number for how urgently a
    market needs to be fetched before its book history ages out)."""
    return _get("/v1/usage", {})


def list_markets(search: str = None, include_closed: bool = True, event_slug: str = None,
                  sort: str = None, order: str = None, cursor: str = None, limit: int = 100) -> dict:
    return _get("/v1/markets", {
        "search": search, "include_closed": include_closed, "event_slug": event_slug,
        "sort": sort, "order": order, "cursor": cursor, "limit": limit,
    })


def iter_markets(search: str = None, include_closed: bool = True, max_markets: int = None):
    """Pages through list_markets via cursor until exhausted or max_markets hit."""
    cursor = None
    n = 0
    while True:
        page = list_markets(search=search, include_closed=include_closed, cursor=cursor)
        for m in page.get("data", []):
            yield m
            n += 1
            if max_markets is not None and n >= max_markets:
                return
        cursor = page.get("metadata", {}).get("next_cursor")
        if not cursor or not page.get("data"):
            return


def get_market(id_or_slug: str) -> dict:
    return _get(f"/v1/markets/{id_or_slug}", {})


def reduce_to_touch(snapshot: dict) -> list:
    """A raw {"t":..., "bids": [[price,size],...], "asks": [[price,size],...]}
    snapshot reduced to a compact 5-element array
    [epoch_seconds, best_bid, best_bid_size, best_ask, best_ask_size]
    (None for a missing side). Discovered the hard way, twice: a full
    40+-level ladder per snapshot, at 1s resolution across a market's whole
    life, runs 7-8MB PER MARKET (measured directly, mid-backfill, before any
    reduction existed) -- committing that to git across even a modest sample
    would be enormous. A first fix (dict-keyed touch-only snapshots, ISO
    timestamp strings) cut that to ~650KB/market -- better, but repeating
    5 JSON key names and a ~20-char ISO string across thousands of snapshots
    is still mostly overhead for data l2_replay_backtest.py reads
    positionally anyway. This compact array form (epoch int, no repeated
    keys) cuts it further, to roughly a third of that -- with zero loss of
    anything this project's models use."""
    bids, asks = snapshot.get("bids") or [], snapshot.get("asks") or []
    ts = int(datetime.datetime.fromisoformat(snapshot["t"].replace("Z", "+00:00")).timestamp())
    return [
        ts,
        bids[0][0] if bids else None, bids[0][1] if bids else None,
        asks[0][0] if asks else None, asks[0][1] if asks else None,
    ]


def fetch_market_books(id_or_slug: str, start_ts: int, end_ts: int, resolution: str = "1s") -> dict:
    """Full, paginated, touch-reduced (see reduce_to_touch) L2 book history
    for one market across its whole [start_ts, end_ts] life -- both outcome
    tokens are returned together in one logical fetch (the API already
    batches them per request). Returns {"market_id":..., "tokens": {label:
    token_id}, "data": {label: [touch snapshots]}} with every page's
    snapshots concatenated and de-duplicated by timestamp (cursor pagination
    can repeat the boundary row across pages)."""
    merged = {"market_id": None, "tokens": None, "data": {}}
    cursor = None
    seen_ts = {}
    while True:
        page = _get(f"/v1/markets/{id_or_slug}/books", {
            "start_ts": start_ts, "end_ts": end_ts, "resolution": resolution,
            "limit": MAX_BOOKS_PAGE_LIMIT, "cursor": cursor,
        })
        merged["market_id"] = page.get("market_id", merged["market_id"])
        merged["tokens"] = page.get("tokens", merged["tokens"])
        for label, snaps in page.get("data", {}).items():
            bucket = merged["data"].setdefault(label, [])
            ts_set = seen_ts.setdefault(label, set())
            for s in snaps:
                if s["t"] not in ts_set:
                    ts_set.add(s["t"])
                    bucket.append(reduce_to_touch(s))
        cursor = page.get("metadata", {}).get("next_cursor")
        got_full_page = any(len(v) >= MAX_BOOKS_PAGE_LIMIT for v in page.get("data", {}).values())
        if not cursor or not got_full_page:
            break
    return merged


def _cache_path(slug: str) -> Path:
    BOOKS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = slug.replace("/", "_")
    return BOOKS_CACHE_DIR / f"{safe}.json"


def fetch_and_cache_market_books(slug: str, start_ts: int, end_ts: int, resolution: str = "1s") -> Path:
    """Same as fetch_market_books, but writes to (and short-circuits from)
    a permanent on-disk cache -- unlike every other cache in this repo, this
    one is NOT re-fetchable later: PolyOrderbooks' free-tier retention is 7
    days, so a market's book history that isn't captured within that window
    is gone for good. Returns the cache file path without re-fetching if it
    already exists."""
    path = _cache_path(slug)
    if path.exists():
        return path
    books = fetch_market_books(slug, start_ts, end_ts, resolution)
    path.write_text(json.dumps(books))
    return path
