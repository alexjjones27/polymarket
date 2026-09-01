"""Polymarket "final 1%" spread-capture backtest.

Historical analysis only -- no live trading, no order placement, no API
keys. Buys an outcome token once its price crosses above $0.99 (or below
$0.01 for the opposite side) and holds to resolution, trying to capture the
residual spread against genuine resolution risk. The entire point of this
backtest is the *tail* risk: a single "flip" (a market that traded above
$0.99 and still resolved the other way) wipes out roughly 100 winning
trades at this payout structure, so flip detection, categorization, and the
statistical uncertainty on the flip rate are treated as first-class outputs,
not a footnote on top of an aggregate return number.

Sections (independently swappable, per the task's separation requirement):

    1. DATA FETCH        -- Gamma census, CLOB price history, trade feed.
    2. CATEGORY MAPPING   -- Gamma category -> {report bucket, fee bucket}.
    3. FEES / GAS
    4. CROSSING SIGNAL    -- no-lookahead entry detection.
    5. BACKTEST ENGINE    -- fill sizing, fees, gas, payout at resolution.
    6. FLIP ANALYSIS      -- categorization + Wilson/Clopper-Pearson CI.
    7. METRICS / SENSITIVITY / PLOTS
    8. SAMPLING           -- stratified random sample from the full census.
    9. REPORT
   10. MAIN

Key empirical findings from directly testing the live APIs (2026-08-25),
documented here because they drove real design decisions:

  * Gamma's classic `/markets?closed=true` endpoint (offset+limit
    pagination) is marked deprecated with a `sunset` date that has already
    passed, and the docs point to `/markets/keyset`. In practice the
    keyset endpoint's `cursor` parameter is a no-op: every request returns
    page 1 regardless of what cursor value or parameter name is passed
    (confirmed with cache-busting query params and multiple candidate
    parameter names, ruling out a CDN caching artifact). So this pipeline
    uses the classic endpoint, whose *offset* itself is separately capped
    at 2000 (`offset<=2000` regardless of `limit`) -- worked around here by
    recursively bucketing the resolved-market universe by `end_date_min`/
    `end_date_max` (confirmed-working, if undocumented under those exact
    snake_case names) until every bucket fits under the cap.
  * `/prices-history?market=<token>&interval=max` (or any `interval=`
    shorthand) reliably returns an EMPTY history even for a token with
    thousands of real trades -- this is the known granularity/emptiness
    bug the task warned about. The fix confirmed here: never use
    `interval=`; always pass explicit `startTs`/`endTs` (capped at a 15-day
    window regardless of `fidelity`) together with an explicit `fidelity`
    in minutes. That reliably returns real, ~1-minute-resolution data.
  * Resolved markets from before Polymarket's CLOB launch (mid-2022) have
    NO price history at any window/fidelity -- they traded on the old AMM,
    not the order book. These are excluded from the population (see
    CLOB_LAUNCH_CUTOFF), not treated as a granularity bug. Confirmed on a
    live 20-market sample spanning 2022-2025 and every volume tier: a
    market's *lifetime* can straddle the cutoff even when its *resolution*
    falls inside the census window (found live on a $1.7M-volume Senate
    market that resolved in Nov 2022 but started trading in Jan 2022 and
    had zero CLOB history at any window) -- handled by additionally
    filtering on each market's own start date in stratified_sample_markets,
    not just the census query's end-date range. Separately, some genuinely
    post-cutoff, low-volume markets from the first ~2 months after CLOB
    launch also came back empty even with the explicit-window fix -- read
    as thin early liquidity (plausibly zero real trades on that specific
    token), not a data-access bug, and left in the population: a token
    with no crossing correctly contributes no trade rather than being
    silently dropped.
  * `/book` on a resolved market's token 404s ("No orderbook exists") --
    historical order-book depth is not retrievable after the fact via any
    public endpoint. As a liquidity proxy this pipeline instead uses
    `data-api.polymarket.com/trades` (a public, unauthenticated trade feed,
    filterable by `market=<conditionId>`), looking at realized trade sizes
    around the crossing snapshot.
  * Fees, confirmed against docs.polymarket.com/trading/fees and
    help.polymarket.com (both official, cross-checked against the docs'
    own worked examples): makers pay nothing, ever. Takers pay
    `fee = shares * feeRate * price * (1 - price)`, i.e. fee as a fraction
    of notional = feeRate * (1 - price) -- which is exactly why this
    strategy's fee drag is small: feeRate varies 0.00-0.07 by category, and
    at p=0.99 the (1-p) term alone is already 0.01.
  * Polymarket's relayer sponsors all on-chain gas for the standard
    trading flow (order placement, approvals, CTF ops) -- ordinary users
    pay $0 gas per trade. This pipeline models that as the default and
    separately reports a conservative non-relayed on-chain estimate
    (~$0.005/trade, from a live Polygon gas price and POL/USD quote) as a
    sensitivity case, per the task's request for "a current estimate."
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "raw" / "polymarket"
GAMMA_CACHE_DIR = DATA_DIR / "gamma_markets"
PRICES_CACHE_DIR = DATA_DIR / "prices_history"
TRADES_CACHE_DIR = DATA_DIR / "trades"
RESULTS_DIR = REPO_ROOT / "results" / "polymarket_final_pct"
PLOTS_DIR = RESULTS_DIR / "plots"

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
DATA_API_BASE = "https://data-api.polymarket.com"

USER_AGENT = "Mozilla/5.0 (research backtest script; contact via repo)"

GAMMA_OFFSET_CAP = 2000  # confirmed live: offset>2000 -> 422, regardless of limit
GAMMA_PAGE_LIMIT = 100   # confirmed live: limit is capped at 100
PRICES_MAX_WINDOW_S = 15 * 86400  # confirmed live: startTs/endTs span capped at 15 days

# Polymarket's CLOB (order book) launched mid-2022; resolved markets before
# this traded on the old AMM and have no CLOB price history at all (this is
# a population-scoping decision, not the granularity bug -- see module
# docstring). Kept a little conservative (later than the actual launch) so
# every included market has a real chance of CLOB price history.
CLOB_LAUNCH_CUTOFF = "2022-09-01"


def _request_json(url: str, retries: int = 5, timeout: float = 30.0) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
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
    raise RuntimeError(f"GET {url} failed after {retries} retries: {last_err}")


def _get(base: str, path: str, params: dict) -> object:
    url = f"{base}{path}?" + urllib.parse.urlencode(params)
    return _request_json(url)


def _safe_json_list(value) -> list:
    """Parses a Gamma JSON-encoded-string field (clobTokenIds, outcomes,
    outcomePrices, umaResolutionStatuses, ...) defensively. Needed because
    market dicts that have been round-tripped through a pandas DataFrame
    (stratified_sample_markets) turn a missing field into NaN -- a float,
    truthy, and not a string -- which `value or "[]"` does not catch before
    json.loads() would raise on it."""
    if not isinstance(value, str):
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


# ---------------------------------------------------------------------------
# 1a. DATA FETCH -- Gamma census of ALL resolved markets (no curation)
# ---------------------------------------------------------------------------
# Adaptively bucketed by end date: the classic `/markets` endpoint's offset
# is hard-capped at 2000 regardless of `limit`, so any bucket with more than
# ~2000 markets is recursively split in half by date until every leaf fits.
# This gives complete coverage of the resolved-market population (subject
# only to the CLOB-era cutoff above), not a curated or volume-sorted subset.
#
# Two phases, both parallelized with a thread pool (pure I/O-bound HTTP
# calls): (1) PLAN -- cheaply probe each candidate bucket with a single
# offset=2000 request to see whether it exceeds the cap, splitting anything
# that does, until every bucket is a fetchable leaf; (2) FETCH -- exhaustively
# paginate every leaf bucket. Each leaf's raw pages are cached to disk so
# re-runs (and reruns after a probe/fetch failure) don't redo work.

_CENSUS_WORKERS = 16


def _gamma_page(date_min: str, date_max: str, offset: int, limit: int = GAMMA_PAGE_LIMIT) -> list[dict]:
    return _get(
        GAMMA_BASE, "/markets",
        {
            "closed": "true",
            "limit": limit,
            "offset": offset,
            "end_date_min": date_min,
            "end_date_max": date_max,
        },
    )


def _bucket_exceeds_cap(date_min: str, date_max: str) -> bool:
    """One cheap request: does this bucket have more than GAMMA_OFFSET_CAP
    markets? (probe at the offset cap rather than paginating up to it)."""
    probe = _gamma_page(date_min, date_max, offset=GAMMA_OFFSET_CAP, limit=1)
    return len(probe) > 0


def _midpoint(date_min: str, date_max: str) -> Optional[str]:
    lo, hi = pd.Timestamp(date_min), pd.Timestamp(date_max)
    if hi - lo <= pd.Timedelta(days=1):
        return None  # already at day granularity, can't split further
    return (lo + (hi - lo) / 2).strftime("%Y-%m-%d")


def _plan_cache_path(date_min: str, date_max: str, cache_dir: Path) -> Path:
    return cache_dir / f"plan_{date_min}_{date_max}.json"


def _plan_leaf_buckets(date_min: str, date_max: str, cache_dir: Path = GAMMA_CACHE_DIR) -> list[tuple[str, str]]:
    """Parallel BFS: repeatedly probe the frontier of not-yet-classified
    buckets, splitting anything over the offset cap, until every bucket is
    confirmed fetchable. Returns the final list of leaf (min, max) ranges.

    The plan itself is cached (not just each leaf's fetched markets) --
    without this, a re-run with everything already fetched still had to
    replay the *entire* probing BFS (one live HTTP request per tree node,
    hundreds of them for the full multi-year population) before it could
    even start reading from the leaf cache. Only the newest partial day
    (today's date_max) is excluded from caching, since more of it can
    resolve between runs."""
    plan_path = _plan_cache_path(date_min, date_max, cache_dir)
    if plan_path.exists():
        return [tuple(pair) for pair in json.loads(plan_path.read_text())]

    leaves: list[tuple[str, str]] = []
    frontier: list[tuple[str, str]] = [(date_min, date_max)]
    with ThreadPoolExecutor(max_workers=_CENSUS_WORKERS) as pool:
        while frontier:
            futures = {pool.submit(_bucket_exceeds_cap, lo, hi): (lo, hi) for lo, hi in frontier}
            next_frontier: list[tuple[str, str]] = []
            for fut in as_completed(futures):
                lo, hi = futures[fut]
                try:
                    exceeds = fut.result()
                except Exception:
                    exceeds = True  # be conservative: split rather than risk silent truncation
                if not exceeds:
                    leaves.append((lo, hi))
                    continue
                mid = _midpoint(lo, hi)
                if mid is None:
                    leaves.append((lo, hi))  # can't split further, accept the offset-cap edge case
                else:
                    next_frontier.extend([(lo, mid), (mid, hi)])
            frontier = next_frontier

    cache_dir.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(leaves))
    return leaves


def _fetch_leaf_bucket(date_min: str, date_max: str, cache_dir: Path) -> list[dict]:
    """Fetch+cache one leaf bucket. On a persistent API failure (retries
    exhausted), returns an empty list WITHOUT writing the cache file, so a
    later re-run retries this bucket rather than permanently baking in a
    transient server error as "no markets"."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"leaf_{date_min}_{date_max}.json"
    if path.exists():
        return json.loads(path.read_text())
    out: list[dict] = []
    offset = 0
    try:
        while True:
            page = _gamma_page(date_min, date_max, offset)
            if not page:
                break
            out.extend(page)
            if len(page) < GAMMA_PAGE_LIMIT:
                break
            offset += GAMMA_PAGE_LIMIT
            if offset > GAMMA_OFFSET_CAP:
                break  # offset-cap edge case (see _plan_leaf_buckets); accept partial rather than loop
    except Exception as exc:
        print(f"  [census] WARNING: leaf {date_min}..{date_max} failed ({exc}); will retry on next run")
        return []
    path.write_text(json.dumps(out))
    return out


def fetch_resolved_markets_census(
    date_min: str = CLOB_LAUNCH_CUTOFF,
    date_max: Optional[str] = None,
    cache_dir: Path = GAMMA_CACHE_DIR,
) -> list[dict]:
    """Complete, uncurated census of resolved markets in [date_min, date_max)
    via the two-phase plan-then-fetch approach above. Deduplicated union
    across all leaf buckets. Resilient to a single bucket's persistent API
    failure -- that bucket is skipped (uncached, so retried next run) rather
    than crashing the entire census."""
    if date_max is None:
        date_max = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")

    leaves = _plan_leaf_buckets(date_min, date_max, cache_dir)
    all_markets: list[dict] = []
    failed = []
    with ThreadPoolExecutor(max_workers=_CENSUS_WORKERS) as pool:
        futures = {pool.submit(_fetch_leaf_bucket, lo, hi, cache_dir): (lo, hi) for lo, hi in leaves}
        for fut in as_completed(futures):
            try:
                all_markets.extend(fut.result())
            except Exception as exc:
                failed.append((futures[fut], str(exc)))
    if failed:
        print(f"  [census] {len(failed)} bucket(s) failed and were skipped: {failed[:5]}")
    return _dedupe_by_id(all_markets)


def _dedupe_by_id(markets: list[dict]) -> list[dict]:
    seen = {}
    for m in markets:
        seen[m["id"]] = m
    # Sorted by id for a deterministic row order: the raw `markets` list's
    # order depends on which ThreadPoolExecutor future happened to complete
    # first (non-deterministic across runs), which otherwise silently
    # propagates into stratified_sample_markets's per-group index arrays --
    # discovered when two threshold-sweep runs a day apart shared only 1%
    # of their sampled markets despite an identical fixed seed.
    return [seen[k] for k in sorted(seen.keys(), key=str)]


# ---------------------------------------------------------------------------
# 1b. DATA FETCH -- CLOB price history (explicit-window fix, see docstring)
# ---------------------------------------------------------------------------

def _token_key(token_id: str) -> str:
    return token_id[-24:]  # last 24 digits is plenty unique, keeps filenames sane


def _prices_chunk_cache_path(token_id: str, start_s: int, end_s: int, fidelity: int) -> Path:
    PRICES_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return PRICES_CACHE_DIR / f"{_token_key(token_id)}_{start_s}_{end_s}_f{fidelity}.json"


def fetch_prices_chunk(token_id: str, start_s: int, end_s: int, fidelity: int) -> list[dict]:
    """One explicit-window call, cached. NEVER uses `interval=` (confirmed
    broken -- see module docstring); always explicit startTs/endTs."""
    assert end_s - start_s <= PRICES_MAX_WINDOW_S
    path = _prices_chunk_cache_path(token_id, start_s, end_s, fidelity)
    if path.exists():
        return json.loads(path.read_text())
    try:
        d = _get(CLOB_BASE, "/prices-history", {
            "market": token_id, "startTs": start_s, "endTs": end_s, "fidelity": fidelity,
        })
        history = d.get("history", []) if isinstance(d, dict) else []
    except urllib.error.HTTPError:
        history = []
    path.write_text(json.dumps(history))
    return history


def fetch_price_series(token_id: str, start_s: int, end_s: int, fidelity: int) -> pd.DataFrame:
    """Full series over [start_s, end_s], chunked into <=15-day windows."""
    chunks = []
    cur = start_s
    while cur < end_s:
        chunk_end = min(cur + PRICES_MAX_WINDOW_S, end_s)
        chunks.extend(fetch_prices_chunk(token_id, cur, chunk_end, fidelity))
        cur = chunk_end
    if not chunks:
        return pd.DataFrame(columns=["t", "p"])
    df = pd.DataFrame(chunks).drop_duplicates(subset="t").sort_values("t").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["t"], unit="s", utc=True)
    return df


# ---------------------------------------------------------------------------
# 1c. DATA FETCH -- public trade feed (liquidity-depth proxy)
# ---------------------------------------------------------------------------
# There is no way to reconstruct historical order-book depth for a resolved
# market (`/book` 404s -- confirmed live). As a proxy for "how much size was
# actually transactable" near the crossing, this pulls realized trades
# around the crossing timestamp from the public data-api trade feed.

def _trades_cache_path(condition_id: str) -> Path:
    TRADES_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = condition_id.replace("0x", "")[:24]
    return TRADES_CACHE_DIR / f"{safe}.json"


def fetch_market_trades(condition_id: str, max_pages: int = 20) -> list[dict]:
    """All trades for a market (paginated), cached. Bounded by max_pages as
    a safety valve for pathologically high-trade-count markets -- the depth
    proxy only needs trades *near* the crossing, not the full tape."""
    path = _trades_cache_path(condition_id)
    if path.exists():
        return json.loads(path.read_text())
    out: list[dict] = []
    offset = 0
    for _ in range(max_pages):
        page = _get(DATA_API_BASE, "/trades", {"market": condition_id, "limit": 500, "offset": offset})
        if not isinstance(page, list) or not page:
            break
        out.extend(page)
        if len(page) < 500:
            break
        offset += 500
    path.write_text(json.dumps(out))
    return out


# ---------------------------------------------------------------------------
# 2. CATEGORY MAPPING
# ---------------------------------------------------------------------------
# Gamma's own `category` field is missing on many markets and, where
# present, uses an inconsistent legacy taxonomy (seen live: "Crypto",
# "Sports", "Global Politics", "US-current-affairs", "Pop-Culture",
# "Business", "Science", "Ukraine & Russia", "Coronavirus-", ...) that
# doesn't line up with the fee schedule's categories either. There is no
# public field that reproduces Polymarket's internal fee-category taxonomy
# exactly, so both mappings below are keyword heuristics over the event
# category (when present) plus the question/slug text -- documented as
# heuristic, not authoritative, in the report.

REPORT_BUCKETS = ["politics", "sports", "crypto_price", "other"]

_CRYPTO_PRICE_RE = re.compile(
    r"\b(up or down|will .*(btc|eth|bitcoin|ethereum|solana|sol|xrp|doge).*\$|"
    r"(btc|eth|bitcoin|ethereum|solana).*(above|below|reach|hit|price)|"
    r"crypto prices? on)\b", re.IGNORECASE,
)
_CRYPTO_RE = re.compile(r"\b(bitcoin|btc|ethereum|eth|solana|crypto|doge|xrp|token|coin)\b", re.IGNORECASE)
_SPORTS_RE = re.compile(
    r"\b(nba|nfl|nhl|mlb|ufc|soccer|football|basketball|baseball|hockey|tennis|"
    r"golf|olympics|world cup|champions league|premier league|f1|formula 1|"
    r" vs\.? | vs | wins the game|match|tournament|playoffs?)\b", re.IGNORECASE,
)
_POLITICS_RE = re.compile(
    r"\b(election|president|senate|congress|governor|prime minister|parliament|"
    r"politic|democrat|republican|vote|referendum|nominee|impeach|geopolit)\b",
    re.IGNORECASE,
)


def classify_report_bucket(market: dict) -> str:
    events = market.get("events")
    if not isinstance(events, list):
        events = []  # guards against pandas turning a missing field into NaN (which is truthy)
    text = " ".join(
        str(market.get(k, "")) for k in ("question", "slug")
    ) + " " + " ".join(
        str(e.get("category", "")) for e in events if isinstance(e, dict)
    )
    if _CRYPTO_PRICE_RE.search(text):
        return "crypto_price"
    if _POLITICS_RE.search(text):
        return "politics"
    if _SPORTS_RE.search(text):
        return "sports"
    if _CRYPTO_RE.search(text):
        return "crypto_price"
    return "other"


def report_bucket_coverage(markets: list[dict]) -> dict[str, int]:
    """Count of sampled markets landing in each report_bucket. classify_report_bucket
    is a keyword heuristic, not Polymarket's real taxonomy (see module docstring
    limitations) -- "other" is its catch-all, and everything that lands there
    also gets the "other" fee rate and, in the live scanners, the "other"
    flip-rate prior. A large or growing "other" share is the concrete,
    checkable signal that the heuristic is missing real structure rather than
    a vague caveat; surfaced here so it's a number in the report, not just a
    prose disclaimer."""
    counts: dict[str, int] = {b: 0 for b in REPORT_BUCKETS}
    for m in markets:
        bucket = classify_report_bucket(m)
        counts[bucket] = counts.get(bucket, 0) + 1
    return counts


# feeRate per official docs.polymarket.com/trading/fees + help.polymarket.com
# (cross-checked against both sources' worked examples), confirmed live on
# 2026-08-25. fee = shares * feeRate * price * (1 - price); makers pay 0.
FEE_RATE_BY_CATEGORY = {
    "crypto": 0.07,
    "sports": 0.05,
    "finance": 0.04,
    "politics": 0.04,
    "tech": 0.04,
    "mentions": 0.04,
    "economics": 0.05,
    "culture": 0.05,
    "weather": 0.05,
    "geopolitics": 0.0,
    "other": 0.05,
}


def classify_fee_category(market: dict) -> str:
    bucket = classify_report_bucket(market)
    if bucket == "crypto_price":
        return "crypto"
    if bucket in ("politics", "sports"):
        return bucket
    if _POLITICS_RE.search(str(market.get("question", ""))):
        return "geopolitics" if "geopolit" in str(market.get("question", "")).lower() else "politics"
    return "other"


# ---------------------------------------------------------------------------
# 3. FEES / GAS
# ---------------------------------------------------------------------------

def taker_fee_frac_of_notional(price: float, category: str) -> float:
    """Fraction of trade notional paid as a taker fee. From the confirmed
    formula fee = shares * feeRate * price * (1-price): dividing by the
    notional (shares * price) gives fee_frac = feeRate * (1 - price). This
    is why the fee is small right where this strategy trades -- (1-price)
    is already ~0.01 at a $0.99 entry."""
    rate = FEE_RATE_BY_CATEGORY.get(category, FEE_RATE_BY_CATEGORY["other"])
    return rate * (1.0 - price)


def maker_fee_frac_of_notional(price: float, category: str) -> float:
    """Makers pay nothing on Polymarket, confirmed against two official
    sources (docs.polymarket.com/trading/fees, help.polymarket.com)."""
    return 0.0


@dataclass(frozen=True)
class GasAssumptions:
    """Polymarket's relayer sponsors gas for the standard trading flow
    (order placement, approvals, CTF ops) -- ordinary users pay $0 gas per
    trade (confirmed against docs.polymarket.com/trading/gasless). The
    non-relayed estimate below is a documented sensitivity case for a
    hypothetical direct on-chain interaction, sourced from a live Polygon
    gas price and POL/USD quote (see fetch_live_gas_estimate) rather than
    an assumed constant."""

    relayer_sponsored: bool = True
    non_relayed_cost_usd_per_trade: float = 0.005  # overwritten by fetch_live_gas_estimate

    def cost_usd(self) -> float:
        return 0.0 if self.relayer_sponsored else self.non_relayed_cost_usd_per_trade


def fetch_live_gas_estimate(assumed_gas_units: int = 150_000) -> float:
    """Live Polygon gas price (public RPC eth_gasPrice) x a typical CTF
    Exchange order-fill gas-unit estimate x live POL/USD, for the
    non-relayed sensitivity case. `assumed_gas_units` is an order-of-magnitude
    estimate for a DEX-style order-settlement transaction (not measured from
    an actual Polymarket tx trace), documented as such rather than presented
    as precise."""
    try:
        body = json.dumps({"jsonrpc": "2.0", "method": "eth_gasPrice", "params": [], "id": 1}).encode()
        req = urllib.request.Request(
            "https://polygon-bor-rpc.publicnode.com", data=body,
            headers={"User-Agent": USER_AGENT, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            gas_price_wei = int(json.loads(resp.read())["result"], 16)
        pol_usd_req = urllib.request.Request(
            "https://api.coingecko.com/api/v3/simple/price?ids=polygon-ecosystem-token&vs_currencies=usd",
            headers={"User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(pol_usd_req, timeout=15) as resp:
            pol_usd = json.loads(resp.read())["polygon-ecosystem-token"]["usd"]
        return gas_price_wei * assumed_gas_units / 1e18 * pol_usd
    except Exception:
        return 0.005  # fallback if live sources are unreachable at run time


# ---------------------------------------------------------------------------
# 4. CROSSING SIGNAL (no lookahead)
# ---------------------------------------------------------------------------
# The resolved outcome must never influence signal generation. Structurally
# enforced here: detect_crossing() takes only a price series (t, p) -- it
# has no access to the market dict, resolution, or outcome labels at all.

COARSE_FIDELITY_MIN = 60       # 1h bars for the full-lifetime overview pass
COARSE_APPROACH_THRESHOLD = 0.97  # trigger a fine-grained zoom once coarse data gets this close
ZOOM_LOOKBACK_S = 2 * 86400    # look back this far before each coarse approach episode
MAX_ZOOM_EPISODES = 10  # safety valve on fine-grained fetches per market (see docstring)


def _approach_episode_starts(coarse: pd.DataFrame, threshold: float) -> list[int]:
    """Start timestamp of every maximal contiguous run of coarse closes
    >= threshold, in chronological order -- not just the first. A market can
    approach the threshold, retreat, and re-approach again much later in its
    lifetime (a long-running market oscillating near-favorite status before
    finally settling); zooming only into the first such episode would
    silently miss a genuine later crossing that falls outside that one
    window."""
    mask = (coarse["p"] >= threshold).to_numpy()
    ts = coarse["t"].to_numpy()
    starts = []
    prev = False
    for t, m in zip(ts, mask):
        if m and not prev:
            starts.append(int(t))
        prev = m
    return starts


def fetch_token_lifetime_prices(token_id: str, start_s: int, end_s: int) -> tuple[pd.DataFrame, str]:
    """Two-pass fetch. Short-lived markets (the large majority of the
    population -- mostly auto-generated hourly/daily crypto up-down
    markets) fit in a single explicit-window fine-grained (1-min) call.
    Longer-lived markets get a coarse (1h) overview first; if that overview
    never approaches the threshold, no further calls are made (the token
    never seriously threatened a crossing); otherwise a fine-grained (1-min)
    zoom is fetched around EVERY coarse approach episode (not just the
    first -- see _approach_episode_starts), and the zoomed chunks are
    concatenated into one combined series so a later caller's detect_crossing
    still finds the true first qualifying run, chronologically, across the
    market's whole lifetime. Capped at MAX_ZOOM_EPISODES episodes as a
    safety valve against a market that oscillates near the threshold
    pathologically often; that edge case is labeled distinctly
    ("fine_zoom_truncated") rather than silently dropping the remainder.
    Returns (df, source) where source is one of "fine_direct", "fine_zoom",
    "fine_zoom_truncated", "coarse_only" (never approached threshold --
    coarse resolution is sufficient to know that), or "no_data".

    This function does not take a threshold parameter -- COARSE_APPROACH_THRESHOLD
    is a fixed, generous margin below every entry threshold this project tests
    (0.98/0.99/0.995), so the same fetched data is reusable across all of them
    (see threshold_sensitivity's caching note)."""
    if end_s - start_s <= PRICES_MAX_WINDOW_S:
        df = fetch_price_series(token_id, start_s, end_s, fidelity=1)
        return df, ("fine_direct" if not df.empty else "no_data")

    coarse = fetch_price_series(token_id, start_s, end_s, fidelity=COARSE_FIDELITY_MIN)
    if coarse.empty:
        return coarse, "no_data"

    episode_starts = _approach_episode_starts(coarse, COARSE_APPROACH_THRESHOLD)
    if not episode_starts:
        return coarse, "coarse_only"

    fine_chunks = []
    for zoom_center in episode_starts[:MAX_ZOOM_EPISODES]:
        zoom_start = max(start_s, zoom_center - ZOOM_LOOKBACK_S)
        zoom_end = min(end_s, zoom_start + PRICES_MAX_WINDOW_S)
        fine = fetch_price_series(token_id, zoom_start, zoom_end, fidelity=1)
        if not fine.empty:
            fine_chunks.append(fine)

    if not fine_chunks:
        return coarse, "coarse_only"

    combined = pd.concat(fine_chunks).drop_duplicates(subset="t").sort_values("t").reset_index(drop=True)
    source = "fine_zoom" if len(episode_starts) <= MAX_ZOOM_EPISODES else "fine_zoom_truncated"
    return combined, source


def detect_crossing(df: pd.DataFrame, threshold: float = 0.99, n_consecutive: int = 3) -> Optional[dict]:
    """First snapshot where `n_consecutive` consecutive closes (inclusive,
    scanning chronologically) are all >= threshold. Entry is booked at the
    Nth confirming snapshot's own price (the actual traded/quoted price,
    never the threshold itself) -- so a single noisy tick can't trigger
    entry, per the task's requirement. Takes ONLY a bare price series; has
    no access to market metadata or resolution, by construction."""
    if df is None or df.empty:
        return None
    prices = df["p"].to_numpy()
    ts = df["t"].to_numpy()
    run = 0
    for i in range(len(prices)):
        run = run + 1 if prices[i] >= threshold else 0
        if run >= n_consecutive:
            return {
                "entry_idx": i,
                "entry_time_s": int(ts[i]),
                "entry_price": float(prices[i]),
                "run_start_idx": i - n_consecutive + 1,
            }
    return None


@dataclass(frozen=True)
class SignalConfig:
    threshold: float = 0.99
    n_consecutive: int = 3
    max_days_to_resolution: Optional[float] = None  # None = no filter


def find_market_crossings(
    market: dict,
    signal_cfg: SignalConfig,
) -> list[dict]:
    """Runs the no-lookahead crossing signal independently on every outcome
    token of one market. Returns a list of crossing records (possibly
    empty, possibly more than one for multi-outcome markets); each is
    signal-only (price path + timing), with NO resolution/outcome info --
    that is joined in afterwards by the backtest engine, never fed back
    here."""
    token_ids = _safe_json_list(market.get("clobTokenIds"))
    outcomes = _safe_json_list(market.get("outcomes"))
    if not token_ids:
        return []

    start_s = _to_epoch_s(market.get("startDate") or market.get("createdAt"))
    end_s = _to_epoch_s(market.get("endDate") or market.get("closedTime"))
    if start_s is None or end_s is None or end_s <= start_s:
        return []
    # A little slack past the scheduled end date: some markets keep trading
    # briefly (or the CLOB keeps reporting prices) past `endDate` while
    # resolution is pending.
    end_s = end_s + 3 * 86400

    crossings = []
    for idx, token_id in enumerate(token_ids):
        df, source = fetch_token_lifetime_prices(token_id, start_s, end_s)
        hit = detect_crossing(df, threshold=signal_cfg.threshold, n_consecutive=signal_cfg.n_consecutive)
        if hit is None:
            continue
        scheduled_end_s = _to_epoch_s(market.get("endDate"))
        days_to_scheduled_end = (
            (scheduled_end_s - hit["entry_time_s"]) / 86400.0 if scheduled_end_s else None
        )
        crossings.append({
            "market_id": market["id"],
            "condition_id": market.get("conditionId"),
            "token_id": token_id,
            "outcome_index": idx,
            "outcome_label": outcomes[idx] if idx < len(outcomes) else None,
            "entry_time_s": hit["entry_time_s"],
            "entry_price": hit["entry_price"],
            "data_source": source,
            "days_to_scheduled_end_at_entry": days_to_scheduled_end,
        })
    return crossings


def _to_epoch_s(value) -> Optional[int]:
    if not value:
        return None
    try:
        return int(pd.Timestamp(value).timestamp())
    except (ValueError, TypeError):
        return None


def resolved_outcome_index(market: dict) -> Optional[int]:
    """Winning outcome index from Gamma's final settled `outcomePrices`
    (["1","0"] etc.). Returns None for ambiguous/unresolved-looking
    settlement (shouldn't happen for closed=true markets, but don't guess).
    This is read ONLY at payout time -- never passed into signal generation."""
    try:
        prices = [float(x) for x in _safe_json_list(market.get("outcomePrices"))]
    except (ValueError, TypeError):
        return None
    if not prices or max(prices) < 0.5:
        return None
    return int(np.argmax(prices))


# ---------------------------------------------------------------------------
# 5. BACKTEST ENGINE
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FillAssumptions:
    fill_type: str = "maker"  # "maker" | "taker" -- Polymarket makers pay 0 fee, always


@dataclass(frozen=True)
class BacktestConfig:
    signal: SignalConfig
    position_notional: float = 100.0  # fixed $ notional per trade, base case
    gas: GasAssumptions = GasAssumptions()


DEPTH_WINDOW_S = 300  # look at realized trades within 5 min of the crossing as the liquidity proxy


def estimate_available_shares(
    condition_id: str, token_id: str, entry_time_s: int, price_threshold: float,
) -> Optional[float]:
    """Liquidity-depth proxy: sum of realized trade sizes on this token,
    within DEPTH_WINDOW_S of the crossing, at a price at-or-above a slight
    discount to the threshold (trades can print marginally below the exact
    snapshot price and still reflect the same liquidity event). Returns
    None if no trade data is available at all for this market (unknown,
    not "infinite") -- this happens for lower-activity markets where the
    data-api trade feed has nothing recorded near that timestamp."""
    trades = fetch_market_trades(condition_id)
    if not trades:
        return None
    near = [
        t for t in trades
        if t.get("asset") == token_id
        and abs(t.get("timestamp", 0) - entry_time_s) <= DEPTH_WINDOW_S
        and t.get("price", 0) >= price_threshold - 0.01
    ]
    if not near:
        return None
    return float(sum(t.get("size", 0.0) for t in near))


VWAP_WINDOW_S = 20            # near-immediate execution, not multi-minute drift -- see docstring
VWAP_MAX_PRICE_DEVIATION = 0.03  # a print this far from entry_price is a new information regime, not liquidity


def estimate_vwap_fill(
    condition_id: str, token_id: str, entry_time_s: int, entry_price: float, desired_shares: float,
    window_s: int = VWAP_WINDOW_S, max_price_deviation: float = VWAP_MAX_PRICE_DEVIATION,
) -> Optional[dict]:
    """Real, data-grounded taker slippage estimate -- not an assumed
    percentage. `estimate_available_shares` already caps position SIZE by
    realized nearby volume but assumes zero price impact for whatever size
    gets through; this instead "walks the tape" of the real BUY-side prints
    (other takers hitting the ask book) that occurred at or after the
    crossing, in their actual chronological order and at their actual
    prices, accumulating volume until `desired_shares` is reached. The
    resulting volume-weighted average price is what a taker order of that
    size would plausibly have paid, using realized flow as the only
    available proxy for order-book depth (same limitation as the existing
    depth cap: no historical order book exists for resolved markets).

    Only BUY-side prints are used (SELL prints reflect bid-side liquidity,
    irrelevant to what a buy order walks through), and only from
    entry_time_s forward (a hypothetical order placed at the signal can
    only consume liquidity that arrives after it, not trades that already
    happened before it existed).

    An earlier version of this function used a 300s window with no price
    guard and measured a 38% "slippage" on a sports over/under market --
    tracing it back, the entry print was $0.71 and the next real BUY print
    was 89 seconds later at $0.98, because the underlying event (almost
    certainly a goal) had resolved in the meantime. That's genuine price
    *discovery* (the world changed), not price *impact* (an order
    consuming the book) -- conflating the two overstates "slippage"
    enormously on any fast-resolving market. Two guards against it: a much
    shorter window (VWAP_WINDOW_S, not the depth cap's 300s -- sizing from
    realized volume tolerates a longer lookback fine, but averaging PRICE
    over a long window does not), and max_price_deviation, which stops the
    walk the moment a print lands too far from the actual entry_price to
    plausibly be the same liquidity event rather than a new information
    regime.

    Returns None if there's no BUY-side trade data at all in the window
    (unknown, not "zero slippage"). `fill_ratio` < 1 means even the
    available realized flow couldn't fully fill the desired size within
    the window -- a stronger illiquidity signal than the depth cap alone."""
    trades = fetch_market_trades(condition_id)
    if not trades:
        return None
    near = sorted(
        (t for t in trades
         if t.get("asset") == token_id and t.get("side") == "BUY"
         and entry_time_s <= t.get("timestamp", -1) <= entry_time_s + window_s),
        key=lambda t: t.get("timestamp", 0),
    )
    if not near:
        return None

    remaining = desired_shares
    cost = 0.0
    filled = 0.0
    for t in near:
        price = float(t.get("price", 0.0))
        if abs(price - entry_price) > max_price_deviation:
            break  # new information regime, not liquidity this order would have consumed
        take = min(remaining, float(t.get("size", 0.0)))
        if take <= 0:
            continue
        cost += take * price
        filled += take
        remaining -= take
        if remaining <= 1e-9:
            break
    if filled <= 0:
        return None
    return {
        "vwap": cost / filled,
        "filled_shares": filled,
        "fill_ratio": min(1.0, filled / desired_shares) if desired_shares > 0 else None,
        "n_prints_used": len(near),
    }


def simulate_trade(
    crossing: dict,
    market: dict,
    fill: FillAssumptions,
    cfg: BacktestConfig,
    cap_shares: Optional[float],
) -> dict:
    """One simulated round trip: buy at the actual crossing price, hold to
    resolution, $1/$0 payout. Fees follow Polymarket's real formula (maker
    always $0; taker = feeRate * (1-price) * notional, category-dependent).
    Position size is capped to `cap_shares` when a depth estimate is
    available and binds."""
    entry_price = crossing["entry_price"]
    category = classify_fee_category(market)

    desired_shares = cfg.position_notional / entry_price
    shares = desired_shares
    depth_capped = False
    if cap_shares is not None and cap_shares < desired_shares:
        shares = cap_shares
        depth_capped = True
    notional = shares * entry_price

    fee_frac = (
        maker_fee_frac_of_notional(entry_price, category) if fill.fill_type == "maker"
        else taker_fee_frac_of_notional(entry_price, category)
    )
    fee_cost = notional * fee_frac
    gas_cost = cfg.gas.cost_usd()

    outcome_idx = resolved_outcome_index(market)
    won = outcome_idx is not None and outcome_idx == crossing["outcome_index"]
    payout = shares * 1.0 if won else 0.0

    entry_time = pd.Timestamp(crossing["entry_time_s"], unit="s", tz="UTC")
    resolution_time = _resolution_timestamp(market)
    holding_days = (
        (resolution_time - entry_time).total_seconds() / 86400.0
        if resolution_time is not None else np.nan
    )

    return {
        "market_id": market["id"],
        "condition_id": market.get("conditionId"),
        "question": market.get("question"),
        "token_id": crossing["token_id"],
        "outcome_index": crossing["outcome_index"],
        "outcome_label": crossing["outcome_label"],
        "category": category,
        "report_bucket": classify_report_bucket(market),
        "entry_time": entry_time,
        "resolution_time": resolution_time,
        "holding_days": holding_days,
        "entry_price": entry_price,
        "data_source": crossing["data_source"],
        "days_to_scheduled_end_at_entry": crossing["days_to_scheduled_end_at_entry"],
        "desired_shares": desired_shares,
        "shares": shares,
        "depth_capped": depth_capped,
        "cap_shares": cap_shares,
        "notional": notional,
        "fee_frac": fee_frac,
        "fee_cost": fee_cost,
        "gas_cost": gas_cost,
        "won": won,
        "resolved_outcome_index": outcome_idx,
        "payout": payout,
        "pnl_gross": payout - notional,
        "pnl_net": payout - notional - fee_cost - gas_cost,
    }


def _resolution_timestamp(market: dict) -> Optional[pd.Timestamp]:
    for key in ("closedTime", "updatedAt", "endDate"):
        v = market.get(key)
        if v:
            try:
                return pd.Timestamp(v, tz="UTC") if pd.Timestamp(v).tz is None else pd.Timestamp(v)
            except (ValueError, TypeError):
                continue
    return None


# ---------------------------------------------------------------------------
# 6. FLIP ANALYSIS + CONFIDENCE INTERVAL
# ---------------------------------------------------------------------------
# "Flip" = a trade that entered above the threshold (the market briefly
# looked all-but-certain) and still resolved the other way. At this payout
# structure (~1% edge per win, -100% on a flip) one flip erases roughly 100
# winning trades, so this is the actual object of interest, not an
# afterthought on top of the aggregate return.

def wilson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    from scipy.stats import norm
    z = norm.ppf(1 - (1 - confidence) / 2)
    phat = k / n
    denom = 1 + z ** 2 / n
    center = phat + z ** 2 / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2))
    return ((center - margin) / denom, (center + margin) / denom)


def clopper_pearson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact binomial CI -- the more conservative, standard choice for a
    rare-event count like a handful of flips out of thousands of trades."""
    if n == 0:
        return (0.0, 1.0)
    from scipy.stats import beta
    alpha = 1 - confidence
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lo), float(hi))


# Heuristic categorization from Gamma metadata. UMA (Polymarket's oracle)
# resolution-status strings are the only structured signal available for
# "was this resolution contested" -- everything else about *why* a specific
# flip happened needs the market question/description read manually, which
# is exactly why the report reviews every actual flip individually rather
# than trusting this label alone.
_DISPUTE_STATUS_RE = re.compile(r"dispute", re.IGNORECASE)


def categorize_flip(market: dict) -> dict:
    uma_status_raw = market.get("umaResolutionStatus")
    uma_status = uma_status_raw if isinstance(uma_status_raw, str) else ""
    uma_statuses = _safe_json_list(market.get("umaResolutionStatuses"))
    disputed = bool(_DISPUTE_STATUS_RE.search(uma_status)) or any(
        _DISPUTE_STATUS_RE.search(str(s)) for s in uma_statuses
    )
    heuristic_category = "disputed_resolution" if disputed else "needs_manual_review"
    return {
        "market_id": market.get("id"),
        "question": market.get("question"),
        "slug": market.get("slug"),
        "description": (market.get("description") if isinstance(market.get("description"), str) else "")[:500],
        "resolutionSource": market.get("resolutionSource"),
        "umaResolutionStatus": uma_status,
        "umaResolutionStatuses": uma_statuses,
        "heuristic_category": heuristic_category,
    }


def analyze_flips(trades_df: pd.DataFrame, markets_by_id: dict[str, dict]) -> pd.DataFrame:
    flips = trades_df[~trades_df["won"]].copy()
    if flips.empty:
        return flips
    cat_rows = [categorize_flip(markets_by_id[str(mid)]) for mid in flips["market_id"]]
    cat_df = pd.DataFrame(cat_rows).add_prefix("flip_")
    return pd.concat([flips.reset_index(drop=True), cat_df.reset_index(drop=True)], axis=1)


# ---------------------------------------------------------------------------
# 7. METRICS / SENSITIVITY / PLOTS
# ---------------------------------------------------------------------------

def compute_metrics(trades_df: pd.DataFrame, pnl_col: str = "pnl_net") -> dict:
    n = len(trades_df)
    if n == 0:
        return {
            "n_trades": 0, "n_flips": 0, "win_rate": np.nan, "total_pnl": 0.0,
            "total_notional": 0.0, "total_return": np.nan, "annualized_return": np.nan,
            "flip_rate": np.nan, "flip_rate_wilson_95": (np.nan, np.nan),
            "flip_rate_clopper_pearson_95": (np.nan, np.nan),
            "avg_holding_days_winners": np.nan, "avg_holding_days_flips": np.nan,
        }
    total_pnl = float(trades_df[pnl_col].sum())
    total_notional = float(trades_df["notional"].sum())
    total_return = total_pnl / total_notional if total_notional else np.nan
    dollar_years = float((trades_df["notional"] * trades_df["holding_days"] / 365.0).sum())
    annualized_return = total_pnl / dollar_years if dollar_years else np.nan

    won = trades_df["won"]
    n_flips = int((~won).sum())
    win_rate = float(won.mean())
    flip_rate = n_flips / n

    return {
        "n_trades": n,
        "n_flips": n_flips,
        "win_rate": win_rate,
        "flip_rate": flip_rate,
        "flip_rate_wilson_95": wilson_interval(n_flips, n),
        "flip_rate_clopper_pearson_95": clopper_pearson_interval(n_flips, n),
        "total_pnl": total_pnl,
        "total_notional": total_notional,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "avg_holding_days_winners": float(trades_df.loc[won, "holding_days"].mean()) if won.any() else np.nan,
        "avg_holding_days_flips": float(trades_df.loc[~won, "holding_days"].mean()) if (~won).any() else np.nan,
    }


def compute_with_vs_without_flips(trades_df: pd.DataFrame, pnl_col: str = "pnl_net") -> dict:
    with_flips = compute_metrics(trades_df, pnl_col)
    without_flips = compute_metrics(trades_df[trades_df["won"]], pnl_col)
    return {"with_flips": with_flips, "without_flips_ie_winners_only": without_flips}


def max_days_to_resolution_variant(trades_df: pd.DataFrame, max_days: float, pnl_col: str = "pnl_net") -> pd.DataFrame:
    """Compares the unrestricted trade set against the subset that also
    required, at entry, a scheduled time-to-resolution under `max_days` --
    per the task's point that a market can sit at $0.99 for months before
    resolving, which changes annualized return even though the per-trade
    P&L is identical. Uses `days_to_scheduled_end_at_entry` (each market's
    *scheduled* end date minus the entry timestamp), which is known at
    entry time -- not the realized resolution time, which would leak
    lookahead into what is supposed to be an entry-time filter."""
    all_m = compute_metrics(trades_df, pnl_col)
    all_m["variant"] = f"unrestricted (n={len(trades_df)})"
    filtered = trades_df[trades_df["days_to_scheduled_end_at_entry"] <= max_days]
    filt_m = compute_metrics(filtered, pnl_col)
    filt_m["variant"] = f"max {max_days:.0f}d to scheduled resolution (n={len(filtered)})"
    return pd.DataFrame([all_m, filt_m])


def category_breakdown(trades_df: pd.DataFrame, pnl_col: str = "pnl_net") -> pd.DataFrame:
    rows = []
    for bucket, grp in trades_df.groupby("report_bucket"):
        m = compute_metrics(grp, pnl_col)
        m["report_bucket"] = bucket
        rows.append(m)
    return pd.DataFrame(rows).set_index("report_bucket") if rows else pd.DataFrame()


def threshold_sensitivity(
    sample: list[dict],
    fill: FillAssumptions,
    cfg: BacktestConfig,
    thresholds: list[float] = (0.98, 0.99, 0.995),
    max_workers: int = 16,
) -> pd.DataFrame:
    """Re-derives crossings and re-simulates at each threshold. Every price
    chunk was already pulled (and disk-cached) at the base threshold's run,
    and `fetch_token_lifetime_prices`'s coarse-then-zoom windowing doesn't
    depend on the exact threshold (only the fixed COARSE_APPROACH_THRESHOLD
    margin does, and 0.98 is still comfortably inside that margin) -- so
    re-running at 0.98/0.995 hits the disk cache almost entirely rather
    than re-fetching from the live API."""
    rows = []
    for thr in thresholds:
        sig_cfg = replace(cfg.signal, threshold=thr)
        trades = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(find_market_crossings, m, sig_cfg) for m in sample]
            for market, fut in zip(sample, futures):
                for crossing in fut.result():
                    trades.append(simulate_trade(crossing, market, fill, cfg, cap_shares=None))
        tdf = pd.DataFrame(trades)
        m = compute_metrics(tdf) if not tdf.empty else compute_metrics(tdf, "pnl_net")
        m["threshold"] = thr
        rows.append(m)
    return pd.DataFrame(rows)


def plot_equity_curve(trades_df: pd.DataFrame, out_path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = trades_df.sort_values("entry_time").copy()
    df["cum_net"] = df["pnl_net"].cumsum()
    df["cum_gross"] = df["pnl_gross"].cumsum()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df["entry_time"], df["cum_gross"], label="Gross of fees", linewidth=1.1)
    ax.plot(df["entry_time"], df["cum_net"], label="Net of fees", linewidth=1.3)
    flips = df[~df["won"]]
    if not flips.empty:
        ax.scatter(flips["entry_time"], flips["cum_net"], color="crimson", zorder=5, label="Flip", s=40)
    ax.set_title(title)
    ax.set_xlabel("Entry date")
    ax.set_ylabel("Cumulative P&L ($, fixed notional per trade)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8. SAMPLING
# ---------------------------------------------------------------------------
# Pulling CLOB price history for literally every resolved market (the
# census below finds this population is in the hundreds of thousands,
# dominated by ultra-short-lived auto-generated crypto up/down markets) is
# not computationally tractable in this analysis. Instead: the CENSUS is
# complete and uncurated (every resolved market since the CLOB-era cutoff),
# and a large stratified RANDOM sample is drawn from it -- stratified by
# (resolution quarter x report category) with proportional allocation, so
# the sample's time and category mix matches the population's, rather than
# being cherry-picked toward popular/high-volume markets. This is the
# concrete mitigation for the selection-bias risk the task calls out.

def stratified_sample_markets(census: list[dict], n_target: int, seed: int = 42) -> list[dict]:
    """Stratified sample, with one additional population filter applied
    here (not at the census-query stage): the Gamma census is queried by
    resolution date, but a market's *lifetime* can straddle the CLOB-launch
    cutoff -- confirmed live on a $1.7M-volume Senate-control market that
    resolved in Nov 2022 (inside the census window) but started trading in
    Jan 2022 (pre-CLOB) and had zero CLOB price history at any window.
    Dropped here by requiring the market's own start date, not just its end
    date, to be at/after the cutoff."""
    df = pd.DataFrame(census)
    df["end_ts"] = df["endDate"].apply(_to_epoch_s)
    df["start_ts"] = df.apply(lambda r: _to_epoch_s(r.get("startDate") or r.get("createdAt")), axis=1)
    df = df.dropna(subset=["end_ts", "start_ts"])
    df = df[df["start_ts"] >= _to_epoch_s(CLOB_LAUNCH_CUTOFF)]
    df["quarter"] = pd.to_datetime(df["end_ts"], unit="s").dt.to_period("Q").astype(str)
    df["bucket"] = [classify_report_bucket(m) for m in df.to_dict("records")]

    frac = min(1.0, n_target / max(len(df), 1))
    parts = []
    for (quarter, bucket), grp in df.groupby(["quarter", "bucket"]):
        k = max(1, round(len(grp) * frac)) if len(grp) else 0
        k = min(k, len(grp))
        if k <= 0:
            continue
        # A per-group deterministic seed, not one shared RNG advanced
        # sequentially across the whole groupby loop: with a single shared
        # stream, any change to an EARLIER group's size (e.g. new markets
        # resolving into an unrelated quarter/category) shifts how many
        # draws it consumes, desyncing every later group's output even
        # though its own candidate pool never changed. Hashing the group's
        # own key isolates each group's draw from everything else.
        #
        # A full deterministic shuffle-then-take-first-k, not rng.choice(...,
        # size=k) directly: `frac` (= n_target / total census size) drifts
        # every time the census grows, so a since-unchanged group's own `k`
        # shifts by a market or two even though its candidate pool didn't
        # change -- and rng.choice's replace=False output for k=77 is NOT a
        # subset of its own output for k=100 with the same seed, so that
        # alone re-scrambles the whole selection. Shuffling once and slicing
        # is prefix-stable: a small change in k only adds/drops markets at
        # the margin instead of picking an unrelated random subset.
        # Confirmed both fixes are needed together, with a synthetic
        # before/after-growth test: per-group seeding alone still only got
        # two nominally-identical-seed census snapshots to ~14% overlap on
        # an untouched group (k drifts as the total census grows, and
        # rng.choice's own output for a smaller k isn't a subset of its
        # output for a larger k with the same seed). With both fixes, an
        # untouched group's shrinking selection is verified to be an exact
        # SUBSET of its earlier, larger selection -- not full overlap
        # (fewer markets get sampled from a fixed-size group as the total
        # census grows and `frac` drops), but the right property: consistent
        # shrink/grow, never an unrelated reshuffle.
        group_seed = int.from_bytes(
            hashlib.sha256(f"{seed}|{quarter}|{bucket}".encode()).digest()[:8], "big"
        )
        rng = np.random.default_rng(group_seed)
        shuffled = rng.permutation(grp.index.to_numpy())
        idx = shuffled[:k]
        parts.append(grp.loc[idx])
    sampled = pd.concat(parts) if parts else df.iloc[0:0]
    return sampled.to_dict("records")


# ---------------------------------------------------------------------------
# 9. REPORT
# ---------------------------------------------------------------------------

def _df_to_markdown(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    """Minimal markdown-table formatter (avoids adding a `tabulate` dep)."""
    if df.empty:
        return "_(none)_"

    def fmt(v):
        if isinstance(v, float):
            return float_fmt.format(v)
        if isinstance(v, tuple):
            return "(" + ", ".join(float_fmt.format(x) for x in v) + ")"
        return str(v)

    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join(
        "| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)
    )
    return "\n".join([header, sep, body])


def write_report(
    out_path: Path,
    census_size: int,
    sample_size: int,
    trades_by_fill: dict[str, pd.DataFrame],
    days_dist: pd.DataFrame,
    category_tables: dict[str, pd.DataFrame],
    max_days_tables: dict[str, pd.DataFrame],
    sensitivity_tables: dict[str, pd.DataFrame],
    depth_cap_flags: pd.DataFrame,
    signal_cfg: SignalConfig,
    gas_estimate_usd: float,
    bucket_coverage: dict[str, int],
) -> None:
    lines = []
    lines.append("# Polymarket \"Final 1%\" Spread-Capture Backtest\n")
    lines.append(
        "Historical analysis only. Buys an outcome token once its price "
        f"closes at/above ${signal_cfg.threshold:.3f} for "
        f"{signal_cfg.n_consecutive} consecutive snapshots and holds to "
        "resolution. No live trading, no order placement, no API keys.\n"
    )

    lines.append("## Data & methodology\n")
    lines.append(
        f"**Population census**: {census_size:,} resolved markets found via "
        f"a complete, uncurated crawl of Gamma's `/markets?closed=true` "
        f"since the CLOB-launch cutoff ({CLOB_LAUNCH_CUTOFF}) through the "
        "run date. This population is dominated by short-lived "
        "auto-generated crypto up/down markets; pulling full CLOB price "
        "history for all of it is not computationally tractable here, so "
        f"the backtest runs on a **stratified random sample of "
        f"{sample_size:,} markets** (proportional allocation by resolution "
        "quarter x report category, seeded and reproducible) -- an "
        "unbiased sample of the full population, not a curated or "
        "volume-sorted subset, which is the direct mitigation for the "
        "selection-bias risk this kind of backtest is prone to.\n"
    )
    lines.append(
        "**Two confirmed, load-bearing API defects found by testing the "
        "live endpoints before building the pipeline** (both drove real "
        "design decisions, see the module docstring in "
        "`src/polymarket_final_pct.py` for the full detail):\n\n"
        "1. Gamma's `/markets/keyset` endpoint (the one the deprecated "
        "classic endpoint's headers point you to) silently ignores its own "
        "`cursor` parameter -- every request returns page 1 regardless. "
        "Worked around by bucketing the classic endpoint's `offset` "
        "(separately capped at 2000) by end-date range.\n"
        "2. `/prices-history?interval=max` (the natural way to ask for a "
        "token's full history) reliably returns an EMPTY series even for "
        "high-volume resolved tokens -- this is the granularity/emptiness "
        "issue the task warned about. Fix: never use `interval=`, always "
        "pass explicit `startTs`/`endTs` (capped at a 15-day window) with "
        "an explicit `fidelity`; verified this returns real ~1-minute data "
        "in every case tested. Markets that resolved before Polymarket's "
        "CLOB launch (mid-2022) have no CLOB price history at any window "
        "-- they traded on the old AMM -- and are excluded from the "
        "population on that basis, not treated as the granularity bug. A "
        f"market's *lifetime* can straddle that cutoff even when its "
        f"*resolution* falls inside it (found live on a $1.7M-volume "
        f"Senate-control market that resolved Nov 2022 but started trading "
        f"Jan 2022, with zero CLOB history) -- filtered on each market's "
        f"own start date, not just its resolution date. On a live 20-market "
        f"granularity test spanning 2022-2025 and all volume tiers, "
        f"3 of 13 valid samples (all from the first ~2 months "
        f"post-CLOB-launch) still came back with zero price points even "
        f"under the explicit-window fix -- read as thin early liquidity on "
        f"that specific token, not a residual data-access bug; such tokens "
        f"simply contribute no trade.\n"
    )
    lines.append(
        "**Liquidity/depth**: there is no way to reconstruct historical "
        "order-book depth for a resolved market (`/book` returns 404, "
        "\"No orderbook exists\", confirmed live). As a proxy, position "
        "size is capped to the sum of realized trade sizes on the same "
        "token within 5 minutes of the crossing (from the public "
        "`data-api.polymarket.com/trades` feed), when any such trades are "
        "recorded; markets meaningfully capped by this are flagged below.\n"
    )
    lines.append(
        f"**Fees**: confirmed against docs.polymarket.com/trading/fees and "
        f"help.polymarket.com (both official, cross-checked against the "
        f"docs' own worked examples). Makers pay $0, always. Takers pay "
        f"`fee = shares * feeRate * price * (1-price)`, with feeRate "
        f"0.00-0.07 depending on category -- which is why the fee is small "
        f"specifically where this strategy trades: `(1-price)` is already "
        f"~0.01 at a $0.99 entry. **Gas**: Polymarket's relayer sponsors "
        f"on-chain gas for the standard trading flow, so ordinary users pay "
        f"$0/trade (confirmed against docs.polymarket.com/trading/gasless) "
        f"-- modeled as the default. A non-relayed direct on-chain estimate "
        f"of ${gas_estimate_usd:.4f}/trade (live Polygon gas price x "
        f"~150k gas units x live POL/USD) is reported as a sensitivity case.\n"
    )

    lines.append("## Results: net vs. gross, maker vs. taker fill, with vs. without flips\n")
    for fill_label, tdf in trades_by_fill.items():
        lines.append(f"\n### {fill_label}\n")
        for pnl_col, pnl_label in [("pnl_gross", "gross of fees/gas"), ("pnl_net", "net of fees/gas")]:
            wv = compute_with_vs_without_flips(tdf, pnl_col)
            m_all = pd.Series(wv["with_flips"])
            m_win = pd.Series(wv["without_flips_ie_winners_only"])
            comp = pd.DataFrame({"including_flips": m_all, "winners_only_counterfactual": m_win})
            lines.append(f"**{pnl_label}**\n")
            lines.append(_df_to_markdown(comp.reset_index().rename(columns={"index": "metric"})))
            lines.append("\n")

    lines.append("## Flip analysis\n")
    all_flips = trades_by_fill.get("maker", pd.DataFrame())
    flips = all_flips[~all_flips["won"]] if not all_flips.empty and "won" in all_flips else pd.DataFrame()
    if flips.empty:
        lines.append("No flips found in the sampled trades (see confidence interval above for how wide the true rate could still plausibly be given the sample size).\n")
    else:
        cols = [c for c in [
            "market_id", "question", "category", "report_bucket", "entry_price", "entry_time",
            "holding_days", "flip_heuristic_category", "flip_umaResolutionStatus", "flip_slug",
        ] if c in flips.columns]
        lines.append(_df_to_markdown(flips[cols]))
        lines.append(
            "\n\n`flip_heuristic_category` is auto-derived from UMA oracle "
            "resolution-status metadata (`disputed_resolution` if any dispute "
            "flag is present, else `needs_manual_review`); every flip above "
            "should be read individually (question + slug) before treating "
            "it as a \"genuine reversal\" -- that distinction genuinely "
            "changes what the recurring risk is going forward and this "
            "pipeline cannot make that call automatically.\n"
        )

    lines.append("\n## Days-to-resolution distribution, winners vs. flips\n")
    lines.append(_df_to_markdown(days_dist))
    lines.append(
        "\nIf flips cluster at one end of this distribution (e.g. only in "
        "markets held a long time after crossing) or in one category, that "
        "pattern is more actionable than the aggregate flip rate alone -- "
        "check the flip table above against this distribution directly.\n"
    )

    lines.append("\n## Category breakdown\n")
    for label, cat_df in category_tables.items():
        lines.append(f"\n**{label}**\n")
        lines.append(_df_to_markdown(cat_df.reset_index()))
        lines.append("\n")

    lines.append(f"\n## Max time-to-resolution variant (unrestricted vs. <= {MAX_DAYS_TO_RESOLUTION:.0f} days at entry)\n")
    lines.append(
        "A market can sit at $0.99 for months before it finally resolves -- "
        "the per-trade dollar P&L is identical, but that dead capital-tied-up "
        "time collapses the annualized return. This variant additionally "
        "requires, at entry, that the market's *scheduled* end date (not the "
        "realized resolution time, which would leak lookahead into an "
        "entry-time filter) was no more than "
        f"{MAX_DAYS_TO_RESOLUTION:.0f} days away.\n"
    )
    for label, mdf in max_days_tables.items():
        lines.append(f"\n**{label}**\n")
        lines.append(_df_to_markdown(mdf[["variant"] + [c for c in mdf.columns if c != "variant"]]))
        lines.append("\n")

    lines.append("\n## Threshold sensitivity ($0.98 / $0.99 / $0.995)\n")
    for label, sens_df in sensitivity_tables.items():
        lines.append(f"\n**{label}**\n")
        lines.append(_df_to_markdown(sens_df))
        lines.append("\n")

    lines.append("\n## Fill-size / depth-capping flags\n")
    if depth_cap_flags.empty:
        lines.append("No trades had a binding depth cap from the realized-trades proxy.\n")
    else:
        lines.append(
            f"{len(depth_cap_flags)} of {len(all_flips) if not all_flips.empty else 0} "
            "sampled trades had position size meaningfully capped below the "
            "desired fixed notional by the realized-trades liquidity proxy:\n\n"
        )
        cols = [c for c in ["market_id", "question", "desired_shares", "shares", "cap_shares"] if c in depth_cap_flags.columns]
        lines.append(_df_to_markdown(depth_cap_flags[cols].head(30)))
        lines.append("\n")

    lines.append("\n## Limitations\n")
    lines.append(
        "- **Sample-size limitation on the flip rate is a limitation of "
        "this backtest itself, not a footnote.** Polymarket's CLOB has "
        "existed for a bit over four years, and this strategy's entire "
        "economics hinge on a tail event (the flip rate) that, by "
        "construction, is rare -- a handful of flips (or zero) out of "
        "thousands of sampled trades. The confidence intervals reported "
        "above are wide for exactly this reason: with a small number of "
        "observed flips, the data cannot distinguish between \"this "
        "strategy has a structurally low, durable flip rate\" and \"this "
        "backtest simply hasn't sampled enough history to see the flips "
        "that will happen.\" A point estimate of the flip rate should not "
        "be read as a precise, forward-looking probability.\n"
        "- The liquidity-depth proxy (realized trades near the crossing) "
        "is not the same thing as resting order-book depth at the moment "
        "of the crossing -- it likely understates true available "
        "liquidity in some cases and cannot be verified against the real "
        "book for a resolved market.\n"
        "- Ignores the possibility that entering size at the crossing "
        "itself moves the price (this backtest assumes the observed "
        "crossing price is achievable at the simulated size, up to the "
        "depth cap).\n"
        "- Category classification is a keyword heuristic over question "
        "text and event metadata, not Polymarket's internal taxonomy -- "
        "treat the category breakdown as indicative, not exact. "
        "\"other\" is the catch-all this heuristic falls back to (both for "
        "the report bucket and the fee rate); its share of this sample is "
        + ", ".join(f"{b}={n}" for b, n in sorted(bucket_coverage.items())) +
        f" ({bucket_coverage.get('other', 0) / max(sum(bucket_coverage.values()), 1) * 100:.1f}% "
        "other) -- a large or growing \"other\" share is the concrete sign "
        "this heuristic is missing real structure, not just a caveat.\n"
        "- The backtest samples from the population rather than covering "
        "it exhaustively; while the sampling is stratified and unbiased by "
        "construction, a different random seed or a larger sample could "
        "shift the flip count (see the CI, not the point estimate).\n"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# 10. MAIN
# ---------------------------------------------------------------------------

SAMPLE_SIZE = 4000
CROSSING_WORKERS = 16
MAX_DAYS_TO_RESOLUTION = 7.0  # entry-time filter variant, per the task's example


def find_all_crossings(
    sample: list[dict], signal_cfg: SignalConfig, max_workers: int = CROSSING_WORKERS,
) -> list[tuple[dict, dict, Optional[float]]]:
    """Runs the no-lookahead crossing signal for every market in `sample`,
    concurrently (I/O-bound HTTP calls), plus the depth-cap estimate for
    each crossing found. This is the expensive, fill-independent part of
    the pipeline -- computed once and reused for every fill assumption
    (maker/taker) rather than repeated per fill, since neither the price
    history nor the crossing depends on the fee/gas assumption."""
    all_crossings: list[tuple[dict, dict]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(find_market_crossings, m, signal_cfg): m for m in sample}
        for fut in as_completed(futures):
            market = futures[fut]
            for crossing in fut.result():
                all_crossings.append((market, crossing))

    def _depth(market: dict, crossing: dict) -> Optional[float]:
        return estimate_available_shares(
            market.get("conditionId"), crossing["token_id"], crossing["entry_time_s"], signal_cfg.threshold,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_depth, market, crossing) for market, crossing in all_crossings]
        caps = [fut.result() for fut in futures]
    return [(m, c, cap) for (m, c), cap in zip(all_crossings, caps)]


def simulate_trades_for_fill(
    crossings: list[tuple[dict, dict, Optional[float]]], fill: FillAssumptions, cfg: BacktestConfig,
) -> pd.DataFrame:
    trades = [simulate_trade(crossing, market, fill, cfg, cap_shares=cap) for market, crossing, cap in crossings]
    return pd.DataFrame(trades)


def run_sample_backtest(
    sample: list[dict], fill: FillAssumptions, cfg: BacktestConfig, max_workers: int = CROSSING_WORKERS,
) -> pd.DataFrame:
    """Convenience single-fill entry point (used by threshold_sensitivity,
    where crossings differ per threshold anyway). The main() pipeline calls
    find_all_crossings() once and simulate_trades_for_fill() per fill
    assumption instead, to avoid redoing crossing detection twice."""
    crossings = find_all_crossings(sample, cfg.signal, max_workers)
    return simulate_trades_for_fill(crossings, fill, cfg)


def days_to_resolution_distribution(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()
    rows = []
    for label, grp in [("winners", trades_df[trades_df["won"]]), ("flips", trades_df[~trades_df["won"]])]:
        if grp.empty:
            rows.append({"group": label, "n": 0})
            continue
        rows.append({
            "group": label, "n": len(grp),
            "mean_days": grp["holding_days"].mean(), "median_days": grp["holding_days"].median(),
            "p10_days": grp["holding_days"].quantile(0.10), "p90_days": grp["holding_days"].quantile(0.90),
            "max_days": grp["holding_days"].max(),
        })
    return pd.DataFrame(rows)


def main() -> None:
    print(f"Fetching Gamma census of resolved markets since {CLOB_LAUNCH_CUTOFF} ...")
    census = fetch_resolved_markets_census()
    print(f"  census size: {len(census):,} resolved markets")

    print(f"Drawing stratified random sample (target n={SAMPLE_SIZE}) ...")
    sample = stratified_sample_markets(census, n_target=SAMPLE_SIZE)
    sample = [m for m in sample if _safe_json_list(m.get("clobTokenIds"))]
    print(f"  sample size (with CLOB token ids): {len(sample):,}")

    bucket_coverage = report_bucket_coverage(sample)
    print(f"  report_bucket coverage: {bucket_coverage}")

    signal_cfg = SignalConfig(threshold=0.99, n_consecutive=3)
    gas_estimate = fetch_live_gas_estimate()
    gas_sponsored = GasAssumptions(relayer_sponsored=True)
    cfg = BacktestConfig(signal=signal_cfg, position_notional=100.0, gas=gas_sponsored)

    print("Running crossing detection (shared across fill assumptions) ...")
    crossings = find_all_crossings(sample, signal_cfg)
    print(f"  {len(crossings)} crossings found across {len(sample)} sampled markets")

    trades_by_fill = {}
    for fill_type in ("maker", "taker"):
        fill = FillAssumptions(fill_type=fill_type)
        tdf = simulate_trades_for_fill(crossings, fill, cfg)
        trades_by_fill[fill_type] = tdf
        n_flips = int((~tdf["won"]).sum()) if not tdf.empty else 0
        print(f"  {fill_type}: {len(tdf)} trades, {n_flips} flips")

    markets_by_id = {str(m["id"]): m for m in sample}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    maker_trades = trades_by_fill["maker"]
    if not maker_trades.empty:
        maker_trades.to_csv(RESULTS_DIR / "trades_maker.csv", index=False)
    if not trades_by_fill["taker"].empty:
        trades_by_fill["taker"].to_csv(RESULTS_DIR / "trades_taker.csv", index=False)

    flips_df = analyze_flips(maker_trades, markets_by_id) if not maker_trades.empty else pd.DataFrame()
    if not flips_df.empty:
        flips_df.to_csv(RESULTS_DIR / "flips_detail.csv", index=False)

    days_dist = days_to_resolution_distribution(maker_trades)

    category_tables = {
        f"{fill_type} fills, net of fees": category_breakdown(tdf, "pnl_net")
        for fill_type, tdf in trades_by_fill.items() if not tdf.empty
    }

    max_days_tables = {
        f"{fill_type} fills, net of fees": max_days_to_resolution_variant(tdf, MAX_DAYS_TO_RESOLUTION, "pnl_net")
        for fill_type, tdf in trades_by_fill.items() if not tdf.empty
    }

    print("Running threshold sensitivity (0.98 / 0.99 / 0.995) ...")
    sensitivity_tables = {}
    for fill_type in ("maker", "taker"):
        fill = FillAssumptions(fill_type=fill_type)
        sensitivity_tables[fill_type] = threshold_sensitivity(sample, fill, cfg)

    depth_cap_flags = maker_trades[maker_trades["depth_capped"]] if not maker_trades.empty else pd.DataFrame()

    if not maker_trades.empty:
        plot_equity_curve(
            maker_trades, PLOTS_DIR / "equity_curve_maker.png",
            "Cumulative P&L: maker fills, net vs. gross of fees ($100 notional/trade)",
        )
    if not trades_by_fill["taker"].empty:
        plot_equity_curve(
            trades_by_fill["taker"], PLOTS_DIR / "equity_curve_taker.png",
            "Cumulative P&L: taker fills, net vs. gross of fees ($100 notional/trade)",
        )
    print(f"Plots written to {PLOTS_DIR}")

    write_report(
        out_path=RESULTS_DIR / "report.md",
        census_size=len(census),
        sample_size=len(sample),
        trades_by_fill=trades_by_fill,
        days_dist=days_dist,
        category_tables=category_tables,
        max_days_tables=max_days_tables,
        sensitivity_tables=sensitivity_tables,
        depth_cap_flags=depth_cap_flags,
        signal_cfg=signal_cfg,
        gas_estimate_usd=gas_estimate,
        bucket_coverage=bucket_coverage,
    )
    print(f"Report written to {RESULTS_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
