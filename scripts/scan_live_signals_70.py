"""Live scan: which currently-open Polymarket markets meet the 70%-threshold
entry criteria right now, and what position size that implies for a given
bankroll.

Read-only. This does not place orders, hold API keys, or touch a wallet --
it only reads Gamma/CLOB public endpoints and prints candidates for you to
act on manually. This is a point-in-time snapshot, not a backtest: run it
fresh each time you want current signals.

Adapted from scan_live_signals.py (the 99% version). Differences:
  - THRESHOLD = 0.70 instead of 0.99.
  - CATEGORY_FLIPS uses the per-bucket flip counts measured directly from
    trades_maker_thr07_v2.csv (post exact-score/weather exclusion), not the
    99%-threshold counts -- flip risk at 0.70 is ~12-15% per bucket, not
    ~0-0.2%, and reusing the 99% prior here would badly understate risk.
  - BANKROLL defaults to whatever you pass on the command line (a live test
    account is likely tiny; the 99% script's $1,000 default doesn't apply).

Two-pass design (added after live testing surfaced bad signals the naive
single-pass version couldn't distinguish from good ones):
  Pass 1 (cheap): scan every active market's last 3 days of price history
    for "currently qualifying" candidates -- fast, but the 3-day window
    means a market that crossed threshold long ago, spiked much higher,
    and is now falling back through it looks identical to a genuine fresh
    signal. It also can't tell a market that's been range-bound near
    threshold for months from one that just got there.
  Pass 2 (expensive, top candidates only): for each pass-1 candidate with
    positive margin, pull up to 15 days of full price history and find the
    TRUE first crossing (no lookahead, same detect_crossing() the backtest
    itself uses). Only keep it if that true crossing is recent (<=96h ago)
    AND the window shows price genuinely below threshold before it -- this
    is what actually distinguishes "just became a favorite" from "has been
    hovering here for months" or "cratering back down from a spike."  Also
    re-fetches the real live order book (best ask, actual depth, spread)
    rather than trusting the approximate quote, and recomputes margin off
    that real ask price -- a market can look edge-positive on Gamma's
    approximate price and be flat or negative once you price the real
    spread. Same-event duplicates (multi-outcome elections etc, grouped by
    negRiskMarketID) are collapsed to the single best-margin leg.
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import polymarket_final_pct as pmf
from run_kelly_backtest import load as load_trades, flip_counts_by

BANKROLL = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
MAX_POS_PCT = 0.03
AGG_CAP_PCT = 0.50
CAT_CAP_PCT = 0.25
THRESHOLD = 0.70
N_CONSECUTIVE = 3
PRIOR_A, PRIOR_B = 1.0, 40.0
# Prior mean = 1/41 = 2.4%, deliberately looser than the 99%-threshold
# script's Beta(1,300) (mean 0.33%) -- that prior was tuned for a ~0.2%
# empirical flip-rate regime and would badly understate risk at 70%, where
# measured flip rates run 12-15% per bucket. This prior still converges to
# the real per-bucket rate quickly (all buckets below have n > 700 except
# politics) but doesn't start out falsely confident.

# Read directly from the committed backtest output (trades_maker_thr07_v2.csv,
# post exact-score/weather exclusion -- same load()/exclusion regexes as
# scripts/run_kelly_backtest.py) at run time, not a hand-copied snapshot, so
# this can't silently drift from the file it's supposedly derived from.
# (flips, total resolved trades), by report_bucket.
CATEGORY_FLIPS = flip_counts_by(load_trades("trades_maker_thr07_v2.csv"), "report_bucket")


def qh(bucket: str) -> float:
    k, n = CATEGORY_FLIPS.get(bucket, (0, 0))
    return (PRIOR_A + k) / (PRIOR_A + PRIOR_B + n)


def _active_page(date_min, date_max, offset):
    return pmf._get(pmf.GAMMA_BASE, "/markets", {
        "closed": "false", "active": "true", "limit": pmf.GAMMA_PAGE_LIMIT, "offset": offset,
        "end_date_min": date_min, "end_date_max": date_max,
    })


def fetch_active_markets(date_min: str, date_max: str) -> list[dict]:
    out, offset = [], 0
    while True:
        page = _active_page(date_min, date_max, offset)
        if not page:
            break
        out.extend(page)
        if len(page) < pmf.GAMMA_PAGE_LIMIT:
            break
        offset += pmf.GAMMA_PAGE_LIMIT
        if offset > pmf.GAMMA_OFFSET_CAP:
            break
    return out


def fetch_live_book_depth(token_id: str, price_threshold: float) -> float | None:
    try:
        book = pmf._get(pmf.CLOB_BASE, "/book", {"token_id": token_id})
    except Exception:
        return None
    asks = book.get("asks") or []
    total = sum(float(a["size"]) for a in asks if float(a["price"]) <= price_threshold + 0.05)
    return total if asks else None


def check_market(market: dict) -> dict | None:
    token_ids = pmf._safe_json_list(market.get("clobTokenIds"))
    outcomes = pmf._safe_json_list(market.get("outcomes"))
    prices_raw = pmf._safe_json_list(market.get("outcomePrices"))
    if not token_ids or not prices_raw:
        return None
    try:
        prices = [float(p) for p in prices_raw]
    except (ValueError, TypeError):
        return None

    for idx, (tok, p) in enumerate(zip(token_ids, prices)):
        if p < 0.68:  # cheap pre-screen; confirm properly below
            continue
        now_s = int(time.time())
        df, source = pmf.fetch_token_lifetime_prices(tok, now_s - 3 * 86400, now_s + 60)
        if df.empty:
            continue
        hit = pmf.detect_crossing(df, threshold=THRESHOLD, n_consecutive=N_CONSECUTIVE)
        if hit is None:
            continue
        last_price = float(df["p"].iloc[-1])
        if last_price < THRESHOLD:
            continue
        category = pmf.classify_fee_category(market)
        bucket = pmf.classify_report_bucket(market)
        question = market.get("question", "")
        excluded = bool(pmf.re.search(r"^Exact Score:", question, pmf.re.I)) or \
                   bool(pmf.re.search(r"highest temperature.*(be between|be \d)", question, pmf.re.I))

        depth = fetch_live_book_depth(tok, last_price)

        return {
            "market_id": market["id"], "question": question, "outcome": outcomes[idx] if idx < len(outcomes) else None,
            "current_price": last_price, "entry_price_at_crossing": hit["entry_price"],
            "category": category, "report_bucket": bucket, "excluded": excluded,
            "end_date": market.get("endDate"), "volume": market.get("volumeNum"),
            "live_ask_depth_notional": depth,
        }
    return None


MAX_CROSSING_AGE_HOURS = 96
VERIFY_LOOKBACK_DAYS = 15
MAX_VERIFY_CANDIDATES = 80  # cap the expensive pass-2 fetch volume


def verify_candidate(hit: dict) -> dict | None:
    """Pass 2: full-history true-crossing check + real order book. Returns
    None if the candidate fails freshness (stale/already-above-at-window-open
    or crossing too old) or has no live ask at all."""
    try:
        m = pmf._get(pmf.GAMMA_BASE, f"/markets/{hit['market_id']}", {})
        token_ids = pmf._safe_json_list(m.get("clobTokenIds"))
        outcomes = pmf._safe_json_list(m.get("outcomes"))
        idx = outcomes.index(hit["outcome"])
        tok = token_ids[idx]
    except Exception:
        return None

    now_s = int(time.time())
    df, _ = pmf.fetch_token_lifetime_prices(tok, now_s - VERIFY_LOOKBACK_DAYS * 86400, now_s)
    if df.empty:
        return None

    starts_above = bool(df["p"].iloc[0] >= THRESHOLD)
    true_hit = pmf.detect_crossing(df, threshold=THRESHOLD, n_consecutive=N_CONSECUTIVE)
    if true_hit is None:
        return None
    age_hours = (now_s - true_hit["entry_time_s"]) / 3600
    if starts_above or age_hours > MAX_CROSSING_AGE_HOURS:
        return None  # stale: either already qualifying 15 days ago, or crossed too long ago to trust

    try:
        book = pmf._get(pmf.CLOB_BASE, "/book", {"token_id": tok})
    except Exception:
        return None
    asks = sorted(book.get("asks") or [], key=lambda a: float(a["price"]))
    bids = sorted(book.get("bids") or [], key=lambda b: -float(b["price"]))
    if not asks:
        return None
    best_ask = float(asks[0]["price"])
    ask_depth = float(asks[0]["size"])  # order-book size is in SHARES, not dollars
    spread = best_ask - float(bids[0]["price"]) if bids else None

    sizing = kelly_size(hit["report_bucket"], best_ask, BANKROLL)
    if sizing["margin"] <= 0:
        return None  # edge evaporates once priced off the real ask, not the approximate quote

    return {
        **hit, **sizing, "token_id": tok, "real_best_ask": best_ask, "real_ask_depth": ask_depth,
        "real_ask_depth_notional": ask_depth * best_ask,
        "real_spread": spread, "true_crossing_age_hours": round(age_hours, 1),
        "neg_risk_market_id": m.get("negRiskMarketID"),
    }


def kelly_size(bucket: str, price: float, bankroll: float, fraction: float = 0.25) -> dict:
    b = (1.0 - price) / price  # maker fill, $0 fee
    L = 1.0
    q = qh(bucket)
    p = 1.0 - q
    f_kelly = (p * b - q * L) / (b * L) if b > 0 else 0.0
    desired = max(0.0, f_kelly) * fraction * bankroll
    per_trade_capped = min(desired, MAX_POS_PCT * bankroll)
    return {"flip_belief_q": q, "kelly_fraction_raw": f_kelly, "desired_uncapped": desired,
            "per_trade_capped": per_trade_capped, "margin": p * b - q * L}


def allocate_portfolio(rows: list[dict], bankroll: float) -> list[dict]:
    """Caps the stake by real_ask_depth_notional (pass 2's freshly re-fetched
    real order book, converted from shares to dollars) -- NOT the pass-1
    live_ask_depth_notional field on `hit`, which is a stale estimate taken
    near the ORIGINAL (possibly since-moved) crossing price and can
    undercount or completely miss the real depth once price has moved
    between pass 1 and pass 2 (observed live: a candidate whose price moved
    $0.72 -> $0.85 between passes had a real $34 of depth at the new price,
    but a stale pass-1 estimate of $0 -- which silently zeroed its position
    under the old logic)."""
    ranked = sorted(rows, key=lambda r: r["margin"], reverse=True)
    agg_used = 0.0
    cat_used: dict[str, float] = {}
    for r in ranked:
        bucket = r["report_bucket"]
        room_agg = AGG_CAP_PCT * bankroll - agg_used
        room_cat = CAT_CAP_PCT * bankroll - cat_used.get(bucket, 0.0)
        stake = max(0.0, min(r["per_trade_capped"], room_agg, room_cat))
        if r.get("real_ask_depth_notional") is not None:
            stake = min(stake, r["real_ask_depth_notional"])
        r["portfolio_position_size"] = round(stake, 4)
        agg_used += stake
        cat_used[bucket] = cat_used.get(bucket, 0.0) + stake
    return ranked


def main():
    print(f"Bankroll for this scan: ${BANKROLL:.2f}")
    print("Fetching currently active (open) Polymarket markets ...")
    today = pmf.pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
    far_future = "2028-01-01"
    markets = fetch_active_markets(today, far_future)
    markets += fetch_active_markets(
        (pmf.pd.Timestamp.now("UTC") - pmf.pd.Timedelta(days=1)).strftime("%Y-%m-%d"), today
    )
    markets = pmf._dedupe_by_id(markets)
    markets = [m for m in markets if pmf._safe_json_list(m.get("clobTokenIds"))]
    print(f"  {len(markets):,} active markets with CLOB tokens")

    print(f"Screening for live ${THRESHOLD:.2f}+ signals (checking recent price history for persistence) ...")
    hits = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(check_market, m): m for m in markets}
        for i, fut in enumerate(as_completed(futures)):
            try:
                r = fut.result()
            except Exception:
                r = None
            if r:
                hits.append(r)
            if (i + 1) % 500 == 0:
                print(f"  scanned {i+1}/{len(markets)} ...", flush=True)

    print(f"\n{len(hits)} markets currently meet the ${THRESHOLD:.2f}/3-consecutive-snapshot entry criteria "
          f"(pass 1, approximate -- not yet verified)\n")

    # Pass 1 rows: cheap approximate margin, used only to rank/select who's worth
    # the expensive pass-2 verification below.
    pass1 = []
    for h in hits:
        if h["excluded"]:
            continue
        approx = kelly_size(h["report_bucket"], h["current_price"], BANKROLL)
        if approx["margin"] <= 0.03:
            continue
        pass1.append({**h, "_approx_margin": approx["margin"]})
    pass1.sort(key=lambda r: r["_approx_margin"], reverse=True)
    to_verify = pass1[:MAX_VERIFY_CANDIDATES]

    print(f"Verifying {len(to_verify)} candidates against full price history + real order book "
          f"(pass 2 -- this is what actually filters out stale/spike-reversal signals) ...")
    verified = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(verify_candidate, h): h for h in to_verify}
        for fut in as_completed(futures):
            try:
                r = fut.result()
            except Exception:
                r = None
            if r:
                verified.append(r)

    # Collapse same-event duplicates (e.g. every candidate in a multi-outcome
    # election) to the single best-margin leg.
    by_event: dict[str, dict] = {}
    standalone = []
    for r in verified:
        key = r.get("neg_risk_market_id")
        if not key:
            standalone.append(r)
            continue
        if key not in by_event or r["margin"] > by_event[key]["margin"]:
            by_event[key] = r
    deduped = standalone + list(by_event.values())

    allocated = allocate_portfolio(deduped, BANKROLL)

    print(f"\n{len(verified)} passed verification, {len(deduped)} after collapsing same-event duplicates. "
          f"Portfolio-allocated at ${BANKROLL:.2f} bankroll (50% aggregate cap, 25% per-category cap, "
          f"3% per-trade cap, capped further by real live ask depth):\n")
    total_deployed = 0.0
    n_funded = 0
    for r in sorted(allocated, key=lambda r: r["margin"], reverse=True):
        if r["portfolio_position_size"] <= 0:
            continue
        n_funded += 1
        total_deployed += r["portfolio_position_size"]
        below_min = "  [below Polymarket's min order size -- check the UI]" if r["portfolio_position_size"] < 1.0 else ""
        print(f"- {r['question'][:65]!r} [{r['outcome']}] real ask ${r['real_best_ask']:.3f}  "
              f"({r['report_bucket']}, crossed {r['true_crossing_age_hours']:.1f}h ago)\n"
              f"    position: ${r['portfolio_position_size']:.2f}  margin={r['margin']*100:.1f}%  "
              f"spread={r['real_spread']*100:.1f}c  depth=${r['real_ask_depth_notional']:.0f}{below_min}")

    print(f"\n{n_funded} positions funded, ${total_deployed:.2f} of ${BANKROLL:.2f} deployed "
          f"({total_deployed/BANKROLL*100:.1f}% of bankroll)")

    out_path = Path(pmf.RESULTS_DIR) / "live_signal_scan_70.json"
    with open(out_path, "w") as f:
        json.dump(allocated, f, indent=2, default=str)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
