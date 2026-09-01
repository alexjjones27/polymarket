"""Live scan: which currently-open Polymarket markets meet the Final-1%
entry criteria right now, and what position size does that imply for a
$1,000 Kelly-sized account.

This is a point-in-time snapshot, not a backtest -- run it fresh each time
you want current signals. Reuses the crossing-detection and category
exclusion logic from src/polymarket_final_pct.py; adds a live-market census
(active=true instead of closed=true) and live order-book depth (available
for open markets, unlike resolved ones -- confirmed live: /book 404s on a
resolved token but returns real depth on an open one).
"""
import json
import math
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import polymarket_final_pct as pmf
from run_kelly_backtest import load as load_trades, flip_counts_by

BANKROLL = 1000.0
MAX_POS_PCT = 0.03
THRESHOLD = 0.99
N_CONSECUTIVE = 3
PRIOR_A, PRIOR_B = 1.0, 300.0

# Walk-forward-final flip counts, read directly from the committed backtest
# output (trades_maker.csv, post-exclusion) at run time -- not a hand-copied
# snapshot, so this can't silently go stale the next time trades_maker.csv
# is regenerated. This is a live decision made today, so using the full
# resolved history (rather than re-walking-forward within this scan) is
# correct, not a lookahead violation.
CATEGORY_FLIPS = flip_counts_by(load_trades("trades_maker.csv"), "category")
# (flips, total resolved trades post-exclusion), by classify_fee_category bucket.


def qh(category: str) -> float:
    k, n = CATEGORY_FLIPS.get(category, (0, 0))
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
    total = sum(float(a["size"]) for a in asks if float(a["price"]) <= price_threshold + 0.005)
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
        if p < 0.97:  # cheap pre-screen; confirm properly below
            continue
        now_s = int(time.time())
        df, source = pmf.fetch_token_lifetime_prices(tok, now_s - 3 * 86400, now_s + 60)
        if df.empty:
            continue
        hit = pmf.detect_crossing(df, threshold=THRESHOLD, n_consecutive=N_CONSECUTIVE)
        if hit is None:
            continue
        # must still be qualifying as of the most recent snapshot (live, not stale)
        last_price = float(df["p"].iloc[-1])
        if last_price < THRESHOLD:
            continue
        category = pmf.classify_fee_category(market)
        bucket = pmf.classify_report_bucket(market)
        question = market.get("question", "")
        excluded = bool(pmf.re.search(r"^Exact Score:", question, pmf.re.I)) or \
                   bool(pmf.re.search(r"highest temperature.*(be between|be \d)", question, pmf.re.I))

        depth = fetch_live_book_depth(tok, THRESHOLD)

        return {
            "market_id": market["id"], "question": question, "outcome": outcomes[idx] if idx < len(outcomes) else None,
            "current_price": last_price, "entry_price_at_crossing": hit["entry_price"],
            "category": category, "report_bucket": bucket, "excluded": excluded,
            "end_date": market.get("endDate"), "volume": market.get("volumeNum"),
            "live_ask_depth_notional": depth,
        }
    return None


AGG_CAP_PCT = 0.50
CAT_CAP_PCT = 0.25


def kelly_size(category: str, price: float, bankroll: float, fraction: float = 0.25) -> dict:
    b = (1.0 - price) / price  # maker fill, $0 fee
    L = 1.0
    q = qh(category)
    p = 1.0 - q
    f_kelly = (p * b - q * L) / (b * L) if b > 0 else 0.0
    desired = max(0.0, f_kelly) * fraction * bankroll
    per_trade_capped = min(desired, MAX_POS_PCT * bankroll)
    return {"flip_belief_q": q, "kelly_fraction_raw": f_kelly, "desired_uncapped": desired,
            "per_trade_capped": per_trade_capped, "margin": p * b - q * L}


def allocate_portfolio(rows: list[dict], bankroll: float) -> list[dict]:
    """Many live hits are correlated legs of the same multi-outcome event
    (e.g. 19 candidates' "No" positions in one election) -- taking every
    per-trade-capped size independently would blow through real portfolio
    risk limits. Greedily allocates capital in order of Kelly margin,
    respecting the SAME aggregate (50%) and per-category (25%) caps the
    historical Kelly backtest enforced, using report_bucket as the category
    key (crypto_price/sports/politics/other) to also diversify across
    distinct events, not just the fee-schedule category."""
    ranked = sorted(rows, key=lambda r: r["margin"], reverse=True)
    agg_used = 0.0
    cat_used: dict[str, float] = {}
    for r in ranked:
        bucket = r["report_bucket"]
        room_agg = AGG_CAP_PCT * bankroll - agg_used
        room_cat = CAT_CAP_PCT * bankroll - cat_used.get(bucket, 0.0)
        stake = max(0.0, min(r["per_trade_capped"], room_agg, room_cat))
        if r.get("live_ask_depth_notional") is not None:
            stake = min(stake, r["live_ask_depth_notional"])
        r["portfolio_position_size"] = round(stake, 2)
        agg_used += stake
        cat_used[bucket] = cat_used.get(bucket, 0.0) + stake
    return ranked


def main():
    print("Fetching currently active (open) Polymarket markets ...")
    today = pmf.pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    far_future = "2028-01-01"
    markets = fetch_active_markets(today, far_future)
    # also grab anything ending "today" that the bucket above might clip at the edge
    markets += fetch_active_markets(
        (pmf.pd.Timestamp.utcnow() - pmf.pd.Timedelta(days=1)).strftime("%Y-%m-%d"), today
    )
    markets = pmf._dedupe_by_id(markets)
    markets = [m for m in markets if pmf._safe_json_list(m.get("clobTokenIds"))]
    print(f"  {len(markets):,} active markets with CLOB tokens")

    print("Screening for live $0.99+ signals (checking recent price history for persistence) ...")
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

    print(f"\n{len(hits)} markets currently meet the $0.990/3-consecutive-snapshot entry criteria\n")

    rows = []
    for h in hits:
        if h["excluded"]:
            continue  # exact-score / narrow weather-range: screened out, per the flip review
        sizing = kelly_size(h["category"], h["current_price"], BANKROLL)
        rows.append({**h, **sizing})

    allocated = allocate_portfolio(rows, BANKROLL)

    print(f"{len(allocated)} tradeable after exact-score/weather exclusion. "
          f"Portfolio-allocated at $1,000 bankroll (50% aggregate cap, 25% per-category cap, "
          f"3% per-trade cap, capped further by live ask depth where available):\n")
    total_deployed = 0.0
    n_funded = 0
    for r in allocated:
        if r["portfolio_position_size"] <= 0:
            continue
        n_funded += 1
        total_deployed += r["portfolio_position_size"]
        depth_note = (f"${r['live_ask_depth_notional']:.0f} live depth"
                      if r["live_ask_depth_notional"] is not None else "no book data")
        print(f"- {r['question'][:65]!r} [{r['outcome']}] @ ${r['current_price']:.4f}  ({r['category']})\n"
              f"    position: ${r['portfolio_position_size']:.2f}  "
              f"(q={r['flip_belief_q']*100:.2f}%, margin={r['margin']*100:.3f}%, {depth_note})")

    print(f"\n{n_funded} positions funded, ${total_deployed:.2f} of $1,000 deployed "
          f"({total_deployed/BANKROLL*100:.1f}% of bankroll)")

    out_path = Path(pmf.RESULTS_DIR) / "live_signal_scan.json"
    with open(out_path, "w") as f:
        json.dump(allocated, f, indent=2, default=str)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
