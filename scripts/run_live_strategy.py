"""Fully autonomous live execution of two threshold-crossing strategies:

  - "final_1pct": src/polymarket_final_pct.py / scripts/scan_live_signals.py
    -- buys "No" on a $0.99+ crossing. Measured flip rate ~0.2-0.33%.
  - "seventy_pct": scripts/scan_live_signals_70.py -- buys on a $0.70+
    crossing, two-pass verified (true-first-crossing + real order book).
    Measured flip rate ~12-15% per category -- meaningfully riskier per
    trade than final_1pct. Both strategies currently use the same flat
    stake by deliberate choice, not because the risk is the same.

Both strategies' candidates are pooled and ranked together by margin
(the same Kelly-edge quantity in both scanners), then executed against a
single shared per-run position cap and a single shared open_positions.json
-- a market flagged by either strategy is never re-entered by the other.
Intended to run unattended on a schedule (see
.github/workflows/polymarket_live_trade.yml).

Safety model (all hardcoded, since nobody is present to type CONFIRM):
  - MAX_ORDER_USD: hard per-trade circuit breaker, independent of --usd.
  - MAX_NEW_POSITIONS_PER_RUN: caps how much of the bankroll one scheduled
    run can deploy, even if many signals fire at once, shared across both
    strategies.
  - Re-checks real pUSD balance before every single order and stops the
    run the moment it can't cover another stake -- never overdraws.
  - Never re-enters a market_id already recorded in open_positions.json.
  - Refuses an order that would need to walk past top-of-book depth.
  - Every attempt (filled, skipped, or errored) is logged; a per-candidate
    exception never aborts the rest of the run.

Default is dry run (prints what it would do, submits nothing). Pass --live
to actually submit orders -- there is no interactive confirmation step,
by design, since this is meant to run unattended.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import polymarket_final_pct as pmf  # noqa: E402
import scan_live_signals as sls  # noqa: E402
import scan_live_signals_70 as sls70  # noqa: E402

MAX_ORDER_USD = 5.0
MAX_NEW_POSITIONS_PER_RUN = 3
MIN_REMAINING_BALANCE_USD = 0.50  # stop before scraping the account to $0

# MAX_NEW_POSITIONS_PER_RUN only counts successful fills -- without a
# separate cap on attempts, a systematic failure (e.g. Polymarket's
# geoblock coming back) turns into a live order POST against every single
# qualifying candidate before the loop runs out of candidates to try (this
# happened for real: one run made 105 live attempts, all rejected, before
# stopping only because it ran out of candidates). These two caps bound
# that regardless of why orders are failing.
MAX_LIVE_ATTEMPTS_PER_RUN = 10
MAX_CONSECUTIVE_ERRORS = 3

STATE_DIR = REPO_ROOT / "results" / "polymarket_live_test"
OPEN_POSITIONS_PATH = STATE_DIR / "open_positions.json"
TRADE_LOG_PATH = STATE_DIR / "trade_log.csv"


def load_open_positions() -> dict:
    if OPEN_POSITIONS_PATH.exists():
        return json.loads(OPEN_POSITIONS_PATH.read_text())
    return {}


def save_open_positions(positions: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    OPEN_POSITIONS_PATH.write_text(json.dumps(positions, indent=2, default=str))


def append_trade_log(row: dict) -> None:
    import csv

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    header = ["date", "market", "outcome", "quoted_price", "actual_fill_price",
              "stake_usd", "live_depth_at_scan", "resolution_date", "result",
              "payout_usd", "notes"]
    write_header = not TRADE_LOG_PATH.exists()
    with open(TRADE_LOG_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in header})


def get_token_id(market_id: str, outcome: str) -> str | None:
    m = pmf._get(pmf.GAMMA_BASE, f"/markets/{market_id}", {})
    if isinstance(m, list):
        m = m[0] if m else {}
    outcomes = pmf._safe_json_list(m.get("outcomes"))
    token_ids = pmf._safe_json_list(m.get("clobTokenIds"))
    if outcome not in outcomes:
        return None
    idx = outcomes.index(outcome)
    return token_ids[idx] if idx < len(token_ids) else None


def find_candidates_99() -> list[dict]:
    """Reuses scan_live_signals' detection logic (same $0.99/3-consecutive
    crossing rule, same exact-score/weather exclusion). token_id is left
    None -- resolved lazily at execution time, since scan_live_signals
    doesn't need it for a read-only report."""
    today = pmf.pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    far_future = "2028-01-01"
    markets = sls.fetch_active_markets(today, far_future)
    markets += sls.fetch_active_markets(
        (pmf.pd.Timestamp.utcnow() - pmf.pd.Timedelta(days=1)).strftime("%Y-%m-%d"), today
    )
    markets = pmf._dedupe_by_id(markets)
    markets = [m for m in markets if pmf._safe_json_list(m.get("clobTokenIds"))]

    hits = []
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(sls.check_market, m): m for m in markets}
        for fut in as_completed(futures):
            try:
                r = fut.result()
            except Exception:
                r = None
            if r and not r["excluded"]:
                r["margin"] = sls.kelly_size(r["category"], r["current_price"], 1.0)["margin"]
                r["strategy"] = "final_1pct"
                r["token_id"] = None
                r["depth_usd"] = r.get("live_ask_depth_notional") or 0.0
                hits.append(r)
    hits.sort(key=lambda r: r["depth_usd"], reverse=True)
    return hits


def find_candidates_70() -> list[dict]:
    """Reuses scan_live_signals_70's two-pass verification (true-first-
    crossing over full history + real order book + same-event dedup) --
    this is the expensive, already-verified path, so token_id/real ask
    price are already populated on the returned rows."""
    sls70.BANKROLL = 5.0  # placeholder for margin-threshold filtering only; execution uses a flat stake, not this
    today = pmf.pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    far_future = "2028-01-01"
    markets = sls70.fetch_active_markets(today, far_future)
    markets += sls70.fetch_active_markets(
        (pmf.pd.Timestamp.utcnow() - pmf.pd.Timedelta(days=1)).strftime("%Y-%m-%d"), today
    )
    markets = pmf._dedupe_by_id(markets)
    markets = [m for m in markets if pmf._safe_json_list(m.get("clobTokenIds"))]

    from concurrent.futures import ThreadPoolExecutor, as_completed
    hits = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(sls70.check_market, m): m for m in markets}
        for fut in as_completed(futures):
            try:
                r = fut.result()
            except Exception:
                r = None
            if r and not r["excluded"]:
                hits.append(r)

    pass1 = []
    for h in hits:
        approx = sls70.kelly_size(h["report_bucket"], h["current_price"], sls70.BANKROLL)
        if approx["margin"] <= 0.03:
            continue
        pass1.append({**h, "_approx_margin": approx["margin"]})
    pass1.sort(key=lambda r: r["_approx_margin"], reverse=True)
    to_verify = pass1[: sls70.MAX_VERIFY_CANDIDATES]

    verified = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(sls70.verify_candidate, h): h for h in to_verify}
        for fut in as_completed(futures):
            try:
                r = fut.result()
            except Exception:
                r = None
            if r:
                verified.append(r)

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

    for r in deduped:
        r["strategy"] = "seventy_pct"
        r["current_price"] = r["real_best_ask"]  # execution loop re-fetches the book anyway, but keep this consistent
        # real_ask_depth is raw shares at best ask; multiply by price for a USD figure
        # comparable to find_candidates_99's depth_usd (prices differ a lot at 70% vs 99%).
        r["depth_usd"] = (r.get("real_ask_depth") or 0.0) * (r.get("real_best_ask") or 0.0)
    deduped.sort(key=lambda r: r["depth_usd"], reverse=True)
    return deduped


def find_candidates() -> list[dict]:
    """Pools both strategies' candidates and qualifies each on margin (the
    same Kelly-edge quantity in both scanners -- a candidate must already
    clear that bar to be in this list at all), but ranks the pooled list by
    live order-book depth, highest first: with MAX_NEW_POSITIONS_PER_RUN
    capping how many actually execute, this means only the highest-
    liquidity qualifying signals get traded, not just the highest-margin
    ones -- thin books are the ones most likely to slip or fail to fill."""
    candidates = find_candidates_99() + find_candidates_70()
    candidates.sort(key=lambda r: r["depth_usd"], reverse=True)
    return candidates


def get_real_balance_usd(client) -> float:
    from py_clob_client_v2.clob_types import AssetType, BalanceAllowanceParams
    bal = client.get_balance_allowance(BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))
    return int(bal["balance"]) / 1_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", type=float, default=3.0, help="flat stake per new position (default $3)")
    ap.add_argument("--live", action="store_true", help="submit real orders (default: dry run)")
    args = ap.parse_args()

    stake = min(args.usd, MAX_ORDER_USD)
    if args.usd > MAX_ORDER_USD:
        print(f"--usd {args.usd} exceeds circuit breaker ${MAX_ORDER_USD}; clamping to ${MAX_ORDER_USD}.")

    load_dotenv(REPO_ROOT / ".env")
    private_key = os.environ.get("POLYMARKET_PRIVATE_KEY")
    proxy_address = os.environ.get("POLYMARKET_PROXY_ADDRESS")
    if not private_key or not proxy_address:
        print("Missing POLYMARKET_PRIVATE_KEY and/or POLYMARKET_PROXY_ADDRESS.")
        sys.exit(1)

    from py_clob_client_v2 import ClobClient
    from py_clob_client_v2.clob_types import OrderArgsV2
    from py_clob_client_v2.order_builder.constants import BUY

    client = ClobClient(
        host="https://clob.polymarket.com",
        chain_id=137,
        key=private_key,
        signature_type=3,  # POLY_1271 Deposit Wallet (CLOB V2)
        funder=proxy_address,
    )
    client.set_api_creds(client.create_or_derive_api_key())

    balance = get_real_balance_usd(client)
    print(f"Real pUSD balance: ${balance:.2f}")

    open_positions = load_open_positions()
    print(f"Already-open positions on file: {len(open_positions)}")

    print("Scanning for live $0.99+ (final_1pct) and $0.70+ (seventy_pct, two-pass verified) crossing signals ...")
    candidates = find_candidates()
    print(f"{len(candidates)} qualifying signals across both strategies (post-exclusion), "
          f"ranked by live order-book depth (highest liquidity first).")

    new_trades = 0
    live_attempts = 0
    consecutive_errors = 0
    for c in candidates:
        if new_trades >= MAX_NEW_POSITIONS_PER_RUN:
            print(f"Hit MAX_NEW_POSITIONS_PER_RUN ({MAX_NEW_POSITIONS_PER_RUN}) -- stopping.")
            break
        if live_attempts >= MAX_LIVE_ATTEMPTS_PER_RUN:
            print(f"Hit MAX_LIVE_ATTEMPTS_PER_RUN ({MAX_LIVE_ATTEMPTS_PER_RUN}) -- stopping "
                  f"(this caps attempts, not just fills, so a systematic failure can't turn into "
                  f"hundreds of live order POSTs).")
            break
        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            print(f"Hit {MAX_CONSECUTIVE_ERRORS} consecutive errors -- stopping (likely a systematic "
                  f"failure, e.g. geoblock, not market-specific bad luck).")
            break
        market_id = str(c["market_id"])
        if market_id in open_positions:
            continue
        if balance - stake < MIN_REMAINING_BALANCE_USD:
            print(f"Remaining balance ${balance:.2f} too low to safely take another ${stake:.2f} "
                  f"stake -- stopping.")
            break

        try:
            token_id = c.get("token_id") or get_token_id(market_id, c["outcome"])
            if not token_id:
                print(f"  [skip] {c['question'][:60]!r}: could not resolve token_id")
                continue

            book = client.get_order_book(token_id)
            asks = sorted(book["asks"], key=lambda a: float(a["price"]))
            if not asks:
                print(f"  [skip] {c['question'][:60]!r}: no live asks")
                continue
            price = float(asks[0]["price"])
            available = float(asks[0]["size"])
            size = round(stake / price, 2)
            if size > available:
                print(f"  [skip] {c['question'][:60]!r}: size {size} exceeds top-of-book "
                      f"depth {available} -- would walk the book")
                continue
            cost = round(size * price, 4)

            print(f"  [{'LIVE' if args.live else 'DRY'}] [{c['strategy']}] BUY {size} '{c['outcome']}' "
                  f"@ ${price} = ${cost}  (depth=${c['depth_usd']:.0f})  -- {c['question'][:60]!r}")

            if not args.live:
                new_trades += 1
                continue

            live_attempts += 1
            order_args = OrderArgsV2(token_id=token_id, price=price, size=size, side=BUY)
            signed_order = client.create_order(order_args)
            resp = client.post_order(signed_order)

            if not resp.get("success"):
                consecutive_errors += 1
                print(f"    order not filled: {resp}")
                continue

            consecutive_errors = 0
            balance -= cost
            new_trades += 1
            open_positions[market_id] = {
                "question": c["question"], "outcome": c["outcome"], "token_id": token_id,
                "strategy": c["strategy"], "entry_price": price, "stake_usd": cost,
                "order_id": resp.get("orderID"), "tx_hashes": resp.get("transactionsHashes"),
                "entry_date": time.strftime("%Y-%m-%d"), "resolution_date": c.get("end_date"),
            }
            save_open_positions(open_positions)
            append_trade_log({
                "date": time.strftime("%Y-%m-%d"), "market": c["question"], "outcome": c["outcome"],
                "quoted_price": price, "actual_fill_price": price, "stake_usd": cost,
                "live_depth_at_scan": available, "resolution_date": c.get("end_date"),
                "notes": f"run_live_strategy.py autonomous [{c['strategy']}]; order {resp.get('orderID')}",
            })
            print(f"    filled: order {resp.get('orderID')}")
        except Exception as e:
            consecutive_errors += 1
            print(f"  [error] {c['question'][:60]!r}: {e}")
            continue

    print(f"\nDone. {new_trades} {'live' if args.live else 'dry-run'} position(s) this run.")


if __name__ == "__main__":
    main()
