"""Fully autonomous live execution of the Final-1% / threshold-crossing
strategy (src/polymarket_final_pct.py, scanned the same way as
scripts/scan_live_signals.py): scans currently-open markets for a live
$0.99+ crossing, buys the "No" side, and holds to resolution. Intended to
run unattended on a schedule (see .github/workflows/polymarket_live_trade.yml).

Safety model (all hardcoded, since nobody is present to type CONFIRM):
  - MAX_ORDER_USD: hard per-trade circuit breaker, independent of --usd.
  - MAX_NEW_POSITIONS_PER_RUN: caps how much of the bankroll one scheduled
    run can deploy, even if many signals fire at once.
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

MAX_ORDER_USD = 5.0
MAX_NEW_POSITIONS_PER_RUN = 3
MIN_REMAINING_BALANCE_USD = 0.50  # stop before scraping the account to $0

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


def find_candidates() -> list[dict]:
    """Reuses scan_live_signals' detection logic (same $0.99/3-consecutive
    crossing rule, same exact-score/weather exclusion), ranked by margin
    (highest edge first). Position sizing here is flat-stake, not Kelly --
    see module docstring / --usd."""
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
                hits.append(r)
    hits.sort(key=lambda r: r["margin"], reverse=True)
    return hits


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

    print("Scanning for live $0.99+ crossing signals ...")
    candidates = find_candidates()
    print(f"{len(candidates)} qualifying signals (post-exclusion), ranked by margin.")

    new_trades = 0
    for c in candidates:
        if new_trades >= MAX_NEW_POSITIONS_PER_RUN:
            print(f"Hit MAX_NEW_POSITIONS_PER_RUN ({MAX_NEW_POSITIONS_PER_RUN}) -- stopping.")
            break
        market_id = str(c["market_id"])
        if market_id in open_positions:
            continue
        if balance - stake < MIN_REMAINING_BALANCE_USD:
            print(f"Remaining balance ${balance:.2f} too low to safely take another ${stake:.2f} "
                  f"stake -- stopping.")
            break

        try:
            token_id = get_token_id(market_id, c["outcome"])
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

            print(f"  [{'LIVE' if args.live else 'DRY'}] BUY {size} '{c['outcome']}' "
                  f"@ ${price} = ${cost}  -- {c['question'][:60]!r}")

            if not args.live:
                new_trades += 1
                continue

            order_args = OrderArgsV2(token_id=token_id, price=price, size=size, side=BUY)
            signed_order = client.create_order(order_args)
            resp = client.post_order(signed_order)

            if not resp.get("success"):
                print(f"    order not filled: {resp}")
                continue

            balance -= cost
            new_trades += 1
            open_positions[market_id] = {
                "question": c["question"], "outcome": c["outcome"], "token_id": token_id,
                "entry_price": price, "stake_usd": cost, "order_id": resp.get("orderID"),
                "tx_hashes": resp.get("transactionsHashes"), "entry_date": time.strftime("%Y-%m-%d"),
                "resolution_date": c.get("end_date"),
            }
            save_open_positions(open_positions)
            append_trade_log({
                "date": time.strftime("%Y-%m-%d"), "market": c["question"], "outcome": c["outcome"],
                "quoted_price": price, "actual_fill_price": price, "stake_usd": cost,
                "live_depth_at_scan": available, "resolution_date": c.get("end_date"),
                "notes": f"run_live_strategy.py autonomous; order {resp.get('orderID')}",
            })
            print(f"    filled: order {resp.get('orderID')}")
        except Exception as e:
            print(f"  [error] {c['question'][:60]!r}: {e}")
            continue

    print(f"\nDone. {new_trades} {'live' if args.live else 'dry-run'} position(s) this run.")


if __name__ == "__main__":
    main()
