"""Place a single small BUY order on Polymarket's CLOB, for live-testing the
Final-1% / 70%-threshold strategy against real execution.

Safety model:
  - Defaults to DRY RUN: fetches the live book, computes the order, prints
    it, and stops. No credentials are required to be valid for this mode
    beyond loading them (order derivation needs a signature, so creds must
    still be present -- but nothing is submitted).
  - Only submits a real order with --live AND after typing CONFIRM at the
    interactive prompt this script prints (order details, total cost).
  - MAX_ORDER_USD is a hardcoded circuit breaker independent of any Kelly
    sizing math -- this script will refuse to submit an order above it no
    matter what arguments are passed.

Credentials: read from a local .env file (never committed -- see .gitignore)
via POLYMARKET_PRIVATE_KEY and POLYMARKET_PROXY_ADDRESS. Never pass these on
the command line or hardcode them here.
"""
import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import polymarket_final_pct as pmf  # noqa: E402

MAX_ORDER_USD = 10.0  # hard circuit breaker, independent of --size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token-id", required=True, help="CLOB token_id of the outcome to buy")
    ap.add_argument("--usd", type=float, default=2.0, help="target notional in USD (default $2)")
    ap.add_argument("--live", action="store_true", help="submit for real (default: dry run)")
    args = ap.parse_args()

    if args.usd > MAX_ORDER_USD:
        print(f"Refusing: --usd {args.usd} exceeds the hardcoded circuit breaker (${MAX_ORDER_USD}).")
        sys.exit(1)

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    private_key = os.environ.get("POLYMARKET_PRIVATE_KEY")
    proxy_address = os.environ.get("POLYMARKET_PROXY_ADDRESS")
    if not private_key or not proxy_address:
        print("Missing POLYMARKET_PRIVATE_KEY and/or POLYMARKET_PROXY_ADDRESS in .env")
        sys.exit(1)

    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs
    from py_clob_client.order_builder.constants import BUY

    client = ClobClient(
        host="https://clob.polymarket.com",
        chain_id=137,
        key=private_key,
        signature_type=2,  # browser-wallet (Gnosis Safe proxy) account
        funder=proxy_address,
    )
    client.set_api_creds(client.create_or_derive_api_creds())
    print("API credentials derived OK (no funds moved by this step).")

    book = client.get_order_book(args.token_id)
    asks = sorted(book["asks"], key=lambda a: float(a["price"]))
    if not asks:
        print("No live asks on this book -- cannot buy right now.")
        sys.exit(1)
    best_ask = asks[0]
    price = float(best_ask["price"])
    available = float(best_ask["size"])

    size = round(args.usd / price, 2)
    if size > available:
        print(f"Requested size {size} exceeds top-of-book depth {available} at ${price} -- "
              f"would walk the book. Reduce --usd or accept a worse average price (not modeled here).")
        sys.exit(1)

    cost = round(size * price, 4)
    print(f"\nToken:     {args.token_id}")
    print(f"Best ask:  ${price}  (depth {available} shares)")
    print(f"Order:     BUY {size} shares  @ ${price}  = ${cost} total")

    if not args.live:
        print("\nDRY RUN -- nothing submitted. Re-run with --live to actually place this order.")
        return

    print(f"\nThis will submit a REAL order for ${cost} of real funds. This cannot be undone once filled.")
    typed = input("Type CONFIRM to proceed: ")
    if typed != "CONFIRM":
        print("Not confirmed -- aborting, nothing submitted.")
        return

    order_args = OrderArgs(token_id=args.token_id, price=price, size=size, side=BUY)
    signed_order = client.create_order(order_args)
    resp = client.post_order(signed_order)
    print("\nOrder response:")
    print(resp)


if __name__ == "__main__":
    main()
