"""Validate Polymarket API credentials from .env without ever printing the
private key or derived API secret/passphrase. Confirms:
  1. The private key is well-formed and can sign an L1 auth request
     (create_or_derive_api_creds succeeding proves this).
  2. The funder/proxy address is correct, by fetching its real USDC
     (COLLATERAL) balance -- a wrong address would show $0 or error even
     with a valid key.
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
private_key = os.environ.get("POLYMARKET_PRIVATE_KEY")
proxy_address = os.environ.get("POLYMARKET_PROXY_ADDRESS")

if not private_key or not proxy_address:
    print("Missing POLYMARKET_PRIVATE_KEY and/or POLYMARKET_PROXY_ADDRESS in .env")
    sys.exit(1)

print(f"Funder/proxy address (from .env): {proxy_address}")
print(f"Private key: present, {len(private_key)} chars (not printed)")

from py_clob_client_v2 import ClobClient
from py_clob_client_v2.clob_types import AssetType, BalanceAllowanceParams

client = ClobClient(
    host="https://clob.polymarket.com",
    chain_id=137,
    key=private_key,
    signature_type=3,  # POLY_1271 Deposit Wallet (CLOB V2, post April 2026 migration)
    funder=proxy_address,
)

try:
    creds = client.create_or_derive_api_key()
    client.set_api_creds(creds)
    print("API credentials derived OK -- private key signs correctly.")
except Exception as e:
    print(f"FAILED to derive API credentials: {e}")
    sys.exit(1)

try:
    bal = client.get_balance_allowance(BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))
    raw = int(bal["balance"])
    pusd = raw / 1_000_000  # pUSD has 6 decimals
    print(f"pUSD balance at this address: ${pusd:,.2f}")
except Exception as e:
    print(f"FAILED to fetch balance: {e}")
    sys.exit(1)

print("\nCredentials look valid.")
