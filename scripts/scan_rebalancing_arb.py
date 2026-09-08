"""Scans live Polymarket multi-outcome markets for Market Rebalancing Arbitrage.

In a negRisk market exactly one outcome resolves YES, so the YES prices must sum to
1.00. If every YES can be BOUGHT for a total under 1.00 the payout is guaranteed to
exceed the cost, and if they can all be SOLD for over 1.00 the reverse holds. Neither
requires a view on anything.

This is a different environment from the crypto up/down markets that everything else
in this project was tested on, and the differences all point the right way:

  crypto   fee rate 0.07,  tick 0.01,  no rebate
  politics fee rate 0.04,  tick 0.001, rebateRate 0.25

Every strategy killed so far died on that 0.07 fee curve and the 1-cent tick floor.

Fee handling matters and is easy to get wrong. The schedule is taker-only at
rate*p*(1-p) per share, so the cost of buying a whole basket is rate * SUM of
p_i*(1-p_i) -- which depends on how concentrated the market is. A market with one
outcome at 0.90 costs far less to sweep than one with fifty outcomes at 0.02, even
though both sum to 1.00. Computed per market rather than assumed.

Reports depth-aware profit: an arbitrage of two cents on three shares is not a
business, so the executable size is the binding constraint and is reported alongside.
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from dotenv import load_dotenv
import polymarket_final_pct as pmf

MIN_OUTCOMES = 3
MAX_EVENTS = 120
MIN_EDGE = 0.005          # ignore anything under half a cent per share-set


def fee_rate_for(market):
    try:
        fs = market.get("feeSchedule") or {}
        return float(fs.get("rate", 0.0)), bool(fs.get("takerOnly", True))
    except Exception:
        return 0.0, True


def book_top(client, token):
    try:
        b = client.get_order_book(token)
        bids = sorted(b.get("bids") or [], key=lambda x: float(x["price"]), reverse=True)
        asks = sorted(b.get("asks") or [], key=lambda x: float(x["price"]))
        return {
            "bid": float(bids[0]["price"]) if bids else None,
            "bid_sz": float(bids[0]["size"]) if bids else 0.0,
            "ask": float(asks[0]["price"]) if asks else None,
            "ask_sz": float(asks[0]["size"]) if asks else 0.0,
        }
    except Exception:
        return None


def analyse_event(client, ev):
    ms = [m for m in (ev.get("markets") or []) if m.get("enableOrderBook")
          and not m.get("closed") and m.get("acceptingOrders")]
    if len(ms) < MIN_OUTCOMES:
        return None
    rate, taker_only = fee_rate_for(ms[0])

    tokens = []
    for m in ms:
        t = pmf._safe_json_list(m.get("clobTokenIds"))
        if len(t) == 2:
            tokens.append((m, t[0]))     # index 0 is the YES token
    if len(tokens) < MIN_OUTCOMES:
        return None

    books = {}
    with ThreadPoolExecutor(max_workers=16) as pool:
        futs = {pool.submit(book_top, client, tok): tok for _, tok in tokens}
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception:
                r = None
            if r:
                books[futs[f]] = r

    asks, bids, ask_szs, bid_szs = [], [], [], []
    for _, tok in tokens:
        b = books.get(tok)
        if not b or b["ask"] is None or b["bid"] is None:
            return None                  # need a complete book to claim an arb
        asks.append(b["ask"]); ask_szs.append(b["ask_sz"])
        bids.append(b["bid"]); bid_szs.append(b["bid_sz"])

    sum_ask, sum_bid = sum(asks), sum(bids)
    fee_buy = rate * sum(p * (1 - p) for p in asks)
    fee_sell = rate * sum(p * (1 - p) for p in bids)
    return {
        "title": ev.get("title", "")[:64], "slug": ev.get("slug", ""),
        "n": len(tokens), "negRisk": ev.get("negRisk"),
        "rate": rate, "volume": ev.get("volume"),
        "sum_ask": sum_ask, "sum_bid": sum_bid,
        "long_edge": 1.0 - sum_ask - fee_buy,      # buy every YES
        "short_edge": sum_bid - 1.0 - fee_sell,    # sell every YES
        "min_ask_sz": min(ask_szs), "min_bid_sz": min(bid_szs),
        "fee_buy": fee_buy, "fee_sell": fee_sell,
    }


def main():
    load_dotenv(REPO / ".env")
    from py_clob_client_v2 import ClobClient
    client = ClobClient(host="https://clob.polymarket.com", chain_id=137,
                        key=os.environ["POLYMARKET_PRIVATE_KEY"], signature_type=3,
                        funder=os.environ["POLYMARKET_PROXY_ADDRESS"])
    client.set_api_creds(client.create_or_derive_api_key())

    events = []
    for off in range(0, MAX_EVENTS, 40):
        try:
            batch = pmf._get(pmf.GAMMA_BASE, "/events",
                             {"closed": "false", "limit": 40, "offset": off,
                              "order": "volume", "ascending": "false"})
        except Exception:
            break
        if not batch:
            break
        events.extend(batch)
    multi = [e for e in events if len(e.get("markets") or []) >= MIN_OUTCOMES]
    print(f"{len(events)} active events, {len(multi)} with >= {MIN_OUTCOMES} outcomes\n")

    results = []
    for i, ev in enumerate(multi, 1):
        r = analyse_event(client, ev)
        if r:
            results.append(r)
        if i % 10 == 0:
            print(f"  scanned {i}/{len(multi)} ...", flush=True)

    print(f"\n{len(results)} markets with complete books\n")
    print("=" * 112)
    print("SUM OF YES PRICES  (must be 1.00 in a negRisk market)")
    print("=" * 112)
    print(f"  {'n':>4} {'negRisk':>8} {'sum_ask':>8} {'sum_bid':>8} {'long_edge':>10} "
          f"{'short_edge':>11} {'min_sz':>7} {'title':<40}")
    results.sort(key=lambda r: -max(r["long_edge"], r["short_edge"]))
    for r in results:
        best = max(r["long_edge"], r["short_edge"])
        mark = "  <== ARB" if best > MIN_EDGE else ""
        print(f"  {r['n']:>4} {str(r['negRisk']):>8} {r['sum_ask']:>8.4f} "
              f"{r['sum_bid']:>8.4f} {r['long_edge']:>+10.4f} {r['short_edge']:>+11.4f} "
              f"{min(r['min_ask_sz'], r['min_bid_sz']):>7.0f} {r['title'][:40]:<40}{mark}")

    arbs = [r for r in results if max(r["long_edge"], r["short_edge"]) > MIN_EDGE]
    print(f"\n{len(arbs)} markets show an edge above {MIN_EDGE:.3f} per share-set")
    for r in arbs[:15]:
        side = "BUY all YES" if r["long_edge"] > r["short_edge"] else "SELL all YES"
        edge = max(r["long_edge"], r["short_edge"])
        sz = r["min_ask_sz"] if r["long_edge"] > r["short_edge"] else r["min_bid_sz"]
        print(f"\n  {r['title']}")
        print(f"    {r['n']} outcomes, negRisk={r['negRisk']}, fee rate {r['rate']}")
        print(f"    sum_ask={r['sum_ask']:.4f}  sum_bid={r['sum_bid']:.4f}")
        print(f"    {side}: {edge:+.4f}/share-set after fees, thinnest leg {sz:.0f} shares")
        print(f"    max theoretical profit at that size: ${edge*sz:,.2f}")
        print(f"    slug: {r['slug']}")


if __name__ == "__main__":
    main()
