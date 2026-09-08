"""Rebalancing-arbitrage scan, restricted to genuinely mutually-exclusive markets.

The first attempt flagged 21 "arbitrages" and every one was an artifact of my own
filter. Almost all had negRisk=False, meaning the outcomes are NOT mutually
exclusive:

  "What price will Bitcoin hit in 2026?"  -- threshold ladder. If BTC hits $200k it
                                             also hit $150k and $100k. Nested.
  "Ceasefire agreement by <date>?"        -- date ladder. Same nesting.

For those, a YES-sum of 4.91 across 32 outcomes is CORRECT. The sum-to-1 constraint
only binds when exactly one outcome can resolve YES, which is what the negRisk flag
marks. Scanning without that filter finds nothing but arithmetic that was never
supposed to hold.

This scans negRisk markets only, and reports:
  sum of best asks  -- buying every YES; below 1.00 (net of fee) is a long arb
  sum of best bids  -- selling every YES; above 1.00 (net of fee) is a short arb
  executable size   -- the thinnest leg, since the basket is only as big as its
                       most illiquid outcome
  fee rate          -- read per market, since it varies (0.07 crypto, 0.04 politics,
                       and some markets carry none at all)

A complete book on every leg is required before anything is called an arbitrage; a
basket with a missing leg cannot be closed.
"""
import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from dotenv import load_dotenv
import polymarket_final_pct as pmf

PAGES = 30
PER_PAGE = 40
ASC = os.environ.get("SCAN_ASC","false")
MIN_EDGE = 0.002


def book_top(client, token):
    try:
        b = client.get_order_book(token)
        bids = sorted(b.get("bids") or [], key=lambda x: float(x["price"]), reverse=True)
        asks = sorted(b.get("asks") or [], key=lambda x: float(x["price"]))
        return {"bid": float(bids[0]["price"]) if bids else None,
                "bid_sz": float(bids[0]["size"]) if bids else 0.0,
                "ask": float(asks[0]["price"]) if asks else None,
                "ask_sz": float(asks[0]["size"]) if asks else 0.0}
    except Exception:
        return None


def scan_event(client, ev):
    all_ms = ev.get("markets") or []
    ms = [m for m in all_ms
          if m.get("enableOrderBook") and not m.get("closed") and m.get("acceptingOrders")]
    if len(ms) < 2:
        return None
    # The basket must cover EVERY outcome in the event. Counting completeness against
    # only the legs we managed to collect made an unclosable basket look complete:
    # U.K. Annual Inflation 2026 has 9 brackets, one failed to return a book, and the
    # remaining 8 summed to 0.7280 -- reported as a +0.24 "arbitrage" that was simply
    # a missing leg worth ~0.27.
    n_event = len(all_ms)
    rate = 0.0
    try:
        rate = float((ms[0].get("feeSchedule") or {}).get("rate", 0.0))
    except Exception:
        pass

    toks = []
    for m in ms:
        t = pmf._safe_json_list(m.get("clobTokenIds"))
        if len(t) == 2:
            toks.append(t[0])
    if len(toks) < 2:
        return None

    books = {}
    with ThreadPoolExecutor(max_workers=20) as pool:
        futs = {pool.submit(book_top, client, t): t for t in toks}
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception:
                r = None
            books[futs[f]] = r

    complete = [books[t] for t in toks if books.get(t)
                and books[t]["ask"] is not None and books[t]["bid"] is not None]
    missing = n_event - len(complete)      # against the EVENT, not against what we gathered
    if not complete:
        return None

    asks = [b["ask"] for b in complete]
    bids = [b["bid"] for b in complete]
    sum_ask, sum_bid = sum(asks), sum(bids)
    fee_buy = rate * sum(p * (1 - p) for p in asks)
    fee_sell = rate * sum(p * (1 - p) for p in bids)
    return {
        "title": ev.get("title", "")[:52], "slug": ev.get("slug", ""),
        "n_total": n_event, "n_book": len(complete), "missing": missing,
        "rate": rate, "vol": ev.get("volume"),
        "sum_ask": sum_ask, "sum_bid": sum_bid,
        "long_edge": 1.0 - sum_ask - fee_buy,
        "short_edge": sum_bid - 1.0 - fee_sell,
        "min_ask_sz": min(b["ask_sz"] for b in complete),
        "min_bid_sz": min(b["bid_sz"] for b in complete),
    }


def main():
    # titles contain unicode (>=, accents) that cp1252 cannot encode
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv(REPO / ".env")
    from py_clob_client_v2 import ClobClient
    client = ClobClient(host="https://clob.polymarket.com", chain_id=137,
                        key=os.environ["POLYMARKET_PRIVATE_KEY"], signature_type=3,
                        funder=os.environ["POLYMARKET_PROXY_ADDRESS"])
    client.set_api_creds(client.create_or_derive_api_key())

    events = []
    for pg in range(PAGES):
        try:
            b = pmf._get(pmf.GAMMA_BASE, "/events",
                         {"closed": "false", "limit": PER_PAGE, "offset": pg * PER_PAGE,
                          "order": "volume", "ascending": ASC})
        except Exception:
            break
        if not b:
            break
        events.extend(b)

    neg = [e for e in events if e.get("negRisk") and len(e.get("markets") or []) >= 2]
    print(f"{len(events)} active events scanned, {len(neg)} are negRisk "
          f"(mutually exclusive)\n")

    out = []
    for i, ev in enumerate(neg, 1):
        r = scan_event(client, ev)
        if r:
            out.append(r)
        if i % 10 == 0:
            print(f"  {i}/{len(neg)} ...", flush=True)

    full = [r for r in out if r["missing"] == 0]
    print(f"\n{len(out)} scanned, {len(full)} with a COMPLETE book on every leg\n")
    print("=" * 116)
    print("NEGRISK MARKETS: sum of YES prices (must be 1.00)")
    print("=" * 116)
    print(f"  {'legs':>5} {'book':>5} {'fee':>6} {'sum_ask':>8} {'sum_bid':>8} "
          f"{'long_edge':>10} {'short_edge':>11} {'min_sz':>7}  title")
    for r in sorted(full, key=lambda x: -max(x["long_edge"], x["short_edge"])):
        best = max(r["long_edge"], r["short_edge"])
        mark = "  <== ARB" if best > MIN_EDGE else ""
        print(f"  {r['n_total']:>5} {r['n_book']:>5} {r['rate']:>6.3f} {r['sum_ask']:>8.4f} "
              f"{r['sum_bid']:>8.4f} {r['long_edge']:>+10.4f} {r['short_edge']:>+11.4f} "
              f"{min(r['min_ask_sz'], r['min_bid_sz']):>7.0f}  {r['title']}{mark}")

    print("\n  markets with an INCOMPLETE book (cannot close the basket):")
    for r in sorted(out, key=lambda x: -x["missing"])[:8]:
        if r["missing"]:
            print(f"    {r['missing']:>4} of {r['n_total']:>4} legs unquoted  {r['title']}")

    arbs = [r for r in full if max(r["long_edge"], r["short_edge"]) > MIN_EDGE]
    print(f"\n{'='*116}")
    if not arbs:
        print("NO rebalancing arbitrage found in any fully-quoted negRisk market.")
        if full:
            sa = [r["sum_ask"] for r in full]
            sb = [r["sum_bid"] for r in full]
            print(f"  sum_ask ranged {min(sa):.4f}-{max(sa):.4f} (all above 1.00 = the spread)")
            print(f"  sum_bid ranged {min(sb):.4f}-{max(sb):.4f} (all below 1.00 = the spread)")
    else:
        for r in arbs:
            side = "BUY all YES" if r["long_edge"] > r["short_edge"] else "SELL all YES"
            edge = max(r["long_edge"], r["short_edge"])
            sz = r["min_ask_sz"] if r["long_edge"] > r["short_edge"] else r["min_bid_sz"]
            print(f"\n  {r['title']}  ({r['n_total']} legs, fee {r['rate']})")
            print(f"    {side}  edge {edge:+.4f}/set, thinnest leg {sz:.0f} -> "
                  f"${edge*sz:,.2f} max")
            print(f"    slug: {r['slug']}")


if __name__ == "__main__":
    main()
