"""Live scanner: NegRisk basket arbitrage, right now, on real order-book depth.

Companion to run_negrisk_arb_full_backtest.py, but reading the CURRENT book
instead of historical snapshots -- the one gap that backtest could not close
(no historical order-book depth exists for a resolved market; this uses
Polymarket's live `/book` endpoint, which works fine for markets that are
still open). For each complete NegRisk event, walks every leg's live ask
ladder and finds the largest number of complete sets (K) buyable RIGHT NOW
such that the total cost across all N legs stays under $1.00/set --
including Polymarket's real per-category taker fee, and requiring every leg
to actually have that much resting ask depth, not a snapshot price assumed
to be achievable at any size.

Read-only. Computes and reports opportunities; places no orders. Top-200
qualifying events by volume (98.4% of this population's total volume,
6,866 legs) rather than the full ~680-event population, so the scan
finishes in a few minutes and the prices it reports are still live when
you read them -- a scan slow enough to take an hour is stale by the time
it finishes, which defeats the point of a *live* scanner.
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import polymarket_final_pct as pmf

REPO = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO / "results" / "polymarket_final_pct"
CACHE_DIR = REPO / "data" / "raw" / "polymarket" / "negrisk_events"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

N_EVENTS = 200
MAX_LEGS = 120
FETCH_WORKERS = 24


def fetch_top_open_negrisk_events(n=N_EVENTS):
    print("[negrisk-live] paging /events?closed=false by volume desc (capped at offset 2000) ...")
    all_events = []
    for offset in range(0, pmf.GAMMA_OFFSET_CAP, 100):
        page = pmf._get(pmf.GAMMA_BASE, "/events", {
            "closed": "false", "order": "volume", "ascending": "false", "limit": 100, "offset": offset,
        })
        if not page:
            break
        all_events.extend(page)

    qualifying = []
    for e in all_events:
        if not e.get("negRisk"):
            continue
        markets = e.get("markets", [])
        if not (3 <= len(markets) <= MAX_LEGS):
            continue
        if not any(m.get("negRiskOther") for m in markets):
            continue
        qualifying.append(e)

    qualifying.sort(key=lambda e: -(e.get("volume") or 0))
    top = qualifying[:n]
    print(f"[negrisk-live] {len(all_events)} open events scanned, {len(qualifying)} complete NegRisk "
          f"baskets qualify, taking top {len(top)} by volume")
    return top


def extract_live_legs(event):
    legs = []
    for m in event.get("markets", []):
        token_ids = pmf._safe_json_list(m.get("clobTokenIds"))
        if not token_ids:
            continue
        fee_schedule = m.get("feeSchedule") or {}
        fee_rate = fee_schedule.get("rate")
        if fee_rate is None:
            fee_rate = 0.04
        legs.append({
            "market_id": m["id"], "question": m.get("question", ""),
            "leg_name": m.get("groupItemTitle", m.get("question", "")),
            "token_id": token_ids[0], "fee_rate": float(fee_rate),
            "is_other": bool(m.get("negRiskOther")),
        })
    return legs


def fetch_live_asks(token_id):
    """Ascending-price ask ladder [(price, size), ...] for one token, live.
    Empty list (not an error) if the book has no resting asks right now."""
    try:
        book = pmf._get(pmf.CLOB_BASE, "/book", {"token_id": token_id})
    except Exception:
        return None
    asks = book.get("asks", []) if isinstance(book, dict) else []
    levels = []
    for lvl in asks:
        try:
            levels.append((float(lvl["price"]), float(lvl["size"])))
        except (KeyError, ValueError, TypeError):
            continue
    levels.sort(key=lambda x: x[0])  # cheapest first, regardless of the API's own ordering
    return levels


def cost_at_k(asks_asc, fee_rate, k):
    """Total $ cost to buy k shares of this leg, walking the live ask ladder,
    fee included per fill level. None if the book can't supply k shares."""
    remaining = k
    cost = 0.0
    for price, size in asks_asc:
        if remaining <= 1e-9:
            break
        take = min(remaining, size)
        cost += take * (price + fee_rate * price * (1 - price))
        remaining -= take
    if remaining > 1e-9:
        return None
    return cost


def max_profitable_k(legs_asks, depth_cap):
    """Largest K (complete sets) such that summed cost across every leg is
    still < K dollars -- i.e. still profitable -- found by bisection on
    [0, depth_cap]. Returns (k, total_cost, profit_usd) or None if not even
    1 share is profitable."""
    def total_cost(k):
        total = 0.0
        for asks_asc, fee_rate in legs_asks:
            c = cost_at_k(asks_asc, fee_rate, k)
            if c is None:
                return None
            total += c
        return total

    c1 = total_cost(1.0)
    if c1 is None or c1 >= 1.0:
        return None  # not even one full set is both fillable and profitable

    lo, hi = 1.0, depth_cap
    c_hi = total_cost(hi)
    if c_hi is not None and c_hi < hi:
        k = hi  # profitable all the way to the depth ceiling
    else:
        for _ in range(40):
            mid = (lo + hi) / 2
            c_mid = total_cost(mid)
            if c_mid is not None and c_mid < mid:
                lo = mid
            else:
                hi = mid
        k = lo
    cost = total_cost(k)
    return {"k_sets": round(k, 2), "total_cost": round(cost, 4), "profit_usd": round(k - cost, 4)}


def scan_event(event):
    legs = extract_live_legs(event)
    if len(legs) < 3:
        return None

    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(fetch_live_asks, l["token_id"]): l for l in legs}
        asks_by_leg = {}
        for fut in as_completed(futures):
            l = futures[fut]
            asks_by_leg[l["token_id"]] = fut.result()

    n_no_book = sum(1 for l in legs if not asks_by_leg.get(l["token_id"]))
    legs_asks = [(asks_by_leg.get(l["token_id"]) or [], l["fee_rate"]) for l in legs]

    top_of_book_sum = 0.0
    top_of_book_ok = True
    for asks_asc, fee_rate in legs_asks:
        if not asks_asc:
            top_of_book_ok = False
            continue
        p = asks_asc[0][0]
        top_of_book_sum += p + fee_rate * p * (1 - p)

    depth_cap = 200.0  # shares; a live scan's practical ceiling per leg, not a claim the book has more
    result = max_profitable_k(legs_asks, depth_cap) if n_no_book == 0 else None

    return {
        "event_id": event["id"], "title": event["title"].strip(), "n_legs": len(legs),
        "n_no_book": n_no_book, "volume": event.get("volume"),
        "top_of_book_sum": round(top_of_book_sum, 4) if top_of_book_ok else None,
        "top_of_book_profit_frac": round(1.0 - top_of_book_sum, 4) if top_of_book_ok else None,
        "executable": result,
    }


def main():
    events = fetch_top_open_negrisk_events(N_EVENTS)
    scanned_at = pmf.pd.Timestamp.utcnow().isoformat()
    print(f"[negrisk-live] scanning {len(events)} events, {sum(len(e['markets']) for e in events)} legs total ...")

    results = []
    t0 = time.time()
    for i, event in enumerate(events):
        r = scan_event(event)
        if r is None:
            continue
        results.append(r)
        tag = ""
        if r["executable"]:
            tag = f"  <<< LIVE ARB: {r['executable']['k_sets']} sets, ${r['executable']['profit_usd']:.2f} profit"
        elif r["top_of_book_profit_frac"] is not None and r["top_of_book_profit_frac"] > 0:
            tag = f"  (top-of-book only, {r['top_of_book_profit_frac']*100:.2f}%, doesn't survive 1 full share)"
        print(f"[negrisk-live] ({i+1}/{len(events)}) {r['title'][:50]:50s} legs={r['n_legs']:3d}{tag}", flush=True)
        if (i + 1) % 25 == 0:
            print(f"  ... {i+1}/{len(events)} done, {time.time()-t0:.0f}s elapsed", flush=True)

    live_hits = sorted([r for r in results if r["executable"]], key=lambda r: -r["executable"]["profit_usd"])
    top_of_book_hits = sorted(
        [r for r in results if not r["executable"] and r["top_of_book_profit_frac"] and r["top_of_book_profit_frac"] > 0],
        key=lambda r: -r["top_of_book_profit_frac"])

    summary = {
        "scanned_at_utc": scanned_at,
        "n_events_scanned": len(results),
        "n_live_arb_hits": len(live_hits),
        "total_live_profit_usd": round(sum(r["executable"]["profit_usd"] for r in live_hits), 2),
        "live_hits": live_hits,
        "top_of_book_only_hits": top_of_book_hits[:20],
        "all_events": results,
    }

    print(f"\n=== {len(live_hits)}/{len(results)} events have a REAL, depth-executable arbitrage right now ===")
    print(f"Total riskless profit available across all of them: ${summary['total_live_profit_usd']:.2f}")
    for r in live_hits[:15]:
        ex = r["executable"]
        print(f"  {r['title'][:45]:45s} {ex['k_sets']:6.1f} sets  ${ex['profit_usd']:7.2f}  "
              f"(cost ${ex['total_cost']:.2f} for ${ex['k_sets']:.2f} redemption)")

    out_path = RESULTS_DIR / "negrisk_arb_live_scan.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
