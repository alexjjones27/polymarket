"""Scans for monotonicity violations in nested ladder markets -- riskless if they exist.

This is the mechanically-detectable subset of the paper's Combinatorial Arbitrage.
That definition needs two markets whose outcomes are logically linked, and the paper
inferred those links with an LLM. But a large family of Polymarket events carries the
link in the labels themselves:

  date ladder       "Ceasefire by March" implies "Ceasefire by June"
  threshold ladder  "BTC above $150k"    implies "BTC above $100k"

so P(by March) <= P(by June) and P(above 150k) <= P(above 100k). These are not
statistical regularities, they are entailments -- if the earlier/tighter condition
resolves YES, the later/looser one must also resolve YES. Prices that violate the
ordering are riskless money regardless of any view on the event.

The trade on a violation, using the DATE ladder as the example:
  observe bid(earlier) > ask(later)
  SELL the earlier YES at its bid, BUY the later YES at its ask
  if earlier resolves YES then later does too, so the legs cancel and the spread is
  kept; if earlier resolves NO the short expires worthless while the long may still
  pay. The position cannot lose.

These events were explicitly EXCLUDED from the earlier negRisk scan, which required
mutual exclusivity -- a YES-sum of 4.91 across 32 nested outcomes is correct, not an
arbitrage. Nested ladders are the opposite case: exclusivity is absent by design, and
monotonicity is the constraint that binds instead.

Reports edge net of the real fee on both legs, and the executable size, since the
trade is capped by the thinner of the two.
"""
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from dotenv import load_dotenv
import polymarket_final_pct as pmf

PAGES = 30
PER_PAGE = 40
MIN_EDGE = 0.002

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], 1)}


RANGE_PAT = re.compile(r"\d[\d,\.]*\s*[-–—]\s*\d")   # "950-999", "140–159"


def rung_arrow(label):
    """Up/down marker inside a rung label, or None.

    Polymarket writes threshold rungs as "^ 130,000" / "v 55,000" meaning RISES TO
    and FALLS TO. These are opposite ladders and implication runs opposite ways:
    rising to 130k implies rising to 100k, but FALLING to 66k implies falling to 78k.
    Ignoring the marker made "sell v78,000 at 0.920, buy v66,000 at 0.013" look like
    arbitrage when it is the wrong side of the ladder entirely -- pure directional
    risk. A single event mixes both markers, so they must be separated before any
    ordering is applied.
    """
    if "↑" in label or "↑" in label:
        return "up"
    if "↓" in label or "↓" in label:
        return "down"
    return None


def is_range_bucket(label):
    """'950-999' style buckets are MUTUALLY EXCLUSIVE, not nested, so monotonicity
    says nothing about them. Treating them as a ladder flagged 'How many Tornadoes'
    and 'How many SpaceX launches' as arbitrage when the rungs cannot both be true."""
    return bool(RANGE_PAT.search(label or ""))


def parse_rung(label):
    """Return (kind, value) so rungs can be ordered within one event.

    Dates are matched FIRST and the string is never mutated by a computed year. An
    earlier version stripped the year with s.replace(str(year), "") and, when no year
    was present, year was 0 -- so every "0" character vanished and "September 30"
    became "September 3", sorting it BEFORE "September 15". That silently inverted
    whole ladders and manufactured 65 false arbitrages.
    """
    s = (label or "").strip().lower()
    if not s:
        return None
    for name, idx in MONTHS.items():
        if name in s:
            yr = re.search(r"\b(20\d\d)\b", s)
            year = int(yr.group(1)) if yr else 2026
            rest = s.replace(name, " ")
            if yr:
                rest = rest.replace(yr.group(1), " ")
            day = re.search(r"\b(\d{1,2})\b", rest)
            d = int(day.group(1)) if day else 15
            return ("date", year * 10000 + idx * 100 + min(max(d, 1), 31))
    mm = re.search(r"\$?\s*([\d,]+(?:\.\d+)?)\s*([kmbt])?\b", s)
    if mm:
        try:
            v = float(mm.group(1).replace(",", ""))
            mult = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}.get(mm.group(2) or "", 1.0)
            return ("num", v * mult)
        except ValueError:
            return None
    return None


def ladder_direction(title, rung_kind=None):
    """Which way does implication run along this ladder?

    The RUNGS decide the family, not the title. "How high will the 10-year yield go
    before 2027?" matches a date pattern in its title while its rungs are yield
    levels; treating it as a date ladder flagged "sell 4.8% at 0.975, buy 5.7% at
    0.037" as arbitrage, when that is merely selling a likely event to buy an
    unlikely one.
    """
    t = (title or "").lower()
    if rung_kind == "date":
        if re.search(r"\bby\b|before|deadline|when", t):
            return "date_by"      # later date = looser, implied by every earlier one
        return None
    if rung_kind == "num":
        if re.search(r"below|under|less than|at most|or lower", t):
            return "thresh_below"   # lower bar = tighter, implies every higher one
        if re.search(r"above|over|reach|hit|exceed|at least|higher|how high|how many|top", t):
            return "thresh_above"   # higher bar = tighter, implies every lower one
        return None
    return None


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
    title = ev.get("title") or ""
    # "... - More Markets" events bundle unrelated bet types under one event: it paired
    # "O/U 2.5" (total goals) with "Chicago Fire FC (-1.5)" (a point spread) and called
    # the numbers a ladder. They share no implication whatsoever.
    if re.search(r"more markets", title, re.I):
        return None
    ms = [m for m in (ev.get("markets") or [])
          if m.get("enableOrderBook") and not m.get("closed") and m.get("acceptingOrders")]
    if len(ms) < 3:
        return None
    rate = 0.0
    try:
        rate = float((ms[0].get("feeSchedule") or {}).get("rate", 0.0) or 0.0)
    except Exception:
        pass

    rungs = []
    for m in ms:
        lab = m.get("groupItemTitle") or m.get("question") or ""
        if is_range_bucket(lab):
            return None            # exclusive buckets, not a nested ladder
        pr = parse_rung(lab)
        toks = pmf._safe_json_list(m.get("clobTokenIds"))
        if pr and len(toks) == 2:
            rungs.append({"label": lab, "kind": pr[0], "val": pr[1],
                          "arrow": rung_arrow(lab), "tok": toks[0]})
    if len(rungs) < 3:
        return None
    kinds = {r["kind"] for r in rungs}
    if len(kinds) != 1:
        return None
    # An event mixing ^ and v rungs holds two opposite ladders; keep only the larger
    # one, because implication does not run between them.
    arrows = {r["arrow"] for r in rungs}
    if len(arrows) > 1:
        groups = {}
        for r in rungs:
            groups.setdefault(r["arrow"], []).append(r)
        rungs = max(groups.values(), key=len)
        if len(rungs) < 3:
            return None
    arrow = rungs[0]["arrow"]
    kind = kinds.pop()
    if kind == "num" and arrow == "down":
        direction = "thresh_below"     # falling to a LOWER level is the tighter claim
    elif kind == "num" and arrow == "up":
        direction = "thresh_above"     # rising to a HIGHER level is the tighter claim
    else:
        direction = ladder_direction(title, kind)
    if not direction:
        return None

    books = {}
    with ThreadPoolExecutor(max_workers=16) as pool:
        futs = {pool.submit(book_top, client, r["tok"]): r["tok"] for r in rungs}
        for f in as_completed(futs):
            try:
                books[futs[f]] = f.result()
            except Exception:
                books[futs[f]] = None
    for r in rungs:
        r["book"] = books.get(r["tok"])
    rungs = [r for r in rungs if r["book"] and r["book"]["bid"] is not None
             and r["book"]["ask"] is not None]
    if len(rungs) < 2:
        return None

    # order so that index increases with LOOSER (higher probability) condition
    if direction == "date_by":
        rungs.sort(key=lambda r: r["val"])            # earlier first = tighter first
    elif direction == "thresh_above":
        rungs.sort(key=lambda r: -r["val"])           # higher bar first = tighter first
    else:
        rungs.sort(key=lambda r: r["val"])            # "below X": lower X = tighter

    viols = []
    for i in range(len(rungs)):
        for j in range(i + 1, len(rungs)):
            tight, loose = rungs[i], rungs[j]
            # tight implies loose, so P(tight) <= P(loose).
            # violation: we can SELL tight at its bid ABOVE what we pay to BUY loose.
            sell = tight["book"]["bid"]
            buy = loose["book"]["ask"]
            gross = sell - buy
            fee = rate * (sell * (1 - sell) + buy * (1 - buy))
            net = gross - fee
            if net > MIN_EDGE:
                viols.append({
                    "sell": tight["label"], "sell_px": sell,
                    "buy": loose["label"], "buy_px": buy,
                    "gross": gross, "net": net,
                    "size": min(tight["book"]["bid_sz"], loose["book"]["ask_sz"]),
                })
    if not viols:
        return None
    viols.sort(key=lambda v: -v["net"])
    return {"title": title[:60], "slug": ev.get("slug"), "direction": direction,
            "rate": rate, "n_rungs": len(rungs), "viols": viols}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv(REPO / ".env")
    from py_clob_client_v2 import ClobClient
    client = ClobClient(host="https://clob.polymarket.com", chain_id=137,
                        key=os.environ["POLYMARKET_PRIVATE_KEY"], signature_type=3,
                        funder=os.environ["POLYMARKET_PROXY_ADDRESS"])
    client.set_api_creds(client.create_or_derive_api_key())

    evs = []
    for pg in range(PAGES):
        try:
            b = pmf._get(pmf.GAMMA_BASE, "/events",
                         {"closed": "false", "limit": PER_PAGE, "offset": pg * PER_PAGE,
                          "order": "volume", "ascending": "false"})
        except Exception:
            break
        if not b:
            break
        evs.extend(b)
    cands = [e for e in evs if len(e.get("markets") or []) >= 3]
    print(f"{len(evs)} active events, {len(cands)} look like ladders\n")

    found = []
    for i, ev in enumerate(cands, 1):
        try:
            r = scan_event(client, ev)
        except Exception:
            r = None
        if r:
            found.append(r)
        if i % 15 == 0:
            print(f"  scanned {i}/{len(cands)} ({len(found)} with violations)", flush=True)

    print(f"\n{'='*100}")
    print(f"MONOTONICITY VIOLATIONS FOUND: {len(found)} events")
    print("=" * 100)
    if not found:
        print("  none -- the ladders are internally consistent")
        return
    for r in sorted(found, key=lambda x: -x["viols"][0]["net"]):
        v = r["viols"][0]
        print(f"\n  {r['title']}   [{r['direction']}, fee {r['rate']}, "
              f"{r['n_rungs']} quoted rungs, {len(r['viols'])} violating pairs]")
        print(f"    SELL '{v['sell'][:34]}' at {v['sell_px']:.3f}")
        print(f"    BUY  '{v['buy'][:34]}' at {v['buy_px']:.3f}")
        print(f"    gross {v['gross']:+.4f}  net of fees {v['net']:+.4f}  "
              f"size {v['size']:.0f}  -> ${v['net']*v['size']:,.2f}")
        print(f"    slug: {r['slug']}")


if __name__ == "__main__":
    main()
