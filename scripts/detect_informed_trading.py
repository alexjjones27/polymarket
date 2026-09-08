"""Can informed (possibly inside) trading be detected before resolution?

The hypothesis: someone holding private information about a political or geopolitical
event buys the low-priced outcome in size, and it later resolves YES. If that leaves a
detectable footprint, the footprint is tradeable -- you do not need the information,
only the ability to see who has it.

This is a different kind of test from everything else in this project. Every previous
idea derived a signal from PRICE, and price is what the market has already agreed on.
This derives it from BEHAVIOUR -- who bought, how much, how early, how concentrated --
which is information the price has not yet absorbed by construction.

Avoiding lookahead is the whole game here, because "large buys at low prices" is
trivially predictive if you let the window include the run-up. A contract that
resolves YES passes through every price on its way to 1.00, so late buys at 0.08 in a
market already trending up are not evidence of anything. The window is therefore
restricted to trades occurring while the price has NEVER YET exceeded PRICE_CEILING --
a running maximum, so nothing after the first sign of a move can leak in.

Per market, measured only inside that clean early window:
  early_buy_vol    total size bought below PRICE_LOW
  max_single       the largest single buy, since one conviction bet is the signature
                   of private information rather than of many people guessing
  n_wallets        distinct buyers
  top_share        the largest wallet's share of that volume; a lone informed buyer
                   looks different from a crowd
Then compares markets that resolved YES against those that resolved NO.
"""
import csv
import json
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import polymarket_final_pct as pmf

PAGES = 200
PER_PAGE = 40
SAMPLE = 2600
MIN_VOLUME = 2000.0
PRICE_LOW = 0.10        # "low-probability" entry
PRICE_CEILING = 0.15    # window closes once price has ever traded above this
OUT = REPO / "data" / "raw" / "polymarket" / "informed_dataset.csv"

CATS = {
    "crypto": ("crypto", "bitcoin", "ethereum", "solana", "btc", "eth", "xrp"),
    "geopolitics": ("war", "israel", "iran", "ukraine", "russia", "ceasefire",
                    "nuclear", "invade", "military", "geopolitics", "venezuela"),
    "politics": ("politics", "election", "trump", "senate", "congress", "president",
                 "cabinet", "governor", "nominee"),
    "sports": ("sports", "nba", "nfl", "mlb", "soccer", "football", "ufc", "tennis"),
}


def categorise(ev):
    blob = (" ".join((t.get("label") or "").lower() for t in (ev.get("tags") or []))
            + " " + (ev.get("title") or "").lower())
    for cat, keys in CATS.items():
        if any(k in blob for k in keys):
            return cat
    return "other"


def fetch_trades(cond):
    out, off = [], 0
    while True:
        try:
            pg = pmf._get(pmf.DATA_API_BASE, "/trades",
                          {"market": cond, "limit": 500, "offset": off})
        except Exception:
            break
        if not pg:
            break
        out.extend(pg)
        if len(pg) < 500:
            break
        off += 500
        if off > 2000:
            break
    return out


def build(job):
    ev_slug, cat, n_out, m = job
    try:
        pr = m.get("outcomePrices")
        pr = json.loads(pr) if isinstance(pr, str) else pr
        if not m.get("closed") or not pr or len(pr) != 2:
            return None
        yes_won = float(pr[0]) == 1.0
        cond = m.get("conditionId")
        if not cond:
            return None
    except Exception:
        return None
    trades = fetch_trades(cond)
    if len(trades) < 15:
        return None

    rows = []
    for t in trades:
        try:
            p = float(t["price"]); s = float(t.get("size") or 0)
            ts = t["timestamp"]
        except (TypeError, ValueError, KeyError):
            continue
        oc = t.get("outcome")
        if oc not in ("Yes", "No"):
            continue
        # express everything as the YES price and whether this BUYS yes-exposure
        yes_px = p if oc == "Yes" else 1.0 - p
        side = t.get("side")
        buys_yes = (oc == "Yes" and side == "BUY") or (oc == "No" and side == "SELL")
        rows.append((ts, yes_px, s, buys_yes, t.get("proxyWallet") or "?"))
    if len(rows) < 15:
        return None
    rows.sort()

    # clean early window: everything before the YES price has EVER exceeded the ceiling
    running_max = 0.0
    early = []
    for ts, px, s, buys, w in rows:
        running_max = max(running_max, px)
        if running_max > PRICE_CEILING:
            break
        early.append((ts, px, s, buys, w))
    if len(early) < 5:
        return None

    lows = [(s, w) for _, px, s, buys, w in early if buys and px < PRICE_LOW and s > 0]
    if not lows:
        return None
    vol = sum(s for s, _ in lows)
    by_w = defaultdict(float)
    for s, w in lows:
        by_w[w] += s
    top = max(by_w.values())
    return {
        "event_slug": ev_slug, "cat": cat, "n_outcomes": n_out,
        "yes": int(yes_won),
        "early_buy_vol": round(vol, 2),
        "max_single": round(max(s for s, _ in lows), 2),
        "n_wallets": len(by_w),
        "top_share": round(top / vol, 4) if vol else 0.0,
        "top_wallet_vol": round(top, 2),
        "n_early_trades": len(early),
    }


def main():
    print("collecting resolved events ...", flush=True)
    evs = []
    for pg in range(PAGES):
        try:
            b = pmf._get(pmf.GAMMA_BASE, "/events",
                         {"closed": "true", "limit": PER_PAGE, "offset": pg * PER_PAGE,
                          "order": "endDate", "ascending": "false"})
        except Exception:
            break
        if not b:
            break
        evs.extend(b)
    print(f"{len(evs)} resolved events")

    jobs = []
    for e in evs:
        ms = [m for m in (e.get("markets") or []) if m.get("closed")]
        cat = categorise(e)
        for m in ms:
            try:
                if float(m.get("volumeNum") or m.get("volume") or 0) < MIN_VOLUME:
                    continue
            except (TypeError, ValueError):
                continue
            jobs.append((e.get("slug"), cat, len(ms), m))
    random.seed(42)
    random.shuffle(jobs)
    jobs = jobs[:SAMPLE]
    print(f"{len(jobs)} markets queued; fetching trades ...", flush=True)

    out = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futs = [pool.submit(build, j) for j in jobs]
        done = 0
        for f in as_completed(futs):
            done += 1
            try:
                r = f.result()
            except Exception:
                continue
            if r:
                out.append(r)
            if done % 400 == 0:
                print(f"  {done}/{len(jobs)} ({len(out)} usable)", flush=True)

    if not out:
        print("nothing collected")
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    yes = sum(r["yes"] for r in out)
    print(f"\nwrote {len(out)} markets -> {OUT.name}  ({yes} resolved YES, "
          f"{len(out)-yes} NO)")


if __name__ == "__main__":
    main()
