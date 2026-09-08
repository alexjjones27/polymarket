"""Forward paper-trades the hour-08 4h reversion. No real orders, ever.

The strategy, pre-specified from backtest_hour08_reversion.py so it cannot drift:

  consider only the BTC/ETH 4h window that CLOSES at 08:00 UTC
  look up the immediately preceding 4h window's outcome
  if that one was UP   -> paper-buy DOWN
  if that one was DOWN -> paper-buy UP
  enter early in the window, at the ask, and hold to settlement

Backtested on 183 BTC windows it returned +19.12% per trade against a 52.63%
break-even (p=0.0068), with both chronological halves positive -- the only candidate
in this project to manage that. But hour 08 was chosen by searching six hours, which
takes BTC's p to 0.041 under Bonferroni, and ETH corroborates only weakly since the
two share an outcome 85% of the time. Forward data is the honest test.

The shape of this strategy is the opposite of everything that failed here: one trade
per day, at a known time, on information (the previous window's result) that is
settled four hours in advance. Nothing about it is latency-sensitive, which is what
killed the 5-minute work.

Runs as a slow loop -- it only needs to act once per asset per day -- and records
every decision, including the ones it declines to take and why, so the forward
sample cannot be quietly curated after the fact.
"""
import csv
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from dotenv import load_dotenv
import polymarket_final_pct as pmf

ASSETS = ("btc", "eth")
WINDOW = 14400
TARGET_CLOSE_HOUR = 8          # UTC hour at which the traded window closes
# The backtest entered at close-14000s / -13500s / -12600s, i.e. 400-1800s after
# open, when the price still sits near 0.51 and the reversion is unexpressed. The
# entry window must be held to that range. A first version allowed anywhere from
# open+400s to close-600s and promptly fired at 07:13 UTC on a price of 0.057, with
# the outcome already all but settled -- the same "live rule does not match the
# backtested rule" error that cost $39 on the 5-minute market, caught here for free.
ENTRY_AFTER_OPEN_S = 400       # earliest, matches close-14000s
ENTRY_BEFORE_OPEN_END_S = 1800  # latest, matches close-12600s
STAKE = 5.0
HALF_SPREAD = 0.005
FEE_RATE = 0.07
POLL_S = 120

OUT_DIR = REPO / "results" / "hour08_paper"
LOG = OUT_DIR / "paper_trades.csv"
STATE = OUT_DIR / "state.json"
RUNLOG = OUT_DIR / "run_log.txt"

FIELDS = ["asset", "window_close_utc", "prev_outcome", "bet_side", "entry_time_utc",
          "market_price_up", "entry_ask", "shares", "cost_usd", "fee_usd",
          "resolved_up", "won", "pnl_usd", "settle_time_utc", "note"]


def log(msg):
    line = f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC  {msg}"
    print(line, flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUNLOG, "a") as f:
        f.write(line + "\n")


def load_state():
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except Exception:
            pass
    return {"pending": [], "handled": []}


def save_state(s):
    s["handled"] = s.get("handled", [])[-400:]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, indent=2))


def append(row):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    new = not LOG.exists()
    with open(LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def get_event(asset, close_epoch):
    """Slug epoch names the window's START, as with the 5m series."""
    slug = f"{asset}-updown-4h-{close_epoch - WINDOW}"
    try:
        r = pmf._get(pmf.GAMMA_BASE, "/events", {"slug": slug})
    except Exception:
        return None
    if not r:
        return None
    e = r[0]
    m = e["markets"][0] if e.get("markets") else None
    if not m:
        return None
    return {"market": m, "tokens": pmf._safe_json_list(m.get("clobTokenIds")),
            "closed": m.get("closed"),
            "prices": pmf._safe_json_list(m.get("outcomePrices"))}


def resolved_up(info):
    if not info or not info["closed"] or len(info["prices"]) != 2:
        return None
    try:
        return float(info["prices"][0]) == 1.0
    except (TypeError, ValueError):
        return None


def mid_price_up(client, tokens):
    """Mid of the Up token, or None."""
    if len(tokens) != 2:
        return None
    try:
        bk = client.get_order_book(tokens[0])
        bids = sorted(bk.get("bids") or [], key=lambda x: float(x["price"]), reverse=True)
        asks = sorted(bk.get("asks") or [], key=lambda x: float(x["price"]))
        if not bids or not asks:
            return None
        return (float(bids[0]["price"]) + float(asks[0]["price"])) / 2.0
    except Exception:
        return None


def fee(shares, p):
    return shares * FEE_RATE * p * (1 - p)


def next_target_close(now_epoch):
    """Close time of the hour-08 window we should be acting on.

    The currently-OPEN window is the one we can still enter, and its close is
    floor(now)+WINDOW -- not floor(now). Getting that wrong pointed this at
    yesterday's already-settled window and would have skipped every live entry.
    """
    cur_close = (now_epoch // WINDOW) * WINDOW + WINDOW
    if datetime.fromtimestamp(cur_close, timezone.utc).hour == TARGET_CLOSE_HOUR:
        return cur_close
    for back in range(1, 12):
        cand = cur_close - WINDOW * back
        if datetime.fromtimestamp(cand, timezone.utc).hour == TARGET_CLOSE_HOUR:
            return cand
    return None


def cycle(client, state):
    now = int(time.time())
    close = next_target_close(now)
    if close is None:
        return
    open_s = close - WINDOW
    for asset in ASSETS:
        key = f"{asset}:{close}"
        if key in state["handled"]:
            continue
        # only act inside the narrow entry window the backtest actually used
        if not (open_s + ENTRY_AFTER_OPEN_S <= now <= open_s + ENTRY_BEFORE_OPEN_END_S):
            continue

        prev = get_event(asset, close - WINDOW)
        prev_up = resolved_up(prev)
        if prev_up is None:
            continue  # predecessor not settled yet; try again next poll

        cur = get_event(asset, close)
        if not cur or len(cur["tokens"]) != 2:
            continue
        px_up = mid_price_up(client, cur["tokens"])
        if px_up is None or not (0.05 < px_up < 0.95):
            log(f"{asset} {close}: no usable price ({px_up}) -- declining")
            append({"asset": asset, "window_close_utc": datetime.fromtimestamp(close, timezone.utc).isoformat(),
                    "prev_outcome": "UP" if prev_up else "DOWN", "note": f"declined: price={px_up}"})
            state["handled"].append(key)
            save_state(state)
            continue

        # reversion: bet against the previous window
        if prev_up:
            side, entry = "Down", 1.0 - px_up
        else:
            side, entry = "Up", px_up
        ask = min(0.98, entry + HALF_SPREAD)
        shares = STAKE / ask
        cost = shares * ask
        f = fee(shares, ask)
        rec = {
            "asset": asset, "window_close_utc": datetime.fromtimestamp(close, timezone.utc).isoformat(),
            "prev_outcome": "UP" if prev_up else "DOWN", "bet_side": side,
            "entry_time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "market_price_up": round(px_up, 4), "entry_ask": round(ask, 4),
            "shares": round(shares, 4), "cost_usd": round(cost, 4), "fee_usd": round(f, 4),
            "close_epoch": close,
        }
        state["pending"].append(rec)
        state["handled"].append(key)
        save_state(state)
        log(f"PAPER {asset.upper()} {close}: prev={'UP' if prev_up else 'DOWN'} "
            f"-> buy {side} {shares:.2f}@{ask:.4f} (mid_up={px_up:.4f}) cost ${cost:.2f}")


def settle(state):
    still = []
    for p in state["pending"]:
        info = get_event(p["asset"], p["close_epoch"])
        up = resolved_up(info)
        if up is None:
            still.append(p)
            continue
        won = (up and p["bet_side"] == "Up") or ((not up) and p["bet_side"] == "Down")
        pnl = (p["shares"] - p["cost_usd"] - p["fee_usd"]) if won else -(p["cost_usd"] + p["fee_usd"])
        append({**p, "resolved_up": up, "won": won, "pnl_usd": round(pnl, 4),
                "settle_time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        log(f"SETTLED {p['asset'].upper()} {p['close_epoch']}: {p['bet_side']} "
            f"{'WON' if won else 'LOST'} pnl=${pnl:+.4f}")
    state["pending"] = still
    save_state(state)


def main():
    load_dotenv(REPO / ".env")
    from py_clob_client_v2 import ClobClient
    client = ClobClient(host="https://clob.polymarket.com", chain_id=137,
                        key=os.environ["POLYMARKET_PRIVATE_KEY"], signature_type=3,
                        funder=os.environ["POLYMARKET_PROXY_ADDRESS"])
    client.set_api_creds(client.create_or_derive_api_key())
    state = load_state()
    log(f"hour-08 paper trader started (PAPER ONLY, no real orders). "
        f"{len(state['pending'])} pending.")
    while True:
        try:
            settle(state)
            cycle(client, state)
        except Exception as e:
            log(f"cycle error: {e}")
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
