"""Live, continuously-running execution of the sustained-crossing BTC
5-minute edge (see backtest_btc_5m_sustained.py): unlike a single fixed
snapshot, this continuously polls each window's order book and buys the
first time a side's best ask has held at/above PRICE_THRESHOLD for a
full SUSTAIN_S seconds -- confirmed on two disjoint 14-day windows with
net-of-fee edges of +1.4% to +3.7% and ~99.7% of windows qualifying
(vastly stronger and more frequent than the original T-30s-only snapshot
this replaces, which topped out at +0.85% net and fired on ~77% of
windows).

Why "sustained": an earlier "buy at first touch" version was tested and
found WORSE than doing nothing extra -- it catches momentary head-fake
spikes that partially reverse before the window actually settles.
Requiring the price to hold for SUSTAIN_S seconds filters those out while
still catching genuine conviction whenever it develops.

Why monitoring spans past the nominal close: manual inspection (prompted
by a user-flagged "why didn't it trade" case) found that many windows are
still a coin-flip right up to their nominal close and only resolve
gradually over the following 1-5 minutes of continued real trading --
this is genuine, tradeable price discovery, not a data artifact, and the
backtest confirms it: monitoring runs from START_MONITORING_S (before
close) through END_MONITORING_S (after close). Since a window is 300s
long and this span is 390s, two consecutive windows' monitoring periods
can overlap -- this script tracks multiple windows concurrently.

Safety model (same caps as before):
  - Order size: max(5 shares, $5 target) -- Polymarket rejects orders below
    5 shares outright (discovered live, see MIN_SHARES), which at the
    0.85-0.95 price range this trades in always dominates a smaller dollar
    target; actual cost runs ~$4.50-4.95. MAX_ORDER_USD=$5.50 is a hard cap.
  - MAX_CONSECUTIVE_ERRORS per (window, side): gives up on a candidate
    after repeated order failures rather than retrying every second for
    the rest of its ~5-minute monitoring window (this happened for real:
    a sizing bug caused ~8+ rejected orders/sec before being caught).
  - At most one trade per window (removed from active tracking once traded).
  - Settles each pending trade once its window resolves and tracks a
    consecutive-loss streak; MAX_CONSECUTIVE_LOSSES straight losses halts
    the entire loop rather than continuing to trade through what's most
    likely a bug or a broken edge.
  - Real balance re-checked before every order.

Kept running via scripts/btc_5m_signal_watchdog.ps1 (Task Scheduler,
every 5 min, starts a new instance only if one isn't already running).
"""
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
import polymarket_final_pct as pmf  # noqa: E402

PRICE_THRESHOLD = 0.94  # raised from 0.90: live results (3/3 losses confirmed at 0.90-0.92) plus the
                         # historical backtest (584 observations: 0.90-0.94 band ~95-97% win rate vs
                         # ~99.6-100% at 0.94+) both show a real, consistent gap at this cutoff. Trades
                         # roughly half as often as 0.90 did, but the skipped band was the only place
                         # losses occurred in both the live sample and the larger historical one.
SUSTAIN_S = 5
START_MONITORING_S = 90   # start watching a window 90s before its nominal close
END_MONITORING_S = 300    # keep watching up to 5 min after close
MIN_SHARES = 5.0          # Polymarket's enforced minimum order size (discovered live: sub-5-share
                           # orders are rejected outright) -- at price>=0.85 this always dominates
                           # a $1 target, which is why a $1 stake was mechanically impossible here.
TARGET_ORDER_USD = 8.00
MAX_ORDER_USD = 8.50       # hard cap, headroom above the ~$7.20-7.92 the target implies at price<=1
MAX_CONSECUTIVE_LOSSES = 3
MAX_CONSECUTIVE_ERRORS = 5  # per candidate (window, side): give up and require a fresh crossing
POLL_INTERVAL_S = 1

STATE_DIR = REPO_ROOT / "results" / "btc_5m_live"
STATE_PATH = STATE_DIR / "state.json"
TRADE_LOG_PATH = STATE_DIR / "trade_log.csv"
LOG_PATH = STATE_DIR / "run_log.txt"


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"consecutive_losses": 0, "pending": [], "traded_windows": []}


def save_state(state: dict) -> None:
    state["traded_windows"] = state.get("traded_windows", [])[-500:]  # keep it bounded
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def append_trade_log(row: dict) -> None:
    import csv
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    header = ["window_end", "side", "ask_price", "size", "cost_usd", "requested_price", "requested_size",
              "held_for_s", "order_id", "tx_hashes", "resolved_won", "trade_time"]
    write_header = not TRADE_LOG_PATH.exists()
    with open(TRADE_LOG_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in header})


def get_window(window_end: int) -> dict | None:
    res = pmf._get(pmf.GAMMA_BASE, "/events", {"slug": f"btc-updown-5m-{window_end}"})
    if not res:
        return None
    m = res[0]["markets"][0]
    tokens = pmf._safe_json_list(m.get("clobTokenIds"))
    return {"window_end": window_end, "tokens": tokens, "closed": m.get("closed"),
            "outcome_prices": pmf._safe_json_list(m.get("outcomePrices"))}


def settle_pending(state: dict) -> None:
    still_pending = []
    for p in state["pending"]:
        info = get_window(p["window_end"])
        if not info or not info["closed"] or len(info["outcome_prices"]) != 2:
            still_pending.append(p)
            continue
        up_won = float(info["outcome_prices"][0]) == 1.0
        won = up_won if p["side"] == "Up" else (not up_won)
        state["consecutive_losses"] = 0 if won else state["consecutive_losses"] + 1
        log(f"SETTLED window {p['window_end']}: {p['side']} {'WON' if won else 'LOST'} "
            f"(consecutive_losses={state['consecutive_losses']})")
        append_trade_log({**p, "resolved_won": won})
    state["pending"] = still_pending


def get_real_balance_usd(client) -> float:
    from py_clob_client_v2.clob_types import AssetType, BalanceAllowanceParams
    bal = client.get_balance_allowance(BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))
    return int(bal["balance"]) / 1_000_000


def candidate_window_ends(epoch_now: int) -> list[int]:
    """Every 5-min window boundary whose monitoring range currently overlaps now."""
    w = ((epoch_now - START_MONITORING_S) // 300) * 300
    out = []
    while w <= epoch_now + START_MONITORING_S:
        if -END_MONITORING_S <= (w - epoch_now) <= START_MONITORING_S:
            out.append(w)
        w += 300
    return out


def poll_window(client, state, window_end: int, tracked: dict, OrderArgsV2, BUY) -> bool:
    """Returns True if a trade was placed (caller drops the window from tracking)."""
    now = time.time()
    failures = tracked.setdefault("failures", {"Up": 0, "Down": 0})
    for side, token in zip(["Up", "Down"], tracked["tokens"]):
        try:
            book = client.get_order_book(token)
            asks = sorted(book.get("asks") or [], key=lambda a: float(a["price"]))
        except Exception as e:
            log(f"window {window_end}: {side} poll error: {e}")
            continue

        if not asks:
            tracked["candidates"][side] = None
            continue
        price = float(asks[0]["price"])
        avail = float(asks[0]["size"])

        if price < PRICE_THRESHOLD:
            tracked["candidates"][side] = None
            failures[side] = 0
            continue

        if tracked["candidates"][side] is None:
            tracked["candidates"][side] = now
            continue

        held_for = now - tracked["candidates"][side]
        if held_for < SUSTAIN_S:
            continue

        size = max(MIN_SHARES, round(TARGET_ORDER_USD / price, 2))
        cost = round(size * price, 4)
        if cost > MAX_ORDER_USD:
            log(f"window {window_end}: {side} sustained {price:.3f} but computed cost ${cost} "
                f"exceeds hard cap ${MAX_ORDER_USD} -- skipping (shouldn't happen at price<=~0.99)")
            continue
        if size > avail:
            log(f"window {window_end}: {side} sustained {price:.3f} for {held_for:.1f}s but depth "
                f"{avail} < needed {size} -- still watching")
            continue

        balance = get_real_balance_usd(client)
        if balance < cost:
            log(f"window {window_end}: balance ${balance:.2f} too low for ${cost} -- skipping")
            continue

        try:
            order_args = OrderArgsV2(token_id=token, price=price, size=size, side=BUY)
            signed_order = client.create_order(order_args)
            resp = client.post_order(signed_order)
        except Exception as e:
            failures[side] += 1
            log(f"window {window_end}: {side} order error ({failures[side]}/{MAX_CONSECUTIVE_ERRORS}): {e}")
            if failures[side] >= MAX_CONSECUTIVE_ERRORS:
                log(f"window {window_end}: {side} giving up after repeated errors -- "
                    f"requires a fresh crossing to retry")
                tracked["candidates"][side] = None
                failures[side] = 0
            continue
        if not resp.get("success"):
            failures[side] += 1
            log(f"window {window_end}: {side} order not filled ({failures[side]}/{MAX_CONSECUTIVE_ERRORS}): {resp}")
            if failures[side] >= MAX_CONSECUTIVE_ERRORS:
                log(f"window {window_end}: {side} giving up after repeated errors -- "
                    f"requires a fresh crossing to retry")
                tracked["candidates"][side] = None
                failures[side] = 0
            continue

        # Real fill can differ from the submitted limit price -- a marketable limit
        # order gets price IMPROVEMENT (a better, not worse, price) whenever better-priced
        # resting liquidity exists at match time. Discovered live: one trade logged at
        # 0.93 (our submitted limit) actually filled at 0.87 per Polymarket's own public
        # trade record. Use the response's real fill amounts, not our pre-submission quote.
        try:
            real_size = float(resp.get("takingAmount", size))
            real_cost = float(resp.get("makingAmount", cost))
            real_price = round(real_cost / real_size, 4) if real_size else price
        except (TypeError, ValueError, ZeroDivisionError):
            real_size, real_cost, real_price = size, cost, price

        log(f"window {window_end}: BUY {side} {real_size} @ {real_price} (sustained {held_for:.1f}s, "
            f"requested {size}@{price}) = ${real_cost:.4f} -- order {resp.get('orderID')}")
        state["pending"].append({
            "window_end": window_end, "side": side, "ask_price": real_price, "size": real_size,
            "cost_usd": round(real_cost, 4), "requested_price": price, "requested_size": size,
            "held_for_s": round(held_for, 1), "order_id": resp.get("orderID"),
            "tx_hashes": resp.get("transactionsHashes"), "trade_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        state["traded_windows"].append(window_end)
        save_state(state)
        return True
    return False


def main():
    load_dotenv(REPO_ROOT / ".env")
    private_key = os.environ.get("POLYMARKET_PRIVATE_KEY")
    proxy_address = os.environ.get("POLYMARKET_PROXY_ADDRESS")
    if not private_key or not proxy_address:
        log("Missing POLYMARKET_PRIVATE_KEY and/or POLYMARKET_PROXY_ADDRESS.")
        sys.exit(1)

    from py_clob_client_v2 import ClobClient
    from py_clob_client_v2.clob_types import OrderArgsV2
    from py_clob_client_v2.order_builder.constants import BUY

    client = ClobClient(host="https://clob.polymarket.com", chain_id=137,
                         key=private_key, signature_type=3, funder=proxy_address)
    client.set_api_creds(client.create_or_derive_api_key())

    state = load_state()
    state.setdefault("traded_windows", [])
    log(f"Starting. threshold={PRICE_THRESHOLD} sustain={SUSTAIN_S}s consecutive_losses="
        f"{state['consecutive_losses']}, {len(state['pending'])} pending settlement(s).")

    active_windows: dict[int, dict] = {}

    while True:
        try:
            run_one_cycle(client, state, active_windows, OrderArgsV2, BUY)
        except Exception as e:
            log(f"top-level error, will retry: {e}")
            time.sleep(5)


def run_one_cycle(client, state, active_windows, OrderArgsV2, BUY) -> None:
    settle_pending(state)
    save_state(state)

    if state["consecutive_losses"] >= MAX_CONSECUTIVE_LOSSES:
        log(f"STOPPING: {state['consecutive_losses']} consecutive losses -- this is "
            f"~impossible if the edge is real. Almost certainly a bug or a broken edge. "
            f"Not trading further until a human reviews {TRADE_LOG_PATH} and {LOG_PATH}.")
        sys.exit(1)

    epoch_now = int(time.time())
    already_traded = set(state.get("traded_windows", []))

    for window_end in candidate_window_ends(epoch_now):
        if window_end in already_traded or window_end in active_windows:
            continue
        info = get_window(window_end)
        if not info or len(info["tokens"]) != 2:
            continue  # will retry next cycle while still in range
        active_windows[window_end] = {"tokens": info["tokens"], "candidates": {"Up": None, "Down": None}}
        log(f"window {window_end}: now tracking (secs_to_close={window_end - epoch_now})")

    for window_end in list(active_windows.keys()):
        if window_end in already_traded:
            del active_windows[window_end]
            continue
        if epoch_now - window_end > END_MONITORING_S:
            log(f"window {window_end}: monitoring period expired, no qualifying sustained crossing")
            del active_windows[window_end]
            continue
        traded = poll_window(client, state, window_end, active_windows[window_end], OrderArgsV2, BUY)
        if traded:
            del active_windows[window_end]

    time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    main()
