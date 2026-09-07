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

All monitoring times are measured against close_ts(), NOT the raw window
id -- the slug epoch names a window's start, and conflating the two put
this script's monitoring range 300s earlier than the backtest's for the
first ~80 live trades. See close_ts() for the full story.

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

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import shadow_trader  # noqa: E402  -- paper-trades alternative thresholds, never trades real money

PRICE_THRESHOLD = 0.94  # raised from 0.90: live results (3/3 losses confirmed at 0.90-0.92) plus the
                         # historical backtest (584 observations: 0.90-0.94 band ~95-97% win rate vs
                         # ~99.6-100% at 0.94+) both show a real, consistent gap at this cutoff. Trades
                         # roughly half as often as 0.90 did, but the skipped band was the only place
                         # losses occurred in both the live sample and the larger historical one.
SUSTAIN_S = 5
WINDOW_LEN_S = 300        # a window is 5 minutes long; see close_ts() -- the slug epoch is its START
START_MONITORING_S = 90   # start watching a window 90s before its close
END_MONITORING_S = 300    # keep watching up to 5 min after close
MIN_SHARES = 5.0          # Polymarket's enforced minimum order size (discovered live: sub-5-share
                           # orders are rejected outright) -- at price>=0.85 this always dominates
                           # a $1 target, which is why a $1 stake was mechanically impossible here.
TARGET_ORDER_USD = 5.00
MAX_ORDER_USD = 5.50       # hard cap, headroom above the ~$4.50-4.95 the target implies at price<=1

# --- emergency exit (stop-loss) -------------------------------------------------
# Until this existed the bot bought once and held to settlement with no position
# management, so every reversal cost the full stake. Forensics on all 11 live
# losses found no entry filter that identifies them in advance (timing, conviction,
# pre-entry volatility, volume, side, hour all fail to separate losses from a
# price-matched win sample), so the only remaining lever is exiting after the fact.
# The collapses run 10-43s with heavy trading throughout, so there is time.
#
# 0.70 is the level with the lowest break-even loss rate (2.42%) of those tested:
# it costs $0.068/trade in false stops (measured on 575 historical pre-close
# entries) and saves $2.81 per real loss (measured on the 11 actual live losses,
# average realised loss $5.02 -> $2.21). Live loss rate is 6.1% post-fix and 8.5%
# all-in, both above break-even. See scripts/backtest_stop_loss*.py.
#
# STOP_SUSTAIN_S: the backtest actually favours 0 (first touch) at our loss rate --
# waiting costs more on genuine collapses than it saves on wicks, break-even ~1.6-2.2%.
# But the backtest models trade PRINTS while this reads the book's best BID, and a
# lone low print can be someone hitting a thin bid while the book recovers instantly.
# 2s is the cheapest non-zero hedge against that gap (~$0.02/trade modelled).
STOP_LOSS_LEVEL = 0.70
# 0 = exit on first touch. Was 2s, as a hedge against the book wicking below the
# level and recovering -- the print backtest could not see that case because it
# models trade prints while this reads the bid.
#
# Lowered to 0 on evidence. The sustain backtest already favoured first touch at
# any loss rate above ~1.6-2.2% (waiting costs more on genuine collapses than it
# saves on wicks), and we run at 6%. Window 1788789600 then demonstrated it with
# real money: the stop armed at bid 0.590 and filled at 0.27 after the 2s wait, a
# ~$1.67 loss, and checking the path the bid never dipped below the level and
# recovered -- so the confirmation window bought nothing and only delayed the exit
# through a freefall. Worth roughly +$0.21/hr at our current stop rate.
STOP_SUSTAIN_S = 0         # best bid must stay under STOP_LOSS_LEVEL this long
STOP_MIN_EXIT_PRICE = 0.02  # below this the recovery is not worth the fee; just hold
MAX_STOP_ATTEMPTS = 5      # per position, then give up and hold to settlement

# --- trigger on consensus, not on the ask ---------------------------------------
# This is the fix for the whole backtest-vs-live gap. The backtest that produced the
# +2.72% OOS edge scans TRADE PRINTS. This script used to trigger on the BEST ASK.
# Those are not the same signal, and measuring 133 live trades showed how far apart:
#
#   - we paid 0.9514 on average while the market was printing 0.9269 (+2.46pp),
#     and we paid above the prevailing print on 89.4% of trades
#   - applying the backtest's own print rule to our entry moments, only 13.5% of our
#     trades would have been taken at all; 86.5% existed solely because a resting ask
#     sat above threshold while actual trading was happening well below it
#   - 11 of our 12 losses came from that ask-only group
#
# So the backtest never validated the rule we deployed, and the ask-only entries were
# adverse selection: an ask detached from consensus is exactly the signature of a book
# about to reprice. Window 1788770700 is the clean example -- ask 0.96 held 5.3s, we
# bought, filled at 0.52 because the book had already collapsed, bid was 0.03 a second
# later. No exit rule can help there; the entry itself was the error.
#
# Triggering on the MID (and refusing wide books) restores the consensus signal the
# backtest actually measured: mid >= 0.94 with a tight spread means both sides agree,
# whereas ask >= 0.94 can be one stale order. The spread guard does double duty --
# it filters detached books AND bounds how far above mid we can pay, since we still
# execute against the ask.
TRIGGER_ON_MID = True      # False restores the old ask-based trigger
MAX_SPREAD = 0.05          # skip the window entirely if ask - bid exceeds this
MAX_CONSECUTIVE_LOSSES = 3
MAX_CONSECUTIVE_ERRORS = 5  # per candidate (window, side): give up and require a fresh crossing

# 0.25s, not the 0.1s originally proposed. Measured against the live endpoint
# (scripts/measure_book_update_rate.py, 198 samples over 30s):
#   - median round-trip latency is 155ms, so 0.1s is physically unreachable; the
#     achieved rate at a 0.1s target was 6.6 req/s, not 10
#   - 97.5% of responses were byte-identical to the previous one, and the median
#     gap between genuine book changes was 5.3s
# But the minimum observed gap between changes was 249ms and half of all changes
# arrived within 1s of the previous, so the book does move sub-second when it
# matters -- the 5.3s median is dominated by quiet stretches. 0.25s captures
# essentially every real update without a 10x rate-limit gamble on an endpoint
# whose limits are undocumented. Note this also makes the sustain filter stricter:
# SUSTAIN_S is wall-clock, so 5s is now 20 samples rather than 5, and any one of
# them below threshold resets. That is the intended direction given our losses
# came from entering on prices that were not genuinely held.
POLL_INTERVAL_S = 0.25

STATE_DIR = REPO_ROOT / "results" / "btc_5m_live"
STATE_PATH = STATE_DIR / "state.json"
TRADE_LOG_PATH = STATE_DIR / "trade_log.csv"
LOG_PATH = STATE_DIR / "run_log.txt"
BOOK_DIR = REPO_ROOT / "data" / "raw" / "polymarket" / "book_snapshots"


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
              "held_for_s", "order_id", "tx_hashes", "resolved_won", "trade_time",
              "exited", "exit_price", "exit_proceeds_usd", "realized_pnl_usd", "exit_time",
              "false_stop", "entry_bid", "entry_ask", "entry_mid", "entry_spread"]
    # The header must be migrated, not assumed. Adding columns while an old file
    # exists silently writes wider rows under a narrower header, so csv.DictReader
    # maps by the stale names and every new column reads back as None. That happened:
    # a real stop-out was recorded correctly on disk (exit 0.69, realised -$1.515)
    # but read back as a full -$5.00 loss, overstating losses by $3.48.
    existing = []
    if TRADE_LOG_PATH.exists():
        with open(TRADE_LOG_PATH, newline="") as f:
            existing = list(csv.reader(f))
    if existing and existing[0] != header:
        padded = [r + [""] * (len(header) - len(r)) if len(r) < len(header) else r[:len(header)]
                  for r in existing[1:]]
        with open(TRADE_LOG_PATH, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(padded)
        log(f"trade log header migrated {len(existing[0])} -> {len(header)} columns")
        existing = []

    with open(TRADE_LOG_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not TRADE_LOG_PATH.exists() or not existing:
            if TRADE_LOG_PATH.stat().st_size == 0:
                w.writeheader()
        w.writerow({k: row.get(k, "") for k in header})


_last_book_state: dict[str, tuple] = {}


def record_book_snapshot(window_end: int, side: str, token: str, book: dict) -> None:
    """Append a full-depth book snapshot to the collection log.

    Exists because every strategy question still open needs order-book history we
    do not have: whether the mid trigger is actually the right rule (it was deployed
    on a diagnosis, never a backtest, since historical data is trade prints and the
    entire point is that prints != book), whether MAX_SPREAD is calibrated, and
    whether the resting-bid idea survives queue-position modelling -- the print
    backtest for it counted "a trade printed at <= level" as a fill, which ignores
    that orders already resting at that price fill first.

    Two deliberate cheapnesses:
      - piggybacks on polls the bot already makes, so it costs zero extra requests
        and cannot itself trip a rate limit
      - writes only when the top of book actually changes. Measured duplicate rate
        is 97.5%, so this cuts volume ~40x with no information loss.
    """
    try:
        bids = sorted(book.get("bids") or [], key=lambda b: float(b["price"]), reverse=True)[:10]
        asks = sorted(book.get("asks") or [], key=lambda a: float(a["price"]))[:10]
        if not bids and not asks:
            return
        top = (bids[0]["price"] if bids else None, bids[0]["size"] if bids else None,
               asks[0]["price"] if asks else None, asks[0]["size"] if asks else None)
        if _last_book_state.get(token) == top:
            return
        _last_book_state[token] = top

        BOOK_DIR.mkdir(parents=True, exist_ok=True)
        path = BOOK_DIR / f"{time.strftime('%Y-%m-%d')}.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps({
                "ts": round(time.time(), 3), "window_end": window_end,
                "close_ts": close_ts(window_end), "side": side, "token": token,
                "bids": [[b["price"], b["size"]] for b in bids],
                "asks": [[a["price"], a["size"]] for a in asks],
            }) + "\n")
    except Exception:
        pass  # collection must never be able to interfere with trading


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

        if p.get("exited"):
            # Already sold out via the stop-loss. The realised P&L is the exit
            # proceeds minus cost regardless of how the window then resolved; we
            # still record the resolution so false stops (we sold, it recovered
            # and would have won) are visible in the log rather than hidden.
            realized = round(float(p["exit_proceeds_usd"]) - float(p["cost_usd"]), 4)
            false_stop = bool(won)
            state["consecutive_losses"] += 1  # a stop-out is a realised loss
            log(f"SETTLED window {p['window_end']}: {p['side']} STOPPED OUT at "
                f"{p['exit_price']} -- realised ${realized:+.4f}"
                f"{' [FALSE STOP: would have won]' if false_stop else ''} "
                f"(consecutive_losses={state['consecutive_losses']})")
            append_trade_log({**p, "resolved_won": won, "realized_pnl_usd": realized,
                              "false_stop": false_stop})
            continue

        state["consecutive_losses"] = 0 if won else state["consecutive_losses"] + 1
        realized = round((float(p["size"]) - float(p["cost_usd"])) if won
                         else -float(p["cost_usd"]), 4)
        log(f"SETTLED window {p['window_end']}: {p['side']} {'WON' if won else 'LOST'} "
            f"(consecutive_losses={state['consecutive_losses']})")
        append_trade_log({**p, "resolved_won": won, "exited": False,
                          "realized_pnl_usd": realized})
    state["pending"] = still_pending


def manage_open_positions(client, state: dict, OrderArgsV2, SELL) -> None:
    """Emergency exit: sell a position whose best bid has stayed under
    STOP_LOSS_LEVEL for STOP_SUSTAIN_S seconds.

    Reads the BID, not the ask -- the bid is what we can actually liquidate into,
    so it is the honest measure of the position's value (and slightly conservative,
    since bid < mid, meaning we trigger marginally earlier than a mid-price rule).

    Exit liquidity was verified before this was built rather than assumed: across
    all 11 real live losses, 3,376-18,167 shares traded in the 15s after the stop
    would have fired, against our ~5.3-share position (0.0-0.2% of flow).
    """
    now = time.time()
    dirty = False
    for p in state["pending"]:
        if p.get("exited") or not p.get("token_id"):
            continue
        if p.get("stop_attempts", 0) >= MAX_STOP_ATTEMPTS:
            continue

        try:
            book = client.get_order_book(p["token_id"])
            bids = sorted(book.get("bids") or [], key=lambda b: float(b["price"]), reverse=True)
        except Exception:
            continue  # 404s are routine once a window resolves; settle_pending handles it

        # Post-entry book history is the part the resting-bid and stop-loss questions
        # both need, and nothing else records it.
        record_book_snapshot(p["window_end"], p["side"], p["token_id"], book)

        if not bids:
            continue
        bid = float(bids[0]["price"])

        if bid >= STOP_LOSS_LEVEL:
            if p.get("stop_breach_since") is not None:
                log(f"window {p['window_end']}: {p['side']} recovered to bid {bid:.3f} "
                    f"before stop confirmed -- resetting")
                p["stop_breach_since"] = None
                dirty = True
            continue

        if p.get("stop_breach_since") is None:
            p["stop_breach_since"] = now
            dirty = True
            if STOP_SUSTAIN_S > 0:
                log(f"window {p['window_end']}: {p['side']} bid {bid:.3f} below "
                    f"{STOP_LOSS_LEVEL} -- watching for {STOP_SUSTAIN_S}s")
                continue
            # first-touch mode: fall through and sell on this same pass rather than
            # costing another poll interval

        held = now - p["stop_breach_since"]
        if held < STOP_SUSTAIN_S:
            continue
        if bid < STOP_MIN_EXIT_PRICE:
            continue  # already collapsed; recovery is not worth the fee

        size = float(p["size"])
        avail = sum(float(b["size"]) for b in bids if float(b["price"]) >= bid)
        if avail < size:
            log(f"window {p['window_end']}: {p['side']} stop confirmed at bid {bid:.3f} "
                f"but depth {avail} < {size} -- still trying")
            continue

        try:
            order_args = OrderArgsV2(token_id=p["token_id"], price=bid, size=size, side=SELL)
            resp = client.post_order(client.create_order(order_args))
        except Exception as e:
            p["stop_attempts"] = p.get("stop_attempts", 0) + 1
            log(f"window {p['window_end']}: {p['side']} STOP sell error "
                f"({p['stop_attempts']}/{MAX_STOP_ATTEMPTS}): {e}")
            dirty = True
            continue
        if not resp.get("success"):
            p["stop_attempts"] = p.get("stop_attempts", 0) + 1
            log(f"window {p['window_end']}: {p['side']} STOP sell not filled "
                f"({p['stop_attempts']}/{MAX_STOP_ATTEMPTS}): {resp}")
            dirty = True
            continue

        # As with buys, the real fill can beat the submitted limit -- use the response.
        try:
            sold_size = float(resp.get("makingAmount", size))
            proceeds = float(resp.get("takingAmount", size * bid))
            exit_price = round(proceeds / sold_size, 4) if sold_size else bid
        except (TypeError, ValueError, ZeroDivisionError):
            sold_size, proceeds, exit_price = size, size * bid, bid

        p["exited"] = True
        p["exit_price"] = exit_price
        p["exit_proceeds_usd"] = round(proceeds, 4)
        p["exit_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        dirty = True
        log(f"window {p['window_end']}: STOP OUT {p['side']} sold {sold_size} @ {exit_price} "
            f"= ${proceeds:.4f} (held under {STOP_LOSS_LEVEL} for {held:.1f}s, "
            f"cost was ${p['cost_usd']}) -- order {resp.get('orderID')}")

    if dirty:
        save_state(state)


def get_real_balance_usd(client) -> float:
    from py_clob_client_v2.clob_types import AssetType, BalanceAllowanceParams
    bal = client.get_balance_allowance(BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))
    return int(bal["balance"]) / 1_000_000


def close_ts(window_id: int) -> int:
    """True close time of a window.

    IMPORTANT: the number in the slug (`btc-updown-5m-<N>`) is the window's
    START, not its end -- Gamma's endDate is always N+300, and slug 1788708300
    is titled "11:25AM-11:30AM ET" (1788708300 = 11:25 ET). An earlier version
    of this script took N to be the close and so monitored [N-90, N+300] =
    [close-390, close]: the 90s before the window even opened, plus the whole
    window. The backtest that validated this edge monitors [close-90, close+300].
    Those overlap by only 90s, and since the first sustained crossing wins, live
    almost always fired early in the window -- where the outcome is still
    genuinely uncertain -- instead of in the informed late/post-close regime that
    was actually backtested. Live loss rate was ~9% against ~1% backtested, and
    every loss so far entered early by this measure.

    Compare monitoring times against close_ts(), never against the raw window id.
    """
    return window_id + WINDOW_LEN_S


def candidate_window_ends(epoch_now: int) -> list[int]:
    """Every window id whose monitoring range -- [close-START, close+END] -- covers now.

    Returns window ids (slug epochs); call close_ts() on one to get its close.
    """
    close = ((epoch_now - END_MONITORING_S) // WINDOW_LEN_S) * WINDOW_LEN_S
    out = []
    while close <= epoch_now + START_MONITORING_S:
        if -END_MONITORING_S <= (close - epoch_now) <= START_MONITORING_S:
            out.append(close - WINDOW_LEN_S)
        close += WINDOW_LEN_S
    return out


def poll_window(client, state, window_end: int, tracked: dict, OrderArgsV2, BUY) -> bool:
    """Returns True if a trade was placed (caller drops the window from tracking)."""
    now = time.time()
    failures = tracked.setdefault("failures", {"Up": 0, "Down": 0})
    for side, token in zip(["Up", "Down"], tracked["tokens"]):
        try:
            book = client.get_order_book(token)
            asks = sorted(book.get("asks") or [], key=lambda a: float(a["price"]))
            bids = sorted(book.get("bids") or [], key=lambda b: float(b["price"]), reverse=True)
        except Exception as e:
            log(f"window {window_end}: {side} poll error: {e}")
            continue

        record_book_snapshot(window_end, side, token, book)

        # Shadow configs see exactly the book we just fetched, at zero API cost.
        # Fully isolated: paper trades only, and any failure here is swallowed so it
        # can never affect real execution.
        try:
            shadow_trader.observe(window_end, close_ts(window_end), side, bids, asks)
        except Exception:
            pass

        # Once the live bot has traded this window we keep polling it purely to feed
        # the collector and the shadow configs, but place no further real orders.
        # Without this the window is dropped on trade, and the post-close shadow
        # configs would only ever see the ~17% of windows we did NOT trade -- a
        # biased subsample, and exactly the population their hypothesis is not about.
        if tracked.get("live_done"):
            continue

        # Both sides are now required: without a bid there is no consensus to read,
        # only a lone ask -- which is precisely the case that lost us money.
        if not asks or not bids:
            tracked["candidates"][side] = None
            continue
        price = float(asks[0]["price"])   # what we will actually pay
        avail = float(asks[0]["size"])
        bid = float(bids[0]["price"])
        mid = (price + bid) / 2.0
        spread = price - bid

        signal_price = mid if TRIGGER_ON_MID else price

        if spread > MAX_SPREAD:
            # A book this wide is not expressing a consensus, and the ask is not
            # evidence of one. Reset rather than accumulate sustain time on it.
            if tracked["candidates"][side] is not None:
                log(f"window {window_end}: {side} spread {spread:.3f} > {MAX_SPREAD} "
                    f"(bid {bid:.3f} / ask {price:.3f}) -- resetting, book too wide to trust")
            tracked["candidates"][side] = None
            failures[side] = 0
            continue

        if signal_price < PRICE_THRESHOLD:
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

        log(f"window {window_end}: BUY {side} {real_size} @ {real_price} (sustained {held_for:.1f}s "
            f"on {'mid' if TRIGGER_ON_MID else 'ask'} {signal_price:.3f}, bid {bid:.3f} / ask "
            f"{price:.3f}, spread {spread:.3f}) = ${real_cost:.4f} -- order {resp.get('orderID')}")
        state["pending"].append({
            "window_end": window_end, "side": side, "ask_price": real_price, "size": real_size,
            "cost_usd": round(real_cost, 4), "requested_price": price, "requested_size": size,
            "held_for_s": round(held_for, 1), "order_id": resp.get("orderID"),
            "tx_hashes": resp.get("transactionsHashes"), "trade_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            # token_id is what manage_open_positions polls and sells; without it a
            # position cannot be stop-lossed, only held to settlement.
            "token_id": token, "exited": False, "stop_breach_since": None, "stop_attempts": 0,
            # Book state at entry. Logged because the ask-vs-consensus gap is what
            # broke this strategy once already and was invisible without it.
            "entry_bid": bid, "entry_ask": price, "entry_mid": round(mid, 4),
            "entry_spread": round(spread, 4),
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
    from py_clob_client_v2.order_builder.constants import BUY, SELL

    client = ClobClient(host="https://clob.polymarket.com", chain_id=137,
                         key=private_key, signature_type=3, funder=proxy_address)
    client.set_api_creds(client.create_or_derive_api_key())

    state = load_state()
    state.setdefault("traded_windows", [])
    log(f"Starting. threshold={PRICE_THRESHOLD} sustain={SUSTAIN_S}s "
        f"stop={STOP_LOSS_LEVEL}/{STOP_SUSTAIN_S}s consecutive_losses="
        f"{state['consecutive_losses']}, {len(state['pending'])} pending settlement(s).")

    active_windows: dict[int, dict] = {}

    while True:
        try:
            run_one_cycle(client, state, active_windows, OrderArgsV2, BUY, SELL)
        except Exception as e:
            log(f"top-level error, will retry: {e}")
            time.sleep(5)


def run_one_cycle(client, state, active_windows, OrderArgsV2, BUY, SELL) -> None:
    settle_pending(state)
    # Stop-loss runs right after settlement so resolved windows are already gone --
    # no point trying to sell into a market that has finished.
    manage_open_positions(client, state, OrderArgsV2, SELL)
    save_state(state)

    try:
        shadow_trader.settle(get_window, log)
    except Exception:
        pass

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
        log(f"window {window_end}: now tracking (secs_to_close={close_ts(window_end) - epoch_now})")

    for window_end in list(active_windows.keys()):
        tracked = active_windows[window_end]
        if window_end in already_traded:
            # Traded windows stay tracked (in shadow/collect-only mode) until their
            # monitoring period expires, rather than being dropped here.
            tracked["live_done"] = True
        if epoch_now - close_ts(window_end) > END_MONITORING_S:
            if not tracked.get("live_done"):
                log(f"window {window_end}: monitoring period expired, "
                    f"no qualifying sustained crossing")
            del active_windows[window_end]
            continue
        if poll_window(client, state, window_end, tracked, OrderArgsV2, BUY):
            tracked["live_done"] = True

    time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    main()
