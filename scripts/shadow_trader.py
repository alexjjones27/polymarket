"""Paper-trades alternative entry thresholds against the same live order book the
real bot sees, so the 0.70-vs-0.94 question can be settled without risking money.

Why this and not another backtest: the historical data is trade PRINTS, and the
central finding of this project is that prints do not describe execution. The print
sweep says a 0.70 threshold returns +3.7% to +5.3% on stake versus +0.6% to +1.7%
at 0.94 -- but the same print backtest said 0.94 was +$0.99/hr right before we lost
$36 live, and it contradicts the unbiased calibration curve, which puts the
[0.90,0.95) band (where a 0.70 threshold actually fills) at NEGATIVE edge. Only real
book data can break that tie.

Fills are modelled properly rather than assumed:
  - entry is a TAKER buy against the ask ladder, walked level by level, so a size
    that exhausts the best level pays the weighted average across levels. This is
    real slippage, which the print backtest could not see at all.
  - insufficient depth means no trade, exactly as the live bot behaves
  - the true fee, shares * 0.07 * p * (1-p), is charged on entry
  - the same MIN_SHARES floor and dollar target as live

Deliberately NOT modelled: the stop-loss. Shadow positions are held to settlement.
That keeps this a clean measurement of the threshold's raw edge, and keeps the
configs comparable to each other, which is the question being asked. It does mean
these numbers are not directly comparable to live P&L going forward, since live now
exits at 0.70.

Every entry point is wrapped by the caller so that a failure here can never disturb
real trading.
"""
import csv
import json
import time
from pathlib import Path

MIN_SHARES = 5.0
TARGET_ORDER_USD = 5.00
SUSTAIN_S = 5
FEE_RATE = 0.07

# name, threshold, trigger price, whether the live spread guard applies
CONFIGS = [
    {"name": "mid0.70", "threshold": 0.70, "max_spread": 0.05},
    {"name": "mid0.70_wide", "threshold": 0.70, "max_spread": 1.00},
    {"name": "mid0.80", "threshold": 0.80, "max_spread": 0.05},
    {"name": "mid0.94", "threshold": 0.94, "max_spread": 0.05},  # control: mirrors live
]

STATE_DIR = Path(__file__).resolve().parents[1] / "results" / "btc_5m_live"
SHADOW_STATE = STATE_DIR / "shadow_state.json"
SHADOW_LOG = STATE_DIR / "shadow_trades.csv"

_state = None


def _load():
    global _state
    if _state is not None:
        return _state
    if SHADOW_STATE.exists():
        try:
            _state = json.loads(SHADOW_STATE.read_text())
        except Exception:
            _state = {}
    else:
        _state = {}
    _state.setdefault("candidates", {})   # "cfg|window|side" -> first-seen timestamp
    _state.setdefault("pending", [])
    _state.setdefault("traded", {})       # cfg -> [window_end, ...]
    return _state


def _save():
    if _state is None:
        return
    for cfg in list(_state.get("traded", {})):
        _state["traded"][cfg] = _state["traded"][cfg][-400:]
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    SHADOW_STATE.write_text(json.dumps(_state, indent=2))


def fee_for(shares, price):
    return shares * FEE_RATE * price * (1.0 - price)


def walk_ask_ladder(asks, target_shares):
    """Weighted-average fill price for taking target_shares off the ask ladder.

    Returns (filled_shares, avg_price) or (0, None) if the book is too thin.
    This is the slippage the print backtest is blind to: taking size larger than
    the best level pays progressively worse prices.
    """
    remaining = target_shares
    cost = 0.0
    filled = 0.0
    for lvl in asks:
        try:
            px, sz = float(lvl["price"]), float(lvl["size"])
        except (TypeError, ValueError, KeyError):
            continue
        take = min(remaining, sz)
        cost += take * px
        filled += take
        remaining -= take
        if remaining <= 1e-9:
            break
    if remaining > 1e-9:
        return 0.0, None
    return filled, cost / filled


def observe(window_end, close_ts_val, side, bids, asks):
    """Called on every real poll, with the book the live bot just fetched."""
    st = _load()
    if not bids or not asks:
        return
    try:
        ask = float(asks[0]["price"])
        bid = float(bids[0]["price"])
    except (TypeError, ValueError, KeyError):
        return
    mid = (ask + bid) / 2.0
    spread = ask - bid
    now = time.time()

    for cfg in CONFIGS:
        name = cfg["name"]
        if window_end in st["traded"].get(name, []):
            continue
        key = f"{name}|{window_end}|{side}"

        if spread > cfg["max_spread"] or mid < cfg["threshold"]:
            st["candidates"].pop(key, None)
            continue

        if key not in st["candidates"]:
            st["candidates"][key] = now
            continue
        if now - st["candidates"][key] < SUSTAIN_S:
            continue

        target = max(MIN_SHARES, round(TARGET_ORDER_USD / ask, 2))
        shares, avg_px = walk_ask_ladder(asks, target)
        if not shares or avg_px is None or avg_px >= 1.0:
            continue

        cost = shares * avg_px
        fee = fee_for(shares, avg_px)
        st["pending"].append({
            "config": name, "window_end": window_end, "side": side,
            "shares": round(shares, 4), "avg_price": round(avg_px, 4),
            "best_ask": ask, "best_bid": bid, "mid": round(mid, 4),
            "spread": round(spread, 4), "cost_usd": round(cost, 4),
            "fee_usd": round(fee, 4), "slippage": round(avg_px - ask, 4),
            "held_for_s": round(now - st["candidates"][key], 1),
            "secs_to_close": round(close_ts_val - now, 1),
            "trade_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        st["traded"].setdefault(name, []).append(window_end)
        st["candidates"].pop(key, None)
        _save()


def settle(get_window_fn, log_fn=None):
    """Resolve shadow positions whose window has closed."""
    st = _load()
    if not st["pending"]:
        return
    still = []
    changed = False
    for p in st["pending"]:
        info = get_window_fn(p["window_end"])
        if not info or not info.get("closed") or len(info.get("outcome_prices") or []) != 2:
            still.append(p)
            continue
        try:
            up_won = float(info["outcome_prices"][0]) == 1.0
        except (TypeError, ValueError):
            still.append(p)
            continue
        won = up_won if p["side"] == "Up" else (not up_won)
        pnl = (p["shares"] - p["cost_usd"] - p["fee_usd"]) if won else (-p["cost_usd"] - p["fee_usd"])
        row = {**p, "won": won, "pnl_usd": round(pnl, 4),
               "settle_time": time.strftime("%Y-%m-%d %H:%M:%S")}
        _append_log(row)
        changed = True
        if log_fn:
            log_fn(f"SHADOW[{p['config']}] window {p['window_end']} {p['side']} "
                   f"{shares_desc(p)} -> {'WON' if won else 'LOST'} pnl=${pnl:+.4f}")
    st["pending"] = still
    if changed:
        _save()


def shares_desc(p):
    return f"{p['shares']}@{p['avg_price']}"


def _append_log(row):
    header = ["config", "window_end", "side", "shares", "avg_price", "best_ask", "best_bid",
              "mid", "spread", "slippage", "cost_usd", "fee_usd", "held_for_s",
              "secs_to_close", "trade_time", "won", "pnl_usd", "settle_time"]
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not SHADOW_LOG.exists()
    with open(SHADOW_LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in header})
