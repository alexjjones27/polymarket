"""Unit tests for l2_replay_backtest.py's pure functions, on synthetic
order-book touch data. No network calls.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from l2_replay_backtest import l2_market_pnl


def _touch(ts, best_bid, best_ask):
    return {"ts": ts, "best_bid": best_bid, "best_ask": best_ask}


# ---------------------------------------------------------------------------
# l2_market_pnl: quoting, fill detection
# ---------------------------------------------------------------------------

def test_l2_market_pnl_no_fill_when_market_never_crosses_the_quote():
    # mid stays at 0.50 every tick; with half_spread=0.05 our quotes are
    # 0.45/0.55, far outside the tiny 0.49/0.51 real spread -- never crossed.
    touches = [_touch(i, 0.49, 0.51) for i in range(10)]
    r = l2_market_pnl(touches, half_spread=0.05, fill_share=1.0)
    assert r["n_captured"] == 0
    assert r["pnl_best_case"] == pytest.approx(0.0)


def test_l2_market_pnl_captures_a_fill_when_real_market_crosses_our_ask():
    # mid=0.50, half_spread=0.01 -> our_ask=0.51. Next tick's real best_bid
    # jumps to 0.52, crossing our ask -> a genuine, real-market-confirmed sell.
    touches = [_touch(0, 0.49, 0.51), _touch(1, 0.52, 0.54)] + [_touch(i, 0.52, 0.54) for i in range(2, 20)]
    r = l2_market_pnl(touches, half_spread=0.01, fill_share=1.0, order_notional_cap=25.0)
    assert r["n_captured"] == 1
    shares = 25.0 / 0.50
    assert r["pnl_best_case"] == pytest.approx(shares * 0.01)


def test_l2_market_pnl_captures_a_fill_when_real_market_crosses_our_bid():
    touches = [_touch(0, 0.49, 0.51), _touch(1, 0.46, 0.48)] + [_touch(i, 0.46, 0.48) for i in range(2, 20)]
    r = l2_market_pnl(touches, half_spread=0.01, fill_share=1.0, order_notional_cap=25.0)
    assert r["n_captured"] == 1
    shares = 25.0 / 0.50
    assert r["pnl_best_case"] == pytest.approx(shares * 0.01)


def test_l2_market_pnl_skips_ticks_with_an_empty_side():
    # Every tick has an empty ask side -> unquotable, never crossed, never captured.
    touches = [_touch(i, 0.49, None) for i in range(10)]
    r = l2_market_pnl(touches, half_spread=0.01, fill_share=1.0)
    assert r["n_captured"] == 0
    assert r["n_quotable_ticks"] == 0
    assert r["pct_ticks_quotable"] == 0.0


def test_l2_market_pnl_skips_crossed_ticks():
    # best_bid >= best_ask is not a tradeable state -- no mid can be derived.
    touches = [_touch(i, 0.55, 0.50) for i in range(10)]
    r = l2_market_pnl(touches, half_spread=0.01, fill_share=1.0)
    assert r["n_quotable_ticks"] == 0


def test_l2_market_pnl_fill_share_scales_captured_size():
    touches = [_touch(0, 0.49, 0.51), _touch(1, 0.52, 0.54)] + [_touch(i, 0.52, 0.54) for i in range(2, 20)]
    full = l2_market_pnl(touches, half_spread=0.01, fill_share=1.0, order_notional_cap=25.0)
    half = l2_market_pnl(touches, half_spread=0.01, fill_share=0.5, order_notional_cap=25.0)
    assert half["pnl_best_case"] == pytest.approx(full["pnl_best_case"] * 0.5)


# ---------------------------------------------------------------------------
# l2_market_pnl: markout / adverse selection sign convention
# ---------------------------------------------------------------------------

def test_l2_market_pnl_ask_fill_is_adverse_when_price_rises_afterward():
    # We sell at our_ask=0.51 (tick 0->1 crosses it with a jump). Price then
    # keeps climbing for the next 15+ seconds -> markout should show a REAL
    # loss (we're short and it went up), i.e. pnl_with_markout < pnl_best_case.
    touches = [_touch(0, 0.49, 0.51), _touch(1, 0.52, 0.54)]
    touches += [_touch(t, 0.52 + 0.01 * (t - 1), 0.54 + 0.01 * (t - 1)) for t in range(2, 20)]
    r = l2_market_pnl(touches, half_spread=0.01, fill_share=1.0, markout_seconds=15.0)
    assert r["n_captured"] == 1
    assert r["pnl_with_markout"] < r["pnl_best_case"]


def test_l2_market_pnl_ask_fill_is_favorable_when_price_falls_afterward():
    # Symmetric case: we sell, then price DROPS -- good for a short position,
    # markout should show a gain beyond the spread alone.
    touches = [_touch(0, 0.49, 0.51)]
    touches += [_touch(1, 0.52, 0.54)]  # triggers the fill
    touches += [_touch(t, max(0.01, 0.52 - 0.01 * (t - 1)), max(0.02, 0.54 - 0.01 * (t - 1))) for t in range(2, 20)]
    r = l2_market_pnl(touches, half_spread=0.01, fill_share=1.0, markout_seconds=15.0)
    assert r["n_captured"] == 1
    assert r["pnl_with_markout"] > r["pnl_best_case"]


def test_l2_market_pnl_bid_fill_is_adverse_when_price_falls_afterward():
    # We buy at our_bid=0.49 (tick 0->1 crosses it with a downward jump).
    # Price keeps falling afterward -- bad for a long position.
    touches = [_touch(0, 0.49, 0.51), _touch(1, 0.46, 0.48)]
    touches += [_touch(t, max(0.01, 0.46 - 0.01 * (t - 1)), max(0.02, 0.48 - 0.01 * (t - 1))) for t in range(2, 20)]
    r = l2_market_pnl(touches, half_spread=0.01, fill_share=1.0, markout_seconds=15.0)
    assert r["n_captured"] == 1
    assert r["pnl_with_markout"] < r["pnl_best_case"]


def test_l2_market_pnl_falls_back_to_spread_only_when_no_markout_touch_exists():
    # Fill happens right at the tail of the tape -- no tick 15s later exists.
    touches = [_touch(0, 0.49, 0.51), _touch(1, 0.52, 0.54)]
    r = l2_market_pnl(touches, half_spread=0.01, fill_share=1.0, markout_seconds=15.0)
    assert r["n_captured"] == 1
    assert r["pnl_with_markout"] == pytest.approx(r["pnl_best_case"])


# ---------------------------------------------------------------------------
# l2_market_pnl: causality -- markout uses only same-or-later ticks
# ---------------------------------------------------------------------------

def test_l2_market_pnl_fill_decision_unaffected_by_data_beyond_the_next_tick():
    # Truncating the tape right after the crossing tick must not change
    # whether/how many fills were captured through that point.
    touches = [_touch(0, 0.49, 0.51), _touch(1, 0.52, 0.54)]
    touches_long = touches + [_touch(t, 0.10, 0.90) for t in range(2, 50)]  # wildly different future
    short = l2_market_pnl(touches, half_spread=0.01, fill_share=1.0)
    long_ = l2_market_pnl(touches_long, half_spread=0.01, fill_share=1.0)
    assert short["n_captured"] == long_["n_captured"] == 1
