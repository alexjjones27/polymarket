"""Unit tests for the MM-proxy backtest's pure functions, on synthetic data.
No network calls -- market_pnl, parse_and_sort_trades, and
concentration_by_top_n are all pure functions over already-fetched trade
lists / per-market result dicts.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_mm_proxy_backtest import (
    MAX_MARKET_VOLUME_SHARE,
    MAX_NOTIONAL_PER_TRADE,
    concentration_by_top_n,
    market_pnl,
    parse_and_sort_trades,
)


# ---------------------------------------------------------------------------
# parse_and_sort_trades
# ---------------------------------------------------------------------------

def test_parse_and_sort_trades_filters_and_sorts_chronologically():
    raw = [
        {"price": "0.60", "size": "10", "side": "BUY", "timestamp": 200},
        {"price": "0.50", "size": "5", "side": "SELL", "timestamp": 100},
        {"price": "bad", "size": "10", "side": "BUY", "timestamp": 150},  # unparseable -> dropped
        {"price": "1.0", "size": "10", "side": "BUY", "timestamp": 120},  # price >= 1 -> dropped
        {"price": "0.0", "size": "10", "side": "BUY", "timestamp": 130},  # price <= 0 -> dropped
        {"price": "0.5", "size": "0", "side": "BUY", "timestamp": 140},   # size <= 0 -> dropped
        {"price": "0.5", "size": "10", "side": "HOLD", "timestamp": 145},  # bad side -> dropped
    ]
    valid, total_volume = parse_and_sort_trades(raw)
    assert [t["timestamp"] for t in valid] == [100, 200]
    assert valid[0]["price"] == 0.50 and valid[0]["side"] == "SELL"
    assert total_volume == pytest.approx(0.50 * 5 + 0.60 * 10)


def test_parse_and_sort_trades_empty_input():
    valid, total_volume = parse_and_sort_trades([])
    assert valid == []
    assert total_volume == 0.0


# ---------------------------------------------------------------------------
# market_pnl: best case (zero adverse selection) -- same behavior as before
# ---------------------------------------------------------------------------

def _mk(price, size, side, ts):
    return {"price": price, "size": size, "side": side, "timestamp": ts}


def test_market_pnl_best_case_earns_flat_half_spread():
    trades = [_mk(0.50, 100.0, "BUY", i) for i in range(30)]  # constant price, no drift at all
    total_volume = sum(t["price"] * t["size"] for t in trades)
    r = market_pnl(trades, total_volume, half_spread=0.01, fill_share=0.10)
    # each trade: shares = 100*0.10 = 10 (well under the $25/price notional cap), pnl = 10 * 0.01
    assert r["n_captured"] == 30
    assert r["pnl_best_case"] == pytest.approx(30 * 10 * 0.01)
    # constant price -> no drift -> markout PnL equals best case
    assert r["pnl_with_markout"] == pytest.approx(r["pnl_best_case"], rel=0.05)


def test_market_pnl_respects_per_trade_notional_cap():
    trades = [_mk(0.50, 1_000_000.0, "BUY", 0)]  # huge print, fill_share alone would blow past the cap
    r = market_pnl(trades, 500_000.0, half_spread=0.01, fill_share=0.5)
    # capped at MAX_NOTIONAL_PER_TRADE / price shares, not size * fill_share
    expected_shares = MAX_NOTIONAL_PER_TRADE / 0.50
    assert r["captured_notional"] == pytest.approx(expected_shares * 0.50)


def test_market_pnl_effective_spread_capped_near_price_extremes():
    # at price=0.005, MAX_RELATIVE_SPREAD * price = 0.3 * 0.005 = 0.0015, well under the requested 0.02
    trades = [_mk(0.005, 100.0, "BUY", 0)]
    r = market_pnl(trades, 100.0, half_spread=0.02, fill_share=0.10)
    shares = min(100.0 * 0.10, MAX_NOTIONAL_PER_TRADE / 0.005)
    assert r["pnl_best_case"] == pytest.approx(shares * 0.3 * 0.005)


# ---------------------------------------------------------------------------
# market_pnl: liquidity constraint (MAX_MARKET_VOLUME_SHARE)
# ---------------------------------------------------------------------------

def test_market_pnl_liquidity_cap_trims_captured_notional():
    # 100 trades of notional 1.0 each -> total market volume 100. fill_share=0.5
    # with no cap would capture ~50% of volume; the 20% liquidity cap should
    # trim total captured notional to ~20 regardless.
    trades = [_mk(0.5, 2.0, "BUY", i) for i in range(100)]  # price=0.5, size=2 -> notional 1.0 each
    total_volume = 100 * 1.0
    r = market_pnl(trades, total_volume, half_spread=0.01, fill_share=0.5)
    assert r["captured_notional"] <= MAX_MARKET_VOLUME_SHARE * total_volume + 1e-6
    assert r["volume_share_captured"] <= MAX_MARKET_VOLUME_SHARE + 1e-6


def test_market_pnl_liquidity_cap_does_not_bind_when_fill_share_is_small():
    trades = [_mk(0.5, 2.0, "BUY", i) for i in range(100)]
    total_volume = 100 * 1.0
    r = market_pnl(trades, total_volume, half_spread=0.01, fill_share=0.05)  # well under 20%
    assert r["volume_share_captured"] == pytest.approx(0.05, rel=0.05)


# ---------------------------------------------------------------------------
# market_pnl: markout-based adverse selection
# ---------------------------------------------------------------------------

def test_market_pnl_markout_penalizes_adverse_drift_after_a_buy_fill():
    # We capture a SELL print at t=0 (we're now long); price then drifts DOWN
    # sharply and stays down -- that's adverse selection against our new long.
    trades = [_mk(0.50, 10.0, "SELL", 0)] + [_mk(0.30, 10.0, "SELL", i) for i in range(1, 25)]
    total_volume = sum(t["price"] * t["size"] for t in trades)
    r = market_pnl(trades, total_volume, half_spread=0.01, fill_share=1.0)
    # markout PnL on the first fill should be well below the spread-only best case
    assert r["pnl_with_markout"] < r["pnl_best_case"]


def test_market_pnl_markout_rewards_favorable_drift_after_a_buy_fill():
    # Capture a SELL print (we're long), then price drifts UP -- favorable for
    # a long position, so markout PnL should exceed the flat best-case spread.
    trades = [_mk(0.50, 10.0, "SELL", 0)] + [_mk(0.70, 10.0, "SELL", i) for i in range(1, 25)]
    total_volume = sum(t["price"] * t["size"] for t in trades)
    r = market_pnl(trades, total_volume, half_spread=0.01, fill_share=1.0)
    assert r["pnl_with_markout"] > r["pnl_best_case"]


def test_market_pnl_markout_falls_back_to_spread_only_at_tail_of_tape():
    # A captured fill with no trades after it (end of the tape) has no
    # lookahead window -- must fall back to the spread-only estimate, not
    # crash or silently drop the fill.
    trades = [_mk(0.50, 10.0, "BUY", 0)]
    r = market_pnl(trades, 0.50 * 10.0, half_spread=0.01, fill_share=1.0)
    assert r["n_captured"] == 1
    assert r["pnl_with_markout"] == pytest.approx(r["pnl_best_case"])


# ---------------------------------------------------------------------------
# concentration_by_top_n
# ---------------------------------------------------------------------------

def test_concentration_by_top_n_best_case_ranks_descending():
    rows = [
        {"question": "A", "pnl": 100.0, "n_captured_trades": 1},
        {"question": "B", "pnl": 50.0, "n_captured_trades": 1},
        {"question": "C", "pnl": 10.0, "n_captured_trades": 1},
    ]
    out = concentration_by_top_n(rows, "pnl")
    assert out["total_pnl"] == 160.0
    top1 = next(r for r in out["by_top_n"] if r["n"] == 1)
    assert top1["pnl"] == 100.0
    assert top1["pct_of_total"] == pytest.approx(100 / 160 * 100)
    assert top1["top_markets"][0]["question"] == "A"


def test_concentration_by_top_n_worst_mode_ranks_ascending():
    rows = [
        {"question": "A", "pnl_with_markout": -500.0, "n_captured_trades": 1},
        {"question": "B", "pnl_with_markout": -50.0, "n_captured_trades": 1},
        {"question": "C", "pnl_with_markout": 10.0, "n_captured_trades": 1},
    ]
    out = concentration_by_top_n(rows, "pnl_with_markout", worst=True)
    worst1 = next(r for r in out["by_top_n"] if r["n"] == 1)
    assert worst1["pnl"] == -500.0
    assert worst1["top_markets"][0]["question"] == "A"


def test_concentration_by_top_n_handles_zero_total_pnl():
    rows = [{"question": "A", "pnl": 0.0, "n_captured_trades": 1}]
    out = concentration_by_top_n(rows, "pnl")
    top1 = next(r for r in out["by_top_n"] if r["n"] == 1)
    assert top1["pct_of_total"] is None
