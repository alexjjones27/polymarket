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
    REBATE_RATE_BY_FEE_CATEGORY,
    assign_pace_buckets,
    assign_quantile_buckets,
    compute_vpin_series,
    concentration_by_top_n,
    filter_trades_excluding_near_resolution,
    market_pace_seconds,
    market_pnl,
    market_pnl_advanced,
    pace_breakdown,
    parse_and_sort_trades,
    quantile_breakdown,
    resolution_epoch_seconds,
    volume_fraction_near_resolution,
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
    # constant price -> no drift -> both markout variants equal best case
    assert r["pnl_with_markout_trades"] == pytest.approx(r["pnl_best_case"], rel=0.05)
    assert r["pnl_with_markout_time"] == pytest.approx(r["pnl_best_case"], rel=0.05)


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
# market_pnl: rebate_upper_bound (Maker Rebates program, confirmed live
# against docs.polymarket.com -- additive-only, off unless fee_category is passed)
# ---------------------------------------------------------------------------

def test_market_pnl_rebate_upper_bound_is_none_by_default():
    trades = [_mk(0.50, 100.0, "BUY", 0)]
    r = market_pnl(trades, 100.0, half_spread=0.01, fill_share=0.10)
    assert r["rebate_upper_bound"] is None


def test_market_pnl_rebate_upper_bound_uses_the_confirmed_fee_and_rebate_formula():
    trades = [_mk(0.50, 100.0, "BUY", 0)]
    r = market_pnl(trades, 100.0, half_spread=0.01, fill_share=0.10, fee_category="crypto")
    # shares = min(100*0.10, 25/0.5) = 10 -> notional = 5.0
    # taker_fee = notional * feeRate(crypto=0.07) * (1-price=0.5) = 5.0*0.07*0.5 = 0.175
    # rebate_upper_bound = taker_fee * rebate_rate(crypto=0.20) = 0.035
    assert r["rebate_upper_bound"] == pytest.approx(0.035)


def test_market_pnl_rebate_zero_for_fee_free_geopolitics_category():
    trades = [_mk(0.50, 100.0, "BUY", 0)]
    r = market_pnl(trades, 100.0, half_spread=0.01, fill_share=0.10, fee_category="geopolitics")
    assert r["rebate_upper_bound"] == pytest.approx(0.0)


def test_rebate_rate_table_matches_confirmed_polymarket_docs():
    # docs.polymarket.com/programs/maker-rebates, fetched live 2026-09
    assert REBATE_RATE_BY_FEE_CATEGORY["crypto"] == pytest.approx(0.20)
    assert REBATE_RATE_BY_FEE_CATEGORY["sports"] == pytest.approx(0.15)
    assert REBATE_RATE_BY_FEE_CATEGORY["politics"] == pytest.approx(0.25)
    assert REBATE_RATE_BY_FEE_CATEGORY["geopolitics"] == pytest.approx(0.0)


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
    # Timestamps are 1 apart so both the trade-count window (20 trades) and
    # the time window (15s) see the same post-drift prices.
    trades = [_mk(0.50, 10.0, "SELL", 0)] + [_mk(0.30, 10.0, "SELL", i) for i in range(1, 25)]
    total_volume = sum(t["price"] * t["size"] for t in trades)
    r = market_pnl(trades, total_volume, half_spread=0.01, fill_share=1.0)
    # markout PnL on the first fill should be well below the spread-only best case
    assert r["pnl_with_markout_trades"] < r["pnl_best_case"]
    assert r["pnl_with_markout_time"] < r["pnl_best_case"]


def test_market_pnl_markout_rewards_favorable_drift_after_a_buy_fill():
    # Capture a SELL print (we're long), then price drifts UP -- favorable for
    # a long position, so markout PnL should exceed the flat best-case spread.
    trades = [_mk(0.50, 10.0, "SELL", 0)] + [_mk(0.70, 10.0, "SELL", i) for i in range(1, 25)]
    total_volume = sum(t["price"] * t["size"] for t in trades)
    r = market_pnl(trades, total_volume, half_spread=0.01, fill_share=1.0)
    assert r["pnl_with_markout_trades"] > r["pnl_best_case"]
    assert r["pnl_with_markout_time"] > r["pnl_best_case"]


def test_market_pnl_markout_falls_back_to_spread_only_at_tail_of_tape():
    # A captured fill with no trades after it (end of the tape) has no
    # lookahead window -- must fall back to the spread-only estimate, not
    # crash or silently drop the fill.
    trades = [_mk(0.50, 10.0, "BUY", 0)]
    r = market_pnl(trades, 0.50 * 10.0, half_spread=0.01, fill_share=1.0)
    assert r["n_captured"] == 1
    assert r["pnl_with_markout_trades"] == pytest.approx(r["pnl_best_case"])
    assert r["pnl_with_markout_time"] == pytest.approx(r["pnl_best_case"])
    assert r["avg_trades_window_span_s"] is None


def test_market_pnl_time_window_markout_ignores_trades_beyond_the_window():
    # First fill at t=0; a big adverse move happens at t=100, far outside the
    # 15s time window (MARKOUT_WINDOW_SECONDS), so the time-window markout
    # must NOT be penalized by it even though the trade-count window (which
    # has nothing else on the tape) would include it.
    trades = [_mk(0.50, 10.0, "SELL", 0), _mk(0.10, 10.0, "SELL", 100)]
    total_volume = sum(t["price"] * t["size"] for t in trades)
    r = market_pnl(trades, total_volume, half_spread=0.01, fill_share=1.0)
    assert r["pnl_with_markout_time"] == pytest.approx(r["pnl_best_case"])
    assert r["pnl_with_markout_trades"] < r["pnl_best_case"]


def test_market_pnl_reports_actual_window_span_not_assumed():
    # Only 2 trades: the first fill's 20-trade window contains just the
    # second trade (8s later), so the measured real-time span for that one
    # sample is 8s, not the assumed trade count of 20. The second fill is at
    # the tail of the tape (no window), so it contributes no sample.
    trades = [_mk(0.50, 10.0, "SELL", 0), _mk(0.50, 10.0, "SELL", 8)]
    total_volume = sum(t["price"] * t["size"] for t in trades)
    r = market_pnl(trades, total_volume, half_spread=0.01, fill_share=1.0)
    assert r["avg_trades_window_span_s"] == pytest.approx(8.0)


def test_market_pnl_markout_window_seconds_is_a_real_parameter():
    # A big adverse move happens 50s after the captured fill. A large fake
    # total_market_volume keeps the liquidity cap from interfering, so this
    # isolates the effect of the markout_window_seconds parameter itself.
    trades = [_mk(0.50, 10.0, "SELL", 0), _mk(0.20, 10.0, "SELL", 50)]
    r_narrow = market_pnl(trades, 100_000.0, half_spread=0.01, fill_share=1.0, markout_window_seconds=10)
    r_wide = market_pnl(trades, 100_000.0, half_spread=0.01, fill_share=1.0, markout_window_seconds=100)
    # narrow window: the move at t=50 is outside a 10s window -> no penalty
    assert r_narrow["pnl_with_markout_time"] == pytest.approx(r_narrow["pnl_best_case"])
    # wide window: the same move at t=50 IS inside a 100s window -> penalized
    assert r_wide["pnl_with_markout_time"] < r_wide["pnl_best_case"]


# ---------------------------------------------------------------------------
# market_pace_seconds / assign_pace_buckets / pace_breakdown
# ---------------------------------------------------------------------------

def test_market_pace_seconds_computes_median_gap():
    trades = [_mk(0.5, 1.0, "BUY", 0), _mk(0.5, 1.0, "BUY", 10),
              _mk(0.5, 1.0, "BUY", 30), _mk(0.5, 1.0, "BUY", 70)]
    # gaps: 10, 20, 40 -> median 20
    assert market_pace_seconds(trades) == pytest.approx(20.0)


def test_market_pace_seconds_none_below_two_trades():
    assert market_pace_seconds([]) is None
    assert market_pace_seconds([_mk(0.5, 1.0, "BUY", 0)]) is None


def _pace_row(cid, pace, pnl, pnl_trades, pnl_time, bucket="sports"):
    return {
        "condition_id": cid, "median_inter_trade_s": pace,
        "pnl": pnl, "pnl_with_markout_trades": pnl_trades, "pnl_with_markout_time": pnl_time,
        "captured_notional": 100.0, "report_bucket": bucket,
    }


def test_assign_pace_buckets_splits_into_equal_count_quantiles():
    rows = [_pace_row(c, p, 10.0, 5.0, 1.0) for c, p in zip("abcde", [1.0, 2.0, 3.0, 4.0, 5.0])]
    assign_pace_buckets(rows)
    assert [r["pace_bucket"] for r in rows] == ["Q1", "Q2", "Q3", "Q4", "Q5"]


def test_assign_pace_buckets_marks_unmeasurable_pace_as_na():
    rows = [_pace_row("a", None, 10.0, 5.0, 1.0)] + [_pace_row(c, p, 10.0, 5.0, 1.0) for c, p in zip("bcde", [1.0, 2.0, 3.0, 4.0])]
    assign_pace_buckets(rows)
    assert rows[0]["pace_bucket"] == "n/a"
    assert "Q1" in [r["pace_bucket"] for r in rows[1:]]


def test_pace_breakdown_ranks_best_bucket_by_markout_time_not_by_pace():
    # Q5 (slowest) is deliberately given the best markout PnL here -- checks
    # that pace_breakdown ranks by actual profitability, not by assuming
    # "slowest always wins".
    rows = [_pace_row(c, p, 10.0, 5.0, pnl_time) for c, p, pnl_time in
            zip("abcde", [1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 9.0])]
    assign_pace_buckets(rows)
    report = pace_breakdown(rows)
    assert set(report.keys()) == {"Q1", "Q2", "Q3", "Q4", "Q5"}
    assert next(iter(report)) == "Q5"  # highest pnl_with_markout_time (9.0) ranks first
    assert report["Q5"]["n_markets"] == 1
    assert report["Q5"]["median_inter_trade_s_range"] == [5.0, 5.0]


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


# ---------------------------------------------------------------------------
# assign_quantile_buckets / quantile_breakdown -- the generic machinery
# assign_pace_buckets/pace_breakdown are now thin wrappers over
# ---------------------------------------------------------------------------

def test_assign_quantile_buckets_uses_the_given_field_and_prefix():
    rows = [{"volume": v} for v in [50.0, 10.0, 30.0, 40.0, 20.0]]
    assign_quantile_buckets(rows, "volume", "volume_bucket", n_quantiles=5, prefix="V")
    by_volume = sorted(rows, key=lambda r: r["volume"])
    assert [r["volume_bucket"] for r in by_volume] == ["V1", "V2", "V3", "V4", "V5"]


def test_assign_quantile_buckets_none_field_gets_na():
    rows = [{"volume": None}, {"volume": 1.0}, {"volume": 2.0}]
    assign_quantile_buckets(rows, "volume", "volume_bucket", n_quantiles=2, prefix="V")
    assert rows[0]["volume_bucket"] == "n/a"


def test_resolution_epoch_seconds_parses_the_stored_format():
    assert resolution_epoch_seconds("2023-04-21 23:03:21+00:00") == pytest.approx(1682118201.0)


def test_resolution_epoch_seconds_none_on_empty_or_bad_input():
    assert resolution_epoch_seconds("") is None
    assert resolution_epoch_seconds(None) is None
    assert resolution_epoch_seconds("not a date") is None


def test_filter_trades_excluding_near_resolution_drops_only_the_tail():
    trades = [_mk(0.5, 1.0, "BUY", ts) for ts in [0, 100, 200, 300]]
    resolution_epoch = 300.0  # last trade IS the resolution moment
    kept = filter_trades_excluding_near_resolution(trades, resolution_epoch, exclusion_seconds=150)
    # kept: resolution_epoch - ts > 150 -> ts < 150 -> only ts=0 and ts=100
    assert [t["timestamp"] for t in kept] == [0, 100]


def test_filter_trades_excluding_near_resolution_noop_when_unparseable_or_zero_window():
    trades = [_mk(0.5, 1.0, "BUY", ts) for ts in [0, 100, 200]]
    assert filter_trades_excluding_near_resolution(trades, None, 100) == trades
    assert filter_trades_excluding_near_resolution(trades, 300.0, 0) == trades


def test_volume_fraction_near_resolution_computes_the_share():
    # total volume = 4 trades * (0.5*10) = 20.0; the last 2 (ts=200,300) are
    # within 150s of resolution (300) -> 10.0 of 20.0 = 50%.
    trades = [_mk(0.5, 10.0, "BUY", ts) for ts in [0, 100, 200, 300]]
    frac = volume_fraction_near_resolution(trades, resolution_epoch=300.0, window_seconds=150)
    assert frac == pytest.approx(0.5)


def test_volume_fraction_near_resolution_none_when_unparseable():
    trades = [_mk(0.5, 10.0, "BUY", 0)]
    assert volume_fraction_near_resolution(trades, None, 100) is None
    assert volume_fraction_near_resolution([], 300.0, 100) is None


def test_quantile_breakdown_works_on_a_non_pace_field():
    rows = [
        {"volume_bucket": "V1", "total_volume": 10.0, "pnl": 1.0, "pnl_with_markout_trades": 1.0,
         "pnl_with_markout_time": -5.0, "captured_notional": 1.0},
        {"volume_bucket": "V2", "total_volume": 20.0, "pnl": 1.0, "pnl_with_markout_trades": 1.0,
         "pnl_with_markout_time": 5.0, "captured_notional": 1.0},
    ]
    out = quantile_breakdown(rows, "volume_bucket", "total_volume")
    assert set(out.keys()) == {"V1", "V2"}
    assert next(iter(out)) == "V2"  # ranked first: higher pnl_with_markout_time
    assert out["V2"]["total_volume_range"] == [20.0, 20.0]


# ---------------------------------------------------------------------------
# compute_vpin_series
# ---------------------------------------------------------------------------

def test_compute_vpin_series_is_causal_and_matches_hand_computed_values():
    # bucket_notional=10: t0 (BUY 5) starts a bucket; t1 (SELL 5) completes it
    # (buy=5,sell=5 -> imbalance 0.0); t2 (BUY 10) alone completes a second,
    # maximally one-sided bucket (imbalance 1.0); t3 sees both.
    trades = [_mk(1.0, 5.0, "BUY", 0), _mk(1.0, 5.0, "SELL", 1),
              _mk(1.0, 10.0, "BUY", 2), _mk(1.0, 10.0, "BUY", 3)]
    vpin = compute_vpin_series(trades, bucket_notional=10.0, window_buckets=20)
    assert vpin[0] is None   # no bucket has completed yet
    assert vpin[1] is None   # the bucket t1 itself completes isn't usable until AFTER t1
    assert vpin[2] == pytest.approx(0.0)
    assert vpin[3] == pytest.approx(0.5)  # rolling avg of completed buckets [0.0, 1.0]


def test_compute_vpin_series_empty_input():
    assert compute_vpin_series([], bucket_notional=10.0) == []


# ---------------------------------------------------------------------------
# market_pnl_advanced: VPIN-driven dynamic spread + inventory-aware skew
# ---------------------------------------------------------------------------

def test_market_pnl_advanced_reduces_to_plain_market_pnl_when_controls_disabled():
    # vpin_spread_multiplier_max=1.0 (spread never widens) and
    # inventory_limit_notional=inf (skew never binds) should reproduce
    # market_pnl's own numbers exactly.
    trades = [_mk(0.5, 10.0, "SELL", i * 5) for i in range(10)]
    total_volume = 100_000.0  # large on purpose -- keeps the liquidity cap from binding
    base = market_pnl(trades, total_volume, half_spread=0.01, fill_share=0.2)
    advanced = market_pnl_advanced(trades, total_volume, half_spread=0.01, fill_share=0.2,
                                    vpin_spread_multiplier_max=1.0, inventory_limit_notional=float("inf"))
    assert advanced["pnl_best_case"] == pytest.approx(base["pnl_best_case"])
    assert advanced["pnl_with_markout_time"] == pytest.approx(base["pnl_with_markout_time"])
    assert advanced["n_captured"] == base["n_captured"]


def test_market_pnl_advanced_vpin_widening_increases_best_case_pnl():
    # All-BUY flow is maximally one-sided (VPIN -> 1.0 once a bucket
    # completes), so the widened-spread run should earn more per captured
    # share than the same tape with VPIN widening disabled.
    trades = [_mk(0.5, 20.0, "BUY", i) for i in range(10)]
    widened = market_pnl_advanced(trades, 100_000.0, half_spread=0.01, fill_share=0.1,
                                   vpin_bucket_notional=10.0, vpin_window_buckets=5,
                                   vpin_spread_multiplier_max=2.0, inventory_limit_notional=float("inf"))
    flat = market_pnl_advanced(trades, 100_000.0, half_spread=0.01, fill_share=0.1,
                                vpin_bucket_notional=10.0, vpin_window_buckets=5,
                                vpin_spread_multiplier_max=1.0, inventory_limit_notional=float("inf"))
    assert widened["avg_vpin"] == pytest.approx(1.0)
    assert widened["pnl_best_case"] > flat["pnl_best_case"]


def test_market_pnl_advanced_inventory_skew_derates_fills_that_build_a_position():
    # 5 identical SELL prints (we keep buying -> inventory only grows) with a
    # $20 inventory limit; ground truth verified independently against the
    # implementation (see commit message) since this compounds over 5 steps.
    trades = [_mk(0.5, 10.0, "SELL", i * 5) for i in range(5)]
    r = market_pnl_advanced(trades, 100_000.0, half_spread=0.01, fill_share=1.0,
                             vpin_spread_multiplier_max=1.0, inventory_limit_notional=20.0)
    assert r["n_captured"] == 5
    assert r["n_inventory_capped"] == 4  # every fill after the first (which starts at flat)
    assert r["max_abs_inventory_notional"] == pytest.approx(15.25390625)
    # skew must have actually reduced captured flow vs. the unlimited case
    r_unlimited = market_pnl_advanced(trades, 100_000.0, half_spread=0.01, fill_share=1.0,
                                       vpin_spread_multiplier_max=1.0, inventory_limit_notional=float("inf"))
    assert r["captured_notional"] < r_unlimited["captured_notional"]


def test_market_pnl_advanced_never_skews_the_flattening_side():
    # Alternating SELL/BUY prints of equal size keep inventory near flat --
    # a tight inventory limit should never bind since no fill pushes further
    # from flat than the previous one already returned from.
    trades = [_mk(0.5, 10.0, "SELL" if i % 2 == 0 else "BUY", i) for i in range(10)]
    r = market_pnl_advanced(trades, 100_000.0, half_spread=0.01, fill_share=1.0,
                             vpin_spread_multiplier_max=1.0, inventory_limit_notional=1.0)
    assert r["n_inventory_capped"] == 0
    assert r["n_captured"] == 10
