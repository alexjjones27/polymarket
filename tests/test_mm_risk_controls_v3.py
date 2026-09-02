"""Unit tests for mm_risk_controls_v3.py's pure functions, on synthetic data.
No network calls. Mirrors the style of test_mm_proxy_backtest.py's VPIN/
inventory-skew section: causality checks, hand-computed small examples, and
"reduces to the prior model when every new control is disabled" regression
tests tying this file to market_pnl_advanced.
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mm_risk_controls_v3 import (
    MIN_DYNAMIC_BUCKET_NOTIONAL,
    cooldown_heat,
    compute_order_imbalance_series,
    compute_vpin_series_dynamic,
    decay_inventory_toward_target,
    market_pnl_v3,
    realized_price_vol,
    rolling_median_trade_notional,
    sigmoid_skew_headroom,
    volatility_scaled_notional_cap,
)
from run_mm_proxy_backtest import market_pnl_advanced


def _mk(price, size, side, ts):
    return {"price": price, "size": size, "side": side, "timestamp": ts}


# ---------------------------------------------------------------------------
# rolling_median_trade_notional / compute_vpin_series_dynamic
# ---------------------------------------------------------------------------

def test_rolling_median_trade_notional_none_before_min_observations():
    trades = [_mk(0.5, 10.0, "BUY", i) for i in range(4)]
    assert rolling_median_trade_notional(trades, i=4, window=50, min_observations=5) is None


def test_rolling_median_trade_notional_uses_only_prior_trades():
    # trades[0:5] all have notional 5.0 (0.5*10); trade index 5 is a $500
    # notional outlier that must NOT affect the median computed at i=5,
    # since only trades strictly before index 5 are in scope.
    sizes = [10.0, 10.0, 10.0, 10.0, 10.0, 1000.0]
    trades = [_mk(0.5, s, "BUY", i) for i, s in enumerate(sizes)]
    median = rolling_median_trade_notional(trades, i=5, window=50, min_observations=5)
    assert median == pytest.approx(5.0)


def test_compute_vpin_series_dynamic_causal_and_falls_back_before_calibration():
    # Fewer than VPIN_MIN_OBSERVATIONS (5) prior trades -> falls back to
    # fallback_bucket_notional, exactly like the fixed-bucket model would.
    trades = [_mk(1.0, 5.0, "BUY", 0), _mk(1.0, 5.0, "SELL", 1),
              _mk(1.0, 10.0, "BUY", 2), _mk(1.0, 10.0, "BUY", 3)]
    vpin = compute_vpin_series_dynamic(trades, bucket_trade_target=1.0, fallback_bucket_notional=10.0,
                                        rolling_window=50, window_buckets=20)
    assert vpin[0] is None
    assert vpin[1] is None  # bucket completing at t1 isn't usable until AFTER t1
    assert vpin[2] == pytest.approx(0.0)
    assert vpin[3] == pytest.approx(0.5)


def test_compute_vpin_series_dynamic_empty_input():
    assert compute_vpin_series_dynamic([]) == []


def test_compute_vpin_series_dynamic_adapts_bucket_size_to_market_trade_size():
    # A market whose trades are consistently large (notional ~$1000) should
    # complete VPIN buckets after roughly bucket_trade_target trades, not
    # after a huge number of them the way a flat $500 threshold would with
    # trade sizes this large (each single trade would overshoot $500 alone).
    trades = [_mk(1.0, 1000.0, "BUY" if i % 2 == 0 else "SELL", i) for i in range(30)]
    vpin_dynamic = compute_vpin_series_dynamic(trades, bucket_trade_target=2.0, rolling_window=10,
                                                 fallback_bucket_notional=500.0)
    # Once calibrated (after the fallback-using warmup), buckets should be
    # completing roughly every ~2 trades (perfectly alternating BUY/SELL ->
    # imbalance 0.0 each time), giving a defined (non-None) VPIN well before
    # the end of the tape.
    assert vpin_dynamic[-1] is not None


# ---------------------------------------------------------------------------
# sigmoid_skew_headroom
# ---------------------------------------------------------------------------

def test_sigmoid_skew_headroom_no_position_is_barely_derated():
    h = sigmoid_skew_headroom(0.0, 100.0, skew_strength=6.0)
    assert h > 0.99


def test_sigmoid_skew_headroom_exactly_at_limit_is_half_derated():
    h = sigmoid_skew_headroom(100.0, 100.0, skew_strength=6.0)
    assert h == pytest.approx(0.5)


def test_sigmoid_skew_headroom_beyond_limit_approaches_zero():
    h = sigmoid_skew_headroom(300.0, 100.0, skew_strength=6.0)
    assert h < 0.01


def test_sigmoid_skew_headroom_higher_strength_is_steeper_near_the_limit():
    # At u=0.5 (half of the way to the limit), a higher skew_strength should
    # derate LESS (steeper curve concentrates the derating closer to u=1).
    gentle = sigmoid_skew_headroom(50.0, 100.0, skew_strength=2.0)
    steep = sigmoid_skew_headroom(50.0, 100.0, skew_strength=10.0)
    assert steep > gentle


def test_sigmoid_skew_headroom_no_limit_configured_returns_one():
    assert sigmoid_skew_headroom(1000.0, 0.0) == pytest.approx(1.0)
    assert sigmoid_skew_headroom(1000.0, float("inf")) == pytest.approx(1.0)


def test_sigmoid_skew_headroom_extreme_overshoot_does_not_overflow():
    # A position that has run to thousands of times the nominal limit, with
    # a steep skew_strength, previously raised OverflowError from math.exp.
    assert sigmoid_skew_headroom(1_000_000.0, 100.0, skew_strength=50.0) == pytest.approx(0.0)
    assert sigmoid_skew_headroom(-1_000_000.0, 100.0, skew_strength=50.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# decay_inventory_toward_target
# ---------------------------------------------------------------------------

def test_decay_inventory_toward_target_one_half_life_halves_the_gap():
    result = decay_inventory_toward_target(100.0, dt_seconds=30.0, half_life_seconds=30.0, target_shares=0.0)
    assert result == pytest.approx(50.0)


def test_decay_inventory_toward_target_decays_toward_nonzero_target():
    result = decay_inventory_toward_target(0.0, dt_seconds=30.0, half_life_seconds=30.0, target_shares=100.0)
    assert result == pytest.approx(50.0)


def test_decay_inventory_toward_target_disabled_when_half_life_is_none():
    assert decay_inventory_toward_target(100.0, dt_seconds=1000.0, half_life_seconds=None) == pytest.approx(100.0)


def test_decay_inventory_toward_target_no_op_at_zero_elapsed_time():
    assert decay_inventory_toward_target(100.0, dt_seconds=0.0, half_life_seconds=30.0) == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# realized_price_vol / volatility_scaled_notional_cap
# ---------------------------------------------------------------------------

def test_realized_price_vol_none_before_min_observations():
    trades = [_mk(0.5, 10.0, "BUY", i) for i in range(3)]
    assert realized_price_vol(trades, i=3, window=50, min_observations=5) is None


def test_realized_price_vol_is_causal():
    # A huge price swing that happens AT index i or later must not affect
    # the vol estimate used to size trade i's own cap.
    stable = [_mk(0.5, 10.0, "BUY", i) for i in range(10)]
    wild_tail = stable[:10] + [_mk(0.99, 10.0, "BUY", 10)]
    vol_before_swing = realized_price_vol(wild_tail, i=10, window=50, min_observations=5)
    assert vol_before_swing == pytest.approx(0.0)  # all prior prices identical


def test_volatility_scaled_notional_cap_shrinks_at_high_vol():
    base_cap = 25.0
    low_vol_cap = volatility_scaled_notional_cap(base_cap, vol=0.01, vol_reference=0.02)
    high_vol_cap = volatility_scaled_notional_cap(base_cap, vol=0.10, vol_reference=0.02)
    assert low_vol_cap == pytest.approx(base_cap)  # below reference -> unscaled
    assert high_vol_cap < low_vol_cap


def test_volatility_scaled_notional_cap_floored_not_zero():
    cap = volatility_scaled_notional_cap(25.0, vol=1000.0, vol_reference=0.02, sensitivity=1.0, min_cap_fraction=0.2)
    assert cap == pytest.approx(5.0)  # 0.2 * 25.0 floor


def test_volatility_scaled_notional_cap_none_vol_returns_base_cap():
    assert volatility_scaled_notional_cap(25.0, vol=None) == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# compute_order_imbalance_series
# ---------------------------------------------------------------------------

def test_compute_order_imbalance_series_all_buys_is_plus_one():
    trades = [_mk(0.5, 10.0, "BUY", i) for i in range(5)]
    series = compute_order_imbalance_series(trades, window_trades=3)
    assert series[0] is None  # no prior trades
    assert series[4] == pytest.approx(1.0)


def test_compute_order_imbalance_series_balanced_flow_is_zero():
    trades = [_mk(0.5, 10.0, "BUY" if i % 2 == 0 else "SELL", i) for i in range(6)]
    series = compute_order_imbalance_series(trades, window_trades=4)
    assert series[5] == pytest.approx(0.0)


def test_compute_order_imbalance_series_is_causal():
    trades = [_mk(0.5, 10.0, "BUY", i) for i in range(5)] + [_mk(0.5, 1000.0, "SELL", 5)]
    series = compute_order_imbalance_series(trades, window_trades=10)
    assert series[5] == pytest.approx(1.0)  # the huge SELL at index 5 must not affect its own signal


# ---------------------------------------------------------------------------
# cooldown_heat
# ---------------------------------------------------------------------------

def test_cooldown_heat_zero_before_any_trigger():
    assert cooldown_heat(None) == 0.0


def test_cooldown_heat_full_at_zero_elapsed():
    assert cooldown_heat(0.0, half_life_seconds=30.0) == pytest.approx(1.0)


def test_cooldown_heat_halves_after_one_half_life():
    assert cooldown_heat(30.0, half_life_seconds=30.0) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# market_pnl_v3: reduction to market_pnl_advanced when every new control is off
# ---------------------------------------------------------------------------

def test_market_pnl_v3_reduces_to_market_pnl_advanced_with_matching_defaults():
    trades = [_mk(0.5, 10.0, "SELL" if i % 3 else "BUY", i * 4) for i in range(40)]
    total_volume = 100_000.0
    advanced = market_pnl_advanced(trades, total_volume, half_spread=0.01, fill_share=0.2)
    v3 = market_pnl_v3(trades, total_volume, half_spread=0.01, fill_share=0.2,
                        toxicity_mode="vpin_fixed", skew_mode="linear",
                        use_volatility_cap=False, enable_cooldown=False)
    assert v3["pnl_best_case"] == pytest.approx(advanced["pnl_best_case"])
    assert v3["pnl_with_markout_time"] == pytest.approx(advanced["pnl_with_markout_time"])
    assert v3["n_captured"] == advanced["n_captured"]
    assert v3["captured_notional"] == pytest.approx(advanced["captured_notional"])
    assert v3["avg_vpin"] == pytest.approx(advanced["avg_vpin"])
    assert v3["n_inventory_capped"] == advanced["n_inventory_capped"]
    assert v3["max_abs_inventory_notional"] == pytest.approx(advanced["max_abs_inventory_notional"])


def test_market_pnl_v3_disabled_cooldown_never_triggers():
    trades = [_mk(0.5, 10.0, "BUY", i) for i in range(50)] + [_mk(0.3, 10.0, "SELL", 100)]
    r = market_pnl_v3(trades, 100_000.0, half_spread=0.01, fill_share=0.5, enable_cooldown=False)
    assert r["n_cooldown_triggers"] == 0


def test_market_pnl_v3_sigmoid_skew_derates_less_than_linear_for_small_positions():
    # A market maker that has barely built a position should be derated less
    # under the sigmoid (which is gentle near zero) than under the old linear
    # ramp, at the same nominal limit.
    trades = [_mk(0.5, 10.0, "SELL", i * 5) for i in range(3)]  # small position only
    linear = market_pnl_v3(trades, 100_000.0, half_spread=0.01, fill_share=1.0,
                            skew_mode="linear", inventory_limit_notional=100.0)
    sigmoid = market_pnl_v3(trades, 100_000.0, half_spread=0.01, fill_share=1.0,
                             skew_mode="sigmoid", inventory_limit_notional=100.0, skew_strength=6.0)
    assert sigmoid["captured_notional"] >= linear["captured_notional"]


def test_market_pnl_v3_volatility_cap_reduces_capture_in_a_volatile_tape():
    import random
    rng = random.Random(0)
    volatile = [_mk(max(0.01, min(0.99, 0.5 + rng.uniform(-0.2, 0.2))), 50.0, "BUY", i) for i in range(80)]
    calm = [_mk(0.5, 50.0, "BUY", i) for i in range(80)]
    r_volatile = market_pnl_v3(volatile, 1_000_000.0, half_spread=0.01, fill_share=1.0, use_volatility_cap=True,
                                inventory_limit_notional=float("inf"))
    r_calm = market_pnl_v3(calm, 1_000_000.0, half_spread=0.01, fill_share=1.0, use_volatility_cap=True,
                            inventory_limit_notional=float("inf"))
    assert r_volatile["captured_notional"] < r_calm["captured_notional"]


def test_market_pnl_v3_cooldown_widens_effective_spread_after_a_toxic_fill():
    # A resting SELL-side quote gets hit (we captured a BUY print, so we're
    # short), then price drifts sharply and persistently AGAINST that
    # position (up) for the rest of the tape -- a textbook toxic fill. Once
    # cooldown detects it, later fills earn a wider spread and smaller size
    # during the adverse drift, which should PROTECT markout PnL relative to
    # an otherwise-identical run with cooldown disabled (which keeps quoting
    # the same size/spread straight into the adverse move).
    toxic_trade = [_mk(0.40, 20.0, "BUY", 0)]
    drift_up = [_mk(0.40 + 0.01 * i, 20.0, "BUY", 10 + i * 3) for i in range(1, 15)]
    trades = toxic_trade + drift_up
    with_cooldown = market_pnl_v3(trades, 1_000_000.0, half_spread=0.01, fill_share=1.0,
                                   inventory_limit_notional=float("inf"), enable_cooldown=True,
                                   toxic_eval_delay_seconds=5.0, toxic_adverse_spread_multiple=0.5)
    without_cooldown = market_pnl_v3(trades, 1_000_000.0, half_spread=0.01, fill_share=1.0,
                                      inventory_limit_notional=float("inf"), enable_cooldown=False)
    assert with_cooldown["n_cooldown_triggers"] > 0
    assert with_cooldown["pnl_with_markout_time"] >= without_cooldown["pnl_with_markout_time"]


def test_market_pnl_v3_unknown_toxicity_mode_raises():
    trades = [_mk(0.5, 10.0, "BUY", 0)]
    with pytest.raises(ValueError):
        market_pnl_v3(trades, 100_000.0, half_spread=0.01, fill_share=0.5, toxicity_mode="bogus")


def test_market_pnl_v3_unknown_skew_mode_raises():
    trades = [_mk(0.5, 10.0, "SELL", i) for i in range(3)]
    with pytest.raises(ValueError):
        market_pnl_v3(trades, 100_000.0, half_spread=0.01, fill_share=1.0, skew_mode="bogus",
                       inventory_limit_notional=1.0)
