"""Unit tests for politics_kelly_calibration.py's pure functions. No network
calls.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from politics_kelly_calibration import (
    bucket_midpoint,
    bucket_range,
    calibrated_p_yes,
    kelly_fraction,
    kernel_weight,
    local_calibrated_p_yes,
    midpoint_centered_prior,
    price_bucket,
)


# ---------------------------------------------------------------------------
# price_bucket / bucket_midpoint / bucket_range
# ---------------------------------------------------------------------------

def test_price_bucket_assigns_by_ten_percent_band():
    assert price_bucket(0.05) == 0
    assert price_bucket(0.10) == 1
    assert price_bucket(0.55) == 5
    assert price_bucket(0.99) == 9


def test_price_bucket_clamps_price_of_exactly_one():
    assert price_bucket(1.0) == 9


def test_price_bucket_rejects_out_of_range():
    with pytest.raises(ValueError):
        price_bucket(1.5)
    with pytest.raises(ValueError):
        price_bucket(-0.1)


def test_bucket_midpoint_and_range_agree():
    assert bucket_midpoint(5) == pytest.approx(0.55)
    assert bucket_range(5) == pytest.approx((0.5, 0.6))


# ---------------------------------------------------------------------------
# midpoint_centered_prior
# ---------------------------------------------------------------------------

def test_midpoint_centered_prior_matches_bucket_midpoint():
    a, b = midpoint_centered_prior(bucket_idx=5, prior_strength=20.0)  # bucket [0.5, 0.6)
    assert a == pytest.approx(11.0)  # 0.55 * 20
    assert b == pytest.approx(9.0)   # 0.45 * 20
    assert a + b == pytest.approx(20.0)


def test_midpoint_centered_prior_low_bucket_skews_toward_b():
    a, b = midpoint_centered_prior(bucket_idx=0, prior_strength=10.0)  # bucket [0, 0.1)
    assert a < b


# ---------------------------------------------------------------------------
# calibrated_p_yes
# ---------------------------------------------------------------------------

def test_calibrated_p_yes_with_no_observations_returns_the_prior_mean():
    p = calibrated_p_yes(prior_a=11.0, prior_b=9.0, observed_wins=0, observed_n=0)
    assert p == pytest.approx(0.55)


def test_calibrated_p_yes_shifts_toward_observed_data():
    # Strong observed evidence of a higher-than-implied YES rate.
    p = calibrated_p_yes(prior_a=11.0, prior_b=9.0, observed_wins=90, observed_n=100)
    assert p > 0.55


def test_calibrated_p_yes_converges_as_observations_dominate_the_prior():
    p = calibrated_p_yes(prior_a=1.0, prior_b=1.0, observed_wins=9000, observed_n=10000)
    assert p == pytest.approx(0.9, abs=0.01)


# ---------------------------------------------------------------------------
# kelly_fraction
# ---------------------------------------------------------------------------

def test_kelly_fraction_zero_when_belief_equals_price():
    # The core, load-bearing invariant this whole strategy depends on:
    # trusting price as truth must yield exactly zero stake.
    for price in [0.1, 0.3, 0.5, 0.7, 0.9]:
        assert kelly_fraction(p=price, price=price, fee_frac=0.0) == pytest.approx(0.0)


def test_kelly_fraction_positive_when_belief_exceeds_price():
    f = kelly_fraction(p=0.60, price=0.55, fee_frac=0.0)
    assert f > 0.0


def test_kelly_fraction_zero_when_belief_below_price():
    f = kelly_fraction(p=0.50, price=0.55, fee_frac=0.0)
    assert f == pytest.approx(0.0)


def test_kelly_fraction_fees_shrink_or_erase_a_thin_edge():
    thin_edge_no_fee = kelly_fraction(p=0.56, price=0.55, fee_frac=0.0)
    thin_edge_with_fee = kelly_fraction(p=0.56, price=0.55, fee_frac=0.04)
    assert thin_edge_with_fee < thin_edge_no_fee


def test_kelly_fraction_zero_at_degenerate_prices():
    assert kelly_fraction(p=0.9, price=0.0, fee_frac=0.0) == 0.0
    assert kelly_fraction(p=0.9, price=1.0, fee_frac=0.0) == 0.0


def test_kelly_fraction_matches_hand_computed_value():
    # p=0.70, price=0.50 -> b = (1-0.5)/0.5 = 1.0, q=0.30, L=1.0
    # f = (0.7*1.0 - 0.3*1.0) / (1.0*1.0) = 0.4
    assert kelly_fraction(p=0.70, price=0.50, fee_frac=0.0) == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# kernel_weight
# ---------------------------------------------------------------------------

def test_kernel_weight_full_at_zero_distance():
    assert kernel_weight(0.0, bandwidth=0.05) == pytest.approx(1.0)


def test_kernel_weight_linear_decay():
    assert kernel_weight(0.025, bandwidth=0.05) == pytest.approx(0.5)


def test_kernel_weight_zero_beyond_bandwidth():
    assert kernel_weight(0.05, bandwidth=0.05) == 0.0
    assert kernel_weight(0.10, bandwidth=0.05) == 0.0


# ---------------------------------------------------------------------------
# local_calibrated_p_yes
# ---------------------------------------------------------------------------

def test_local_calibrated_p_yes_with_no_history_returns_price_exactly():
    # The load-bearing invariant this fix exists to preserve: with nothing
    # resolved nearby yet, belief must equal price exactly (zero edge).
    p = local_calibrated_p_yes(price=0.55, history=[], prior_strength=20.0)
    assert p == pytest.approx(0.55)


def test_local_calibrated_p_yes_ignores_history_outside_the_bandwidth():
    history = [(0.90, True), (0.95, False)]  # far from 0.55, all outside a 0.05 bandwidth
    p = local_calibrated_p_yes(price=0.55, history=history, prior_strength=20.0, bandwidth=0.05)
    assert p == pytest.approx(0.55)


def test_local_calibrated_p_yes_pulls_toward_nearby_resolved_outcomes():
    # A cluster of markets priced right next to 0.55 that mostly resolved
    # YES should pull the estimate above the bare price.
    history = [(0.54, True), (0.55, True), (0.56, True), (0.55, False)]
    p = local_calibrated_p_yes(price=0.55, history=history, prior_strength=5.0, bandwidth=0.05)
    assert p > 0.55


def test_local_calibrated_p_yes_does_not_conflate_opposite_ends_of_a_wide_bucket():
    # The exact discretization bug this replaces: two prices both inside the
    # old [0.5, 0.6) bucket, evaluated against the SAME nearby history,
    # should NOT get the same estimate -- 0.51 is close to a cluster of
    # losers near 0.50, 0.59 is not.
    history = [(0.50, False), (0.50, False), (0.50, False), (0.50, False)]
    p_near = local_calibrated_p_yes(price=0.51, history=history, prior_strength=1.0, bandwidth=0.05)
    p_far = local_calibrated_p_yes(price=0.59, history=history, prior_strength=1.0, bandwidth=0.05)
    assert p_near < p_far


def test_local_calibrated_p_yes_higher_prior_strength_resists_a_thin_local_sample():
    history = [(0.551, True)]  # one nearby winner
    weak_prior = local_calibrated_p_yes(price=0.55, history=history, prior_strength=1.0)
    strong_prior = local_calibrated_p_yes(price=0.55, history=history, prior_strength=100.0)
    assert weak_prior > strong_prior  # weak prior lets the single winner move the estimate further
    assert strong_prior == pytest.approx(0.55, abs=0.01)
