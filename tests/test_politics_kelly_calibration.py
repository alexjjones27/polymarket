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
