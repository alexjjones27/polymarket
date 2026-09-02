"""Unit tests for run_mm_proxy_advanced.py's pure functions. No network calls.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_mm_proxy_advanced import CATEGORY_WINDOW_MAX_S, CATEGORY_WINDOW_MIN_S, compute_category_windows


def test_compute_category_windows_uses_the_median_pace():
    per_market_pace = {"sports": [10.0, 20.0, 30.0], "politics": [50.0, 60.0]}
    windows = compute_category_windows(per_market_pace)
    assert windows["sports"] == pytest.approx(20.0)
    assert windows["politics"] == pytest.approx(55.0)  # avg of the middle two


def test_compute_category_windows_clamps_to_bounds():
    per_market_pace = {"fast": [0.1, 0.2, 0.3], "slow": [10_000.0, 20_000.0, 30_000.0]}
    windows = compute_category_windows(per_market_pace)
    assert windows["fast"] == pytest.approx(CATEGORY_WINDOW_MIN_S)
    assert windows["slow"] == pytest.approx(CATEGORY_WINDOW_MAX_S)


def test_compute_category_windows_skips_categories_with_no_pace_data():
    per_market_pace = {"sports": [10.0], "empty": []}
    windows = compute_category_windows(per_market_pace)
    assert "sports" in windows
    assert "empty" not in windows
