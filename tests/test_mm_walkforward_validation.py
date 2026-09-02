"""Unit tests for the walk-forward validation script's pure functions, on
synthetic data. No network calls.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_mm_walkforward_validation import (
    bootstrap_ci,
    chronological_split,
    compute_drawdown,
    evaluate_filter,
    select_best_filter,
)


def _row(cid, resolution_date, pace, volume, history_s, pnl, pnl_markout, bucket="other"):
    return {
        "condition_id": cid, "question": f"q-{cid}", "resolution_time": f"{resolution_date} 00:00:00+00:00",
        "report_bucket": bucket, "pnl": pnl, "pnl_with_markout_time": pnl_markout,
        "median_inter_trade_s": pace, "total_market_volume": volume, "pre_resolution_history_s": history_s,
    }


# ---------------------------------------------------------------------------
# chronological_split
# ---------------------------------------------------------------------------

def test_chronological_split_is_earlier_vs_later_by_resolution_time():
    rows = [_row(str(i), f"2024-01-{i+1:02d}", 100, 100, 100, 1.0, 1.0) for i in range(10)]
    train, test = chronological_split(rows, train_frac=0.7)
    assert len(train) == 7 and len(test) == 3
    assert max(r["resolution_time"] for r in train) < min(r["resolution_time"] for r in test)


# ---------------------------------------------------------------------------
# evaluate_filter
# ---------------------------------------------------------------------------

def test_evaluate_filter_applies_all_three_conditions():
    rows = [
        _row("a", "2024-01-01", pace=100, volume=1000, history_s=100_000, pnl=1.0, pnl_markout=5.0),
        _row("b", "2024-01-02", pace=5, volume=1000, history_s=100_000, pnl=1.0, pnl_markout=-5.0),   # fails pace
        _row("c", "2024-01-03", pace=100, volume=10, history_s=100_000, pnl=1.0, pnl_markout=-5.0),   # fails volume
        _row("d", "2024-01-04", pace=100, volume=1000, history_s=10, pnl=1.0, pnl_markout=-5.0),      # fails history
    ]
    out = evaluate_filter(rows, pace_range=(50, 200), volume_min=500, history_min_s=50_000)
    assert out["n_markets"] == 1
    assert out["total_pnl_with_markout_time"] == pytest.approx(5.0)


def test_evaluate_filter_empty_result_has_none_stats_not_a_crash():
    rows = [_row("a", "2024-01-01", pace=5, volume=10, history_s=10, pnl=1.0, pnl_markout=1.0)]
    out = evaluate_filter(rows, pace_range=(1000, 2000), volume_min=0, history_min_s=0)
    assert out["n_markets"] == 0
    assert out["median_pnl_with_markout_time"] is None


# ---------------------------------------------------------------------------
# select_best_filter
# ---------------------------------------------------------------------------

def test_select_best_filter_picks_the_combo_with_highest_train_median():
    # 40 markets with pace/volume/history all in a "good" tight range with
    # positive markout, plus 40 "bad" markets elsewhere with negative
    # markout -- the winning combo should isolate the good group.
    good = [_row(f"g{i}", f"2024-01-{(i % 28) + 1:02d}", pace=100, volume=1000, history_s=100_000,
                 pnl=1.0, pnl_markout=5.0) for i in range(40)]
    bad = [_row(f"b{i}", f"2024-02-{(i % 28) + 1:02d}", pace=5, volume=10, history_s=10,
                pnl=1.0, pnl_markout=-5.0) for i in range(40)]
    train = good + bad
    winner = select_best_filter(train, min_markets=10)
    assert winner is not None
    assert winner["result"]["median_pnl_with_markout_time"] > 0


def test_select_best_filter_returns_none_when_no_combo_meets_minimum():
    rows = [_row(f"a{i}", f"2024-01-{i+1:02d}", pace=100, volume=1000, history_s=100_000, pnl=1.0, pnl_markout=1.0)
            for i in range(5)]
    assert select_best_filter(rows, min_markets=100) is None


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

def test_bootstrap_ci_all_positive_markets_gives_100pct_positive():
    rows = [_row(str(i), "2024-01-01", 100, 100, 100, 1.0, 5.0) for i in range(20)]
    ci = bootstrap_ci(rows, n_iterations=200, seed=1)
    assert ci["pct_iterations_positive"] == pytest.approx(100.0)
    assert ci["median"] > 0


def test_bootstrap_ci_empty_input_returns_zero_markets():
    ci = bootstrap_ci([])
    assert ci["n_markets"] == 0


# ---------------------------------------------------------------------------
# compute_drawdown
# ---------------------------------------------------------------------------

def test_compute_drawdown_tracks_peak_to_trough():
    rows = [
        _row("a", "2024-01-01", 100, 100, 100, 1.0, 10.0),   # equity 10, peak 10
        _row("b", "2024-01-02", 100, 100, 100, 1.0, -15.0),  # equity -5, dd = 15
        _row("c", "2024-01-03", 100, 100, 100, 1.0, 3.0),    # equity -2, dd still 15
    ]
    dd = compute_drawdown(rows)
    assert dd["final_equity"] == pytest.approx(-2.0)
    assert dd["max_drawdown"] == pytest.approx(15.0)
