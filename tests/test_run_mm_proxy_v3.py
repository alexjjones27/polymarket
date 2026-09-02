"""Unit test for run_mm_proxy_v3.py's one piece of non-trivial pure logic
(unfiltered_summary) on synthetic data. No network calls. Every other helper
in that script (per_market_stats_for_config, select_best_config, main) is
thin orchestration over already-tested functions (market_pnl_v3, evaluate_filter,
select_best_filter, bootstrap_ci, compute_drawdown) and real trade tapes --
same convention as run_mm_proxy_advanced.py and run_mm_walkforward_validation.py,
whose own main()/per-market loops aren't unit tested directly either.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_mm_proxy_v3 import unfiltered_summary


def _row(cid, pace=100, volume=1000, history_s=100_000, pnl=1.0, pnl_markout=1.0):
    return {
        "condition_id": cid, "question": f"q-{cid}", "resolution_time": "2024-01-01 00:00:00+00:00",
        "report_bucket": "other", "pnl": pnl, "pnl_with_markout_time": pnl_markout,
        "median_inter_trade_s": pace, "total_market_volume": volume, "pre_resolution_history_s": history_s,
    }


def test_unfiltered_summary_matches_every_row_with_valid_pace_and_history():
    rows = [_row("a", pnl_markout=5.0), _row("b", pnl_markout=-3.0)]
    out = unfiltered_summary(rows)
    assert out["n_markets"] == 2
    assert out["total_pnl_with_markout_time"] == pytest.approx(2.0)


def test_unfiltered_summary_excludes_rows_with_no_measurable_pace_or_history():
    rows = [_row("a", pace=None, pnl_markout=5.0), _row("b", history_s=None, pnl_markout=3.0), _row("c", pnl_markout=1.0)]
    out = unfiltered_summary(rows)
    assert out["n_markets"] == 1
    assert out["total_pnl_with_markout_time"] == pytest.approx(1.0)


def test_unfiltered_summary_empty_input():
    out = unfiltered_summary([])
    assert out["n_markets"] == 0
    assert out["median_pnl_with_markout_time"] is None
