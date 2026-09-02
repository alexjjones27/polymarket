"""Unit tests for the Q3 deep-dive script's pure functions. No network calls
and no dependency on the real mm_proxy_results.json -- load_bucket_condition_ids
is tested against a small synthetic file, find_crossover against synthetic grids.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_mm_proxy_q3_deep_dive import find_crossover, load_bucket_condition_ids


# ---------------------------------------------------------------------------
# load_bucket_condition_ids
# ---------------------------------------------------------------------------

def _write_results(tmp_path, best_pace_bucket="Q3", rows=None):
    path = tmp_path / "mm_proxy_results.json"
    data = {
        "best_pace_bucket": best_pace_bucket,
        "per_market_base_case": rows if rows is not None else [
            {"condition_id": "a", "pace_bucket": "Q3"},
            {"condition_id": "b", "pace_bucket": "Q1"},
            {"condition_id": "c", "pace_bucket": "Q3"},
            {"condition_id": "d", "pace_bucket": "Q5"},
        ],
    }
    path.write_text(json.dumps(data))
    return path


def test_load_bucket_condition_ids_filters_to_the_target_bucket(tmp_path):
    path = _write_results(tmp_path)
    cids = load_bucket_condition_ids(path, bucket="Q3")
    assert cids == ["a", "c"]


def test_load_bucket_condition_ids_raises_when_bucket_is_empty(tmp_path):
    path = _write_results(tmp_path)
    with pytest.raises(SystemExit):
        load_bucket_condition_ids(path, bucket="Q4")


def test_load_bucket_condition_ids_warns_but_still_works_on_stale_best_bucket(tmp_path, capsys):
    # best_pace_bucket says Q5 now, but the caller explicitly asked for Q3 --
    # should still return Q3's markets, just with a warning printed.
    path = _write_results(tmp_path, best_pace_bucket="Q5")
    cids = load_bucket_condition_ids(path, bucket="Q3")
    assert cids == ["a", "c"]
    assert "WARNING" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# find_crossover
# ---------------------------------------------------------------------------

def test_find_crossover_locates_the_sign_change():
    grid = [
        {"window_seconds": 5, "total_pnl": 100.0},
        {"window_seconds": 15, "total_pnl": 50.0},
        {"window_seconds": 60, "total_pnl": 10.0},
        {"window_seconds": 300, "total_pnl": -20.0},
    ]
    assert find_crossover(grid) == (60, 300)


def test_find_crossover_returns_none_when_positive_throughout():
    grid = [{"window_seconds": w, "total_pnl": 10.0} for w in [5, 15, 60]]
    assert find_crossover(grid) is None


def test_find_crossover_returns_none_when_negative_throughout():
    grid = [{"window_seconds": w, "total_pnl": -10.0} for w in [5, 15, 60]]
    assert find_crossover(grid) is None


def test_find_crossover_picks_the_first_crossing_not_the_last():
    # dips negative, recovers positive, then goes negative again -- should
    # report the FIRST crossing, not the last.
    grid = [
        {"window_seconds": 5, "total_pnl": 10.0},
        {"window_seconds": 15, "total_pnl": -5.0},
        {"window_seconds": 60, "total_pnl": 20.0},
        {"window_seconds": 300, "total_pnl": -30.0},
    ]
    assert find_crossover(grid) == (5, 15)
