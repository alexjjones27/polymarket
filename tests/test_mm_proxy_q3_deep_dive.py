"""Unit tests for the Q3 deep-dive script's pure functions. No network calls
and no dependency on the real mm_proxy_results.json -- load_bucket_condition_ids
is tested against a small synthetic file, find_crossover against synthetic grids.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_mm_proxy_q3_deep_dive import (
    build_resolution_exclusion_sweep,
    build_volume_threshold_sweep,
    find_crossover,
    load_bucket_condition_ids,
)


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


# ---------------------------------------------------------------------------
# build_volume_threshold_sweep
# ---------------------------------------------------------------------------

def _vrow(volume, pnl_time):
    return {"total_market_volume": volume, "pnl": 1.0, "pnl_with_markout_time": pnl_time}


def test_build_volume_threshold_sweep_reports_per_market_stats_not_just_the_sum():
    # Deliberately mirrors the real finding: low-volume markets lose a
    # little, the highest-volume one wins big. The sum can hide that most
    # markets are negative -- median and pct_positive must not.
    rows = [_vrow(10.0, -10.0), _vrow(20.0, -5.0), _vrow(30.0, 2.0), _vrow(40.0, 3.0), _vrow(50.0, 100.0)]
    sweep = build_volume_threshold_sweep(rows, percentiles=[0, 50, 90])
    by_p = {row["percentile"]: row for row in sweep}

    assert by_p[0]["n_markets"] == 5
    assert by_p[0]["total_pnl_with_markout_time"] == pytest.approx(90.0)
    assert by_p[0]["median_pnl_with_markout_time"] == pytest.approx(2.0)
    assert by_p[0]["pct_markets_positive_markout"] == pytest.approx(60.0)

    assert by_p[50]["n_markets"] == 3  # volume >= 30 -> the 30/40/50 markets
    assert by_p[50]["median_pnl_with_markout_time"] == pytest.approx(3.0)
    assert by_p[50]["pct_markets_positive_markout"] == pytest.approx(100.0)

    assert by_p[90]["n_markets"] == 1  # just the $50-volume market
    assert by_p[90]["total_pnl_with_markout_time"] == pytest.approx(100.0)


def test_build_volume_threshold_sweep_handles_empty_input():
    sweep = build_volume_threshold_sweep([], percentiles=[0, 50])
    assert all(row["n_markets"] == 0 for row in sweep)
    assert all(row["median_pnl_with_markout_time"] is None for row in sweep)


# ---------------------------------------------------------------------------
# build_resolution_exclusion_sweep
# ---------------------------------------------------------------------------

def _mk(price, size, side, ts):
    return {"price": price, "size": size, "side": side, "timestamp": ts}


def test_build_resolution_exclusion_sweep_drops_trades_near_resolution():
    trades = [_mk(0.5, 10.0, "BUY", ts) for ts in [0, 100, 200, 900]]
    total_volume = sum(t["price"] * t["size"] for t in trades)
    markets = [(trades, total_volume, 1000.0)]  # resolution_epoch = 1000

    sweep = build_resolution_exclusion_sweep(markets, windows=[0, 150])
    by_w = {row["exclusion_seconds"]: row for row in sweep}

    assert by_w[0]["pct_trades_remaining"] == pytest.approx(100.0)
    # window=150 drops only the ts=900 trade (1000-900=100 <= 150)
    assert by_w[150]["pct_trades_remaining"] == pytest.approx(75.0)
    assert by_w[150]["total_pnl_best_case"] < by_w[0]["total_pnl_best_case"]


def test_build_resolution_exclusion_sweep_noop_when_resolution_unparseable():
    trades = [_mk(0.5, 10.0, "BUY", ts) for ts in [0, 100, 200]]
    total_volume = sum(t["price"] * t["size"] for t in trades)
    markets = [(trades, total_volume, None)]
    sweep = build_resolution_exclusion_sweep(markets, windows=[0, 100, 99999])
    assert all(row["pct_trades_remaining"] == pytest.approx(100.0) for row in sweep)


def test_build_resolution_exclusion_sweep_handles_no_markets():
    sweep = build_resolution_exclusion_sweep([], windows=[0, 100])
    assert all(row["pct_trades_remaining"] is None for row in sweep)
    assert all(row["n_markets_active"] == 0 for row in sweep)
