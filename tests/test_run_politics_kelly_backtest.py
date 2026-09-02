"""Unit test for run_politics_kelly_backtest.py's one pure aggregation
function (bucket_calibration_report). No network calls. run_sim/main are
event-loop orchestration over real entry data -- same convention as
run_kelly_backtest.py's own run_sim, which has no dedicated unit test either.
"""
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_politics_kelly_backtest import bucket_calibration_report


def _entry(yes_price, resolved_yes):
    return {"yes_price": yes_price, "resolved_yes": resolved_yes}


def test_bucket_calibration_report_computes_empirical_rate_per_bucket():
    entries = [
        _entry(0.55, True), _entry(0.58, True), _entry(0.52, False),  # 50-60% bucket: 2/3
        _entry(0.15, False), _entry(0.12, False),                      # 10-20% bucket: 0/2
    ]
    report = bucket_calibration_report(entries)
    assert report["50%-60%"]["n"] == 3
    assert report["50%-60%"]["empirical_yes_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert report["50%-60%"]["bucket_midpoint"] == pytest.approx(0.55)
    assert report["10%-20%"]["n"] == 2
    assert report["10%-20%"]["empirical_yes_rate"] == pytest.approx(0.0)


def test_bucket_calibration_report_omits_empty_buckets():
    entries = [_entry(0.05, True)]
    report = bucket_calibration_report(entries)
    assert list(report.keys()) == ["0%-10%"]
