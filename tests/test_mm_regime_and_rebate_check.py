"""Unit tests for the regime-change / rebate check's pure functions. No
network calls.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_mm_regime_and_rebate_check import split_trades_by_regime


def _mk(price, size, side, ts):
    return {"price": price, "size": size, "side": side, "timestamp": ts}


def test_split_trades_by_regime_partitions_on_the_cutoff():
    trades = [_mk(0.5, 1.0, "BUY", ts) for ts in [0, 50, 100, 150, 200]]
    pre, post = split_trades_by_regime(trades, cutoff_epoch=100)
    assert [t["timestamp"] for t in pre] == [0, 50]
    assert [t["timestamp"] for t in post] == [100, 150, 200]


def test_split_trades_by_regime_all_pre_when_cutoff_is_far_in_the_future():
    trades = [_mk(0.5, 1.0, "BUY", ts) for ts in [0, 50, 100]]
    pre, post = split_trades_by_regime(trades, cutoff_epoch=1_000_000)
    assert len(pre) == 3
    assert post == []


def test_split_trades_by_regime_all_post_when_cutoff_is_in_the_past():
    trades = [_mk(0.5, 1.0, "BUY", ts) for ts in [0, 50, 100]]
    pre, post = split_trades_by_regime(trades, cutoff_epoch=-1)
    assert pre == []
    assert len(post) == 3


def test_split_trades_by_regime_empty_input():
    pre, post = split_trades_by_regime([], cutoff_epoch=100)
    assert pre == [] and post == []
