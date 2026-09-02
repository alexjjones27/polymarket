"""Unit test for polyorderbooks_client.py's one pure function (reduce_to_touch).
No network calls -- everything else in that module talks to the live
PolyOrderbooks API and is exercised by actually running
scripts/backfill_polyorderbooks_l2.py, not by a mocked unit test.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polyorderbooks_client import reduce_to_touch


def test_reduce_to_touch_reads_best_bid_and_ask_with_sizes():
    snap = {"t": "2026-09-02T12:00:00Z", "bids": [[0.49, 10.0], [0.48, 5.0]], "asks": [[0.51, 8.0], [0.52, 3.0]]}
    assert reduce_to_touch(snap) == [1788350400, 0.49, 10.0, 0.51, 8.0]


def test_reduce_to_touch_none_for_empty_side():
    snap = {"t": "2026-09-02T12:00:00Z", "bids": [], "asks": [[0.51, 8.0]]}
    ts, best_bid, best_bid_size, best_ask, best_ask_size = reduce_to_touch(snap)
    assert best_bid is None and best_bid_size is None
    assert best_ask == 0.51 and best_ask_size == 8.0


def test_reduce_to_touch_none_for_both_missing():
    snap = {"t": "2026-09-02T12:00:00Z", "bids": [], "asks": []}
    ts, best_bid, best_bid_size, best_ask, best_ask_size = reduce_to_touch(snap)
    assert best_bid is None and best_ask is None
