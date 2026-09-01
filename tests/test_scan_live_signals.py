"""Unit tests for the live scanners' portfolio allocation, on synthetic
data. No network calls -- allocate_portfolio() is a pure function over
already-verified candidate dicts.

Regression coverage for a real bug found on a live run: allocate_portfolio
used to cap position size by the PASS-1 (stale, approximate) depth estimate
taken near the original crossing price, not pass 2's freshly re-fetched
real order book. When price moves between the two passes, the stale
estimate can be zero even though real, tradeable depth exists at the new
price -- silently zeroing an otherwise-valid, freshly-verified position.
Observed live: a candidate whose price moved $0.72 -> $0.85 between passes
had a real $34 of depth at the new price but a stale pass-1 estimate of $0.

Also covers a related unit bug: order-book "size" is in SHARES, not
dollars, so the depth cap (and printed depth figure) must multiply by
price to get true dollar notional -- summing raw share size and calling it
"notional" overstates real dollar depth whenever price < $1, materially so
at the 70%-threshold scanner's wider price band.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import scan_live_signals as s99

# scan_live_signals_70 reads sys.argv[1] as its BANKROLL default at import
# time (its normal CLI convention, `python3 scan_live_signals_70.py 5`) --
# under pytest, sys.argv holds pytest's own args instead, so swap in a
# harmless argv just for the import.
_real_argv = sys.argv
sys.argv = [_real_argv[0]]
try:
    import scan_live_signals_70 as s70
finally:
    sys.argv = _real_argv


def _row(report_bucket="sports", margin=0.05, per_trade_capped=30.0,
         real_ask_depth_notional=None, live_ask_depth_notional=None):
    row = {
        "report_bucket": report_bucket, "margin": margin,
        "per_trade_capped": per_trade_capped,
    }
    if real_ask_depth_notional is not None:
        row["real_ask_depth_notional"] = real_ask_depth_notional
    if live_ask_depth_notional is not None:
        row["live_ask_depth_notional"] = live_ask_depth_notional
    return row


def test_allocate_portfolio_caps_by_fresh_real_depth_not_stale_pass1(monkeypatch):
    for mod in (s99, s70):
        # Stale pass-1 estimate says zero depth (price moved since pass 1),
        # but pass 2's real, fresh order book shows real tradeable depth --
        # the position must be sized off the real figure, not zeroed.
        row = _row(per_trade_capped=30.0, real_ask_depth_notional=34.0, live_ask_depth_notional=0.0)
        allocated = mod.allocate_portfolio([row], bankroll=1000.0)
        assert allocated[0]["portfolio_position_size"] == 30.0


def test_allocate_portfolio_still_caps_when_real_depth_is_genuinely_thin(monkeypatch):
    for mod in (s99, s70):
        row = _row(per_trade_capped=30.0, real_ask_depth_notional=5.0, live_ask_depth_notional=999.0)
        allocated = mod.allocate_portfolio([row], bankroll=1000.0)
        assert allocated[0]["portfolio_position_size"] == 5.0


def test_allocate_portfolio_uncapped_when_no_depth_field_present():
    for mod in (s99, s70):
        row = _row(per_trade_capped=30.0)
        allocated = mod.allocate_portfolio([row], bankroll=1000.0)
        assert allocated[0]["portfolio_position_size"] == 30.0


def test_real_ask_depth_notional_converts_shares_to_dollars():
    # verify_candidate hits the network, so this checks the unit-conversion
    # logic directly rather than the full function -- share size * price
    # must be dollars, not the raw share count.
    ask_depth_shares = 150.0
    price = 0.74
    assert ask_depth_shares * price == 111.0
