"""A market-making backtest built on REAL order-book snapshots
(data/raw/polyorderbooks_l2_live/, fetched via scripts/backfill_polyorderbooks_l2.py)
instead of trade-print inference -- the model every other MM script in this
project (run_mm_proxy_backtest.py and its descendants) has had to use because
Polymarket's own `/book` 404s on resolved markets. This is the first model in
the project that doesn't have that limitation, for the markets it covers
(crypto Up/Down, 5m/15m contracts, whatever PolyOrderbooks' 7-day retention
window still holds -- see docs/mm_strategy_methodology.md Section 11).

What's genuinely better here: the spread is the REAL observed spread at
each moment, not a flat assumed half_spread; a fill is only credited when
the REAL book's own touch demonstrably crossed our quoted price, not
inferred from a trade print's side tag; markout is computed against the
REAL subsequent quote midpoint, not a VWAP of trade prints (which is itself
an approximation of "fair value" the trade-print model needed because it
had no book to look at).

What's still approximate, stated plainly: this is L2 SNAPSHOTS, not an
executed-trades feed, so fill SIZE still has to be assumed (fill_share
applied to an assumed order size, same convention as market_pnl), because
there is no way to distinguish "our resting order was one of several filled
when the book crossed our price" from "the book crossed our price and
somebody else's resting order was the only one filled" -- true queue
position is unknowable from snapshots alone, on ANY L2-snapshot-based
backtest, not just this one. A fill is credited whenever the real market
crosses our price, which is a necessary condition for a real fill, not a
sufficient one.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO / "data" / "raw" / "polyorderbooks_l2_live"

MAX_RELATIVE_SPREAD = 0.3          # same convention as run_mm_proxy_backtest.py
ORDER_NOTIONAL_CAP = 25.0          # same $ convention as MAX_NOTIONAL_PER_TRADE there
MARKOUT_SECONDS = 15.0             # same reaction-latency assumption as MARKOUT_WINDOW_SECONDS there


def load_touch_series(cache_file: Path, token_label: str) -> list[dict]:
    """Loads one cached market's book history for one outcome token --
    already touch-reduced at fetch time into compact
    [epoch_ts, best_bid, best_bid_size, best_ask, best_ask_size] arrays (see
    polyorderbooks_client.reduce_to_touch) -- into a chronologically sorted
    list of {"ts": epoch_seconds, "best_bid":, "best_ask":}. None-valued
    touches (empty side) are kept, not dropped -- l2_market_pnl needs to see
    them to correctly treat those seconds as unquotable."""
    raw = json.loads(cache_file.read_text())
    snaps = raw.get("data", {}).get(token_label, [])
    out = [{"ts": s[0], "best_bid": s[1], "best_ask": s[3]} for s in snaps]
    out.sort(key=lambda r: r["ts"])
    return out


def _mid(best_bid, best_ask):
    if best_bid is None or best_ask is None or best_bid >= best_ask:
        return None  # no two-sided, non-crossed market to quote a fair value from
    return (best_bid + best_ask) / 2.0


def _markout_touch(touches: list[dict], i: int, markout_seconds: float):
    """The touch at (or the first available touch at/after) ts[i] + markout_seconds,
    scanning forward from i -- touches are 1-second-cadence, so this is a
    short scan, not a full re-search."""
    target = touches[i]["ts"] + markout_seconds
    j = i + 1
    n = len(touches)
    while j < n and touches[j]["ts"] < target:
        j += 1
    return touches[j] if j < n else None


def l2_market_pnl(touches: list[dict], half_spread: float, fill_share: float,
                   order_notional_cap: float = ORDER_NOTIONAL_CAP,
                   markout_seconds: float = MARKOUT_SECONDS,
                   max_relative_spread: float = MAX_RELATIVE_SPREAD) -> dict:
    """Quotes half_spread around the REAL mid at every tick with a two-sided
    book; a fill is credited on the ask side when the NEXT tick's real
    best_bid reaches or crosses our quoted ask (symmetric for the bid side)
    -- i.e., the real market's own touch demonstrably traded through our
    price. Markout marks the fill against the REAL mid `markout_seconds`
    later. Ticks with no two-sided market (one side empty, or crossed) are
    skipped for quoting -- they cannot be quoted against, not a gap in the
    model."""
    pnl_best_case = 0.0
    pnl_with_markout = 0.0
    n_captured = 0
    n_quotable_ticks = 0
    n_total_ticks = len(touches)

    for i in range(len(touches) - 1):
        best_bid, best_ask = touches[i]["best_bid"], touches[i]["best_ask"]
        mid = _mid(best_bid, best_ask)
        if mid is None:
            continue
        n_quotable_ticks += 1

        eff_half_spread = min(half_spread, max_relative_spread * mid, max_relative_spread * (1 - mid))
        if eff_half_spread <= 0:
            continue
        our_bid = mid - eff_half_spread
        our_ask = mid + eff_half_spread

        next_bid, next_ask = touches[i + 1]["best_bid"], touches[i + 1]["best_ask"]
        shares = fill_share * (order_notional_cap / mid)

        fills = []
        if next_bid is not None and next_bid >= our_ask:
            fills.append(("ask", our_ask))  # we sold at our_ask
        if next_ask is not None and next_ask <= our_bid:
            fills.append(("bid", our_bid))  # we bought at our_bid

        for side, price in fills:
            pnl_best_case += shares * eff_half_spread
            markout_touch = _markout_touch(touches, i, markout_seconds)
            if markout_touch is not None:
                markout_mid = _mid(markout_touch["best_bid"], markout_touch["best_ask"])
            else:
                markout_mid = None
            if markout_mid is not None:
                # "ask" fill = we sold -> now short -> adverse if price RISES afterward.
                # "bid" fill = we bought -> now long -> adverse if price FALLS afterward.
                adverse = (markout_mid - price) if side == "ask" else (price - markout_mid)
                pnl_with_markout += shares * eff_half_spread - adverse * shares
            else:
                pnl_with_markout += shares * eff_half_spread
            n_captured += 1

    return {
        "pnl_best_case": pnl_best_case,
        "pnl_with_markout": pnl_with_markout,
        "n_captured": n_captured,
        "n_quotable_ticks": n_quotable_ticks,
        "n_total_ticks": n_total_ticks,
        "pct_ticks_quotable": round(n_quotable_ticks / n_total_ticks * 100, 1) if n_total_ticks else None,
    }
