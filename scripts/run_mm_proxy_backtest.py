"""Stylized market-making PnL estimate -- NOT a real backtest.

Real market-making can't be rigorously backtested here: Polymarket's `/book`
endpoint 404s on resolved markets (confirmed live, see
polymarket_final_pct.py), so there is no historical order-book depth to
determine what price a resting quote would actually have been filled at, or
how often. This script instead builds an explicitly toy, parameterized
estimate from the one thing that *is* recoverable historically -- the
public trade-print tape (data-api /trades) -- and states its assumptions as
assumptions, not measurements.

Model: pure spread capture. For a fixed assumed half-spread and an assumed
"fill share" (the fraction of each real trade's size a resting maker quote
is assumed to capture), every captured unit earns exactly the half-spread,
full stop -- no inventory-direction term.

Two earlier, more elaborate versions of this model were tried and rejected
before landing here, and it's worth recording why, since both looked
plausible before the numbers gave them away:
  1. Mark inventory to the market's RESOLVED payout (0 or 1). This let a
     handful of "decays to zero" longshot markets dominate the total with
     pure directional P&L that has nothing to do with market-making --
     $133k of a $428k total from one Florida-senator-appointment longshot.
  2. Mark inventory to the NEXT observed trade print instead. This still
     inflated results ($282k), because consecutive raw trade prints
     naturally alternate between hitting the real bid and the real ask
     (bid-ask bounce) even with a constant fair value -- using that bounce
     as a "mark" double-counts a spread that already exists in the prints,
     on top of the model's own assumed half-spread.
Removing the inventory term entirely avoids both failure modes. It also
means this model is a deliberate BEST CASE: it assumes perfect, instant,
costless flattening of every position, i.e. zero adverse selection. Real
market makers lose money precisely when informed flow moves the market
against inventory they haven't unwound yet; that risk is real and this
number does not include it, which is disclosed prominently in the report
rather than papered over with an unreliable price-based correction.

Reuses the SAME market population and (already disk-cached, no new network
calls) trade tapes as the Final-1% backtest, via fetch_market_trades.
"""
import csv
import datetime as dt
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import polymarket_final_pct as pmf

REPO = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO / "results" / "polymarket_final_pct"

START_BANKROLL = 10000.0
MAX_NOTIONAL_PER_TRADE = 25.0  # caps a single captured trade's size, so one whale print can't dominate

HALF_SPREADS = [0.005, 0.01, 0.02]
FILL_SHARES = [0.05, 0.15, 0.30]
BASE_HALF_SPREAD = 0.01
BASE_FILL_SHARE = 0.15


def load_market_meta():
    """cid -> {resolved_outcome_index, resolution_time, report_bucket}, from
    the existing Final-1% trade sample (same population, zero extra fetches).
    report_bucket (sports/crypto_price/politics/other) is the same keyword
    heuristic used throughout the Final-1% project -- see its own module
    docstring for why it's indicative, not exact."""
    meta = {}
    with open(RESULTS_DIR / "trades_maker.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cid = r["condition_id"]
            if cid in meta:
                continue
            meta[cid] = {
                "resolved_outcome_index": int(r["resolved_outcome_index"]) if r["resolved_outcome_index"] not in ("", None) else None,
                "resolution_time": r["resolution_time"],
                "question": r["question"],
                "report_bucket": r.get("report_bucket") or "other",
            }
    return meta


MAX_RELATIVE_SPREAD = 0.3  # a quoted half-spread can't exceed 30% of the distance to 0 or 1

# Liquidity constraint: cumulative captured notional in one market can't exceed
# this fraction of that market's own total real trade volume, in addition to the
# existing per-trade $25 cap. The per-trade cap alone only stops one whale print
# from dominating -- it does nothing to stop the model from assuming it captured
# a large share of EVERY print all market long, which no passive resting quote
# realistically achieves. 20% is itself an assumption (no real depth data exists
# to calibrate it against, same limitation as everywhere else in this model) but
# it keeps the model from implicitly claiming to have been the dominant source
# of liquidity in a market it never actually quoted in.
MAX_MARKET_VOLUME_SHARE = 0.20

# Adverse selection via markout: a standard technique in real market-making
# performance analysis -- mark each assumed fill against the volume-weighted
# average price of the trades that follow it, not the single next print. A
# single next print is exactly what got the earlier "mark to next print"
# version rejected (see module docstring): consecutive real prints naturally
# alternate between hitting the bid and the ask even with a constant fair
# value, so marking against just one is pure bid-ask bounce, not information.
# Averaging over a window of many subsequent prints cancels that bounce out
# (it alternates and roughly nets to zero over enough trades) while still
# catching genuine directional drift (informed flow moving the market
# persistently one way), which is the actual thing adverse selection is.
#
# Two different windows, reported side by side, because they answer different
# questions:
#   - MARKOUT_WINDOW_TRADES (a trade COUNT): how much does price drift over the
#     next 20 real prints, whatever real time that happens to span. In a fast,
#     liquid market that's seconds; in a slow one it can be hours -- the count
#     doesn't hold real exposure time constant across markets, which is exactly
#     why a real market maker's actual reaction speed isn't represented by it
#     directly (see markout_trades_window_seconds_p50/p90 in the output for the
#     actual, measured real-time span this implied per market -- not assumed).
#   - MARKOUT_WINDOW_SECONDS (a TIME horizon): how much does price drift in the
#     next N real seconds, regardless of how many trades that takes. This is
#     the more realistic proxy for "how long does it take a market maker to
#     notice adverse movement and pull/reprice a quote" -- a fixed reaction
#     latency, not a fixed print count. 15s matches the same "near-immediate
#     execution" assumption used for the Final-1% strategy's own taker-slippage
#     model (VWAP_WINDOW_S in polymarket_final_pct.py).
MARKOUT_WINDOW_TRADES = 20
MARKOUT_WINDOW_SECONDS = 15
MAX_TIME_WINDOW_SCAN = 500  # safety cap on how many trades the time-window scan will walk through

# How does the adverse-selection picture change as the assumed MM reaction
# speed changes? 15s is one point estimate, borrowed from elsewhere in this
# codebase, not derived from anything MM-specific -- this grid stress-tests
# it against faster (5s) and slower (60s, 300s) reactions, at the base
# half_spread/fill_share only (crossing it with the full HALF_SPREADS x
# FILL_SHARES grid too would be a lot of extra compute for a question this
# script doesn't need to answer at every spread/fill combination).
MARKOUT_TIME_GRID_SECONDS = [5, 15, 60, 300]

# "Which markets are better for market making" (report_bucket) turned out to
# really be a question of trading PACE -- politics survives markout because
# it's slow, sports doesn't because it's fast. PACE_QUANTILES buckets markets
# by their own median inter-trade gap (Q1 = fastest/most active .. Q5 =
# slowest) using data-driven quantile edges rather than a hand-picked
# threshold like "slow = 1+ hour between trades", so the cut reflects the
# actual empirical distribution instead of a guess.
PACE_QUANTILES = 5


def _time_window_vwap(sorted_trades: list[dict], i: int, seconds: float):
    """VWAP (and trade count) of real trades within `seconds` after
    sorted_trades[i], scanning forward at most MAX_TIME_WINDOW_SCAN trades as
    a safety valve for pathologically high-frequency markets."""
    t0 = sorted_trades[i]["timestamp"]
    w_notional = 0.0
    w_shares = 0.0
    n = 0
    j = i + 1
    limit = min(len(sorted_trades), i + 1 + MAX_TIME_WINDOW_SCAN)
    while j < limit and sorted_trades[j]["timestamp"] - t0 <= seconds:
        w = sorted_trades[j]
        w_notional += w["size"] * w["price"]
        w_shares += w["size"]
        n += 1
        j += 1
    if w_shares <= 0:
        return None, 0
    return w_notional / w_shares, n


# Maker Rebates program: confirmed directly against docs.polymarket.com
# (fetched live 2026-09) -- makers pay 0% and the exchange redistributes a
# category-specific share of TAKER fees back to makers whose resting orders
# get filled, split proportionally among all makers active in that market
# that day (rebate_pool = rebate_rate% of total taker fees generated in the
# market; your cut = your_fee_equivalent / everyone's fee_equivalent). We
# have no data on competing makers' activity, so REBATE_UPPER_BOUND below
# assumes 100% of the pool (i.e. sole liquidity provider) -- a ceiling, not
# a realistic expectation, and reported as such everywhere it's used.
REBATE_RATE_BY_FEE_CATEGORY = {
    "crypto": 0.20, "sports": 0.15, "politics": 0.25,
    "geopolitics": 0.0, "other": 0.25,
}


def market_pnl(sorted_trades, total_market_volume, half_spread, fill_share,
                markout_window_seconds: float = MARKOUT_WINDOW_SECONDS,
                fee_category: str = None):
    """Returns the existing best-case PnL (zero adverse selection, exactly as
    before) alongside TWO markout-adjusted variants (trade-count window and
    time window -- see the module-level comment above), never silently
    replacing the original number, same as every other before/after
    comparison in this codebase. `sorted_trades` must already be
    chronologically sorted, each a dict with price/size/side/timestamp
    (parsed once per market, not per config, for performance -- see caller).
    `markout_window_seconds` overrides the module default so callers can
    build a sensitivity grid over assumed MM reaction speed (see
    MARKOUT_TIME_GRID_SECONDS) without touching the trade-count window.
    `fee_category` (one of REBATE_RATE_BY_FEE_CATEGORY's keys) is optional
    and additive-only: leaving it None (the default) skips rebate
    computation entirely and returns rebate_upper_bound=None, so every
    existing caller is unaffected.

    A flat, price-independent half_spread is itself unrealistic at the
    extremes this dataset is full of (it's built from markets that crossed
    $0.99+, so the complementary side sits near $0.001-0.01 for most of the
    market's life): a $0.01 absolute spread on a $0.001 token is a 1,000%+
    relative spread, and combined with a fixed-dollar notional cap this
    let captured share counts explode at tiny prices (25/0.001 = 25,000
    shares from one $25 trade) and turned a handful of longshot markets
    into an implied 2,600%+ total return. The effective half-spread used is
    therefore capped at MAX_RELATIVE_SPREAD of the distance to whichever
    boundary (0 or 1) is closer -- still an assumption, but one that keeps
    quoted spreads sane in dollar terms near the extremes instead of
    silently blowing up."""
    pnl_best_case = 0.0
    pnl_with_markout_trades = 0.0
    pnl_with_markout_time = 0.0
    n_captured = 0
    captured_notional = 0.0
    window_span_seconds_sum = 0.0
    n_windows_with_data = 0
    rebate_upper_bound = 0.0 if fee_category is not None else None
    rebate_rate = REBATE_RATE_BY_FEE_CATEGORY.get(fee_category, REBATE_RATE_BY_FEE_CATEGORY["other"]) if fee_category is not None else 0.0
    volume_cap = MAX_MARKET_VOLUME_SHARE * total_market_volume

    for i, t in enumerate(sorted_trades):
        price, size, side = t["price"], t["size"], t["side"]
        eff_half_spread = min(half_spread, MAX_RELATIVE_SPREAD * price, MAX_RELATIVE_SPREAD * (1 - price))
        if eff_half_spread <= 0:
            continue
        shares = min(size * fill_share, MAX_NOTIONAL_PER_TRADE / price)
        if shares <= 0:
            continue
        notional = shares * price
        if captured_notional + notional > volume_cap:
            remaining = volume_cap - captured_notional
            if remaining <= 1e-9:
                break  # liquidity constraint: this market is fully capped, no more capacity at any price
            shares = remaining / price
            notional = remaining

        pnl_best_case += shares * eff_half_spread
        if fee_category is not None:
            # Each captured fill is a taker's print landing on our resting
            # maker order -> it generates a taker fee, a category-specific
            # share of which funds the rebate pool. Uses the SAME confirmed
            # fee formula as the Final-1% strategy (fee = notional *
            # feeRate * (1-price)) -- pmf.taker_fee_frac_of_notional.
            taker_fee = notional * pmf.taker_fee_frac_of_notional(price, fee_category)
            rebate_upper_bound += taker_fee * rebate_rate

        # Captured a SELL print -> we were the resting bid, so we're now long:
        # adverse if the market drifted DOWN after we bought.
        # Captured a BUY print -> we were the resting ask, so we're now short:
        # adverse if the market drifted UP after we sold.
        def _adverse(markout_price):
            return (price - markout_price) * shares if side == "SELL" else (markout_price - price) * shares

        window = sorted_trades[i + 1: i + 1 + MARKOUT_WINDOW_TRADES]
        if window:
            w_shares = sum(w["size"] for w in window)
            markout_price = sum(w["size"] * w["price"] for w in window) / w_shares if w_shares > 0 else price
            pnl_with_markout_trades += shares * eff_half_spread - _adverse(markout_price)
            window_span_seconds_sum += window[-1]["timestamp"] - t["timestamp"]
            n_windows_with_data += 1
        else:
            pnl_with_markout_trades += shares * eff_half_spread  # tail of the tape; spread-only

        time_markout_price, _ = _time_window_vwap(sorted_trades, i, markout_window_seconds)
        if time_markout_price is not None:
            pnl_with_markout_time += shares * eff_half_spread - _adverse(time_markout_price)
        else:
            pnl_with_markout_time += shares * eff_half_spread  # no trades within the window; spread-only

        n_captured += 1
        captured_notional += notional
        if captured_notional >= volume_cap:
            break

    return {
        "pnl_best_case": pnl_best_case,
        "pnl_with_markout_trades": pnl_with_markout_trades,
        "pnl_with_markout_time": pnl_with_markout_time,
        "n_captured": n_captured,
        "captured_notional": captured_notional,
        "volume_share_captured": captured_notional / total_market_volume if total_market_volume else None,
        "avg_trades_window_span_s": window_span_seconds_sum / n_windows_with_data if n_windows_with_data else None,
        "rebate_upper_bound": rebate_upper_bound,
    }


def parse_and_sort_trades(trades: list[dict]) -> tuple[list[dict], float]:
    """Once per market (not once per sensitivity-grid config): parse, filter,
    and chronologically sort the raw trade tape, plus the market's total real
    notional turnover (needed for the liquidity-share cap). Chronological order
    is required for the markout lookahead to mean anything."""
    valid = []
    for t in trades:
        try:
            price = float(t["price"])
            size = float(t["size"])
            side = t["side"]
            ts = float(t.get("timestamp", 0))
        except (KeyError, ValueError, TypeError):
            continue
        if price <= 0 or price >= 1 or size <= 0 or side not in ("BUY", "SELL"):
            continue
        valid.append({"price": price, "size": size, "side": side, "timestamp": ts})
    valid.sort(key=lambda t: t["timestamp"])
    total_market_volume = sum(t["price"] * t["size"] for t in valid)
    return valid, total_market_volume


CONCENTRATION_TOP_NS = [1, 5, 10, 20]


def concentration_by_top_n(per_market: list[dict], pnl_key: str = "pnl", worst: bool = False) -> dict:
    """What fraction of total PnL (under `pnl_key`) comes from the top N
    markets by PnL, for N in CONCENTRATION_TOP_NS -- or, with worst=True, the
    bottom N by PnL (used for the markout-adjusted case, whose total is
    typically negative and dominated by a handful of catastrophic losses, not
    gains -- see module docstring's own worked example, "$133k of a $428k
    total from one Florida-senator-appointment longshot," for why this shape
    of concentration is expected in this model family). Previously only
    discoverable by manually sorting per_market_base_case; surfaced here as a
    first-class number instead."""
    ranked = sorted(per_market, key=lambda r: r[pnl_key] if worst else -r[pnl_key])
    total_pnl = sum(r[pnl_key] for r in ranked)
    out = {"total_pnl": round(total_pnl, 2), "by_top_n": []}
    for n in CONCENTRATION_TOP_NS:
        top_pnl = sum(r[pnl_key] for r in ranked[:n])
        out["by_top_n"].append({
            "n": n,
            "pnl": round(top_pnl, 2),
            "pct_of_total": round(top_pnl / total_pnl * 100, 2) if total_pnl else None,
            "top_markets": [
                {"question": r["question"], "pnl": r[pnl_key], "n_captured_trades": r["n_captured_trades"]}
                for r in ranked[:n]
            ] if n <= 5 else None,  # only list markets by name for the smaller cuts
        })
    return out


def _percentile(values: list[float], p: float):
    """Sorted-index percentile, no numpy dependency (nothing else in this
    script needs it). Returns None on an empty input."""
    if not values:
        return None
    s = sorted(values)
    idx = min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))
    return s[idx]


def category_breakdown(per_market: list[dict]) -> dict:
    """Best-case / trade-window-markout / time-window-markout PnL broken down
    by report_bucket (sports/crypto_price/politics/other) -- this is the
    "which markets are actually better for market making" answer: a bucket
    whose PnL survives the realistic time-window markout is a real
    candidate, one where markout wipes it out (or flips it negative) is not,
    regardless of how good its best-case number looks in isolation."""
    buckets = {}
    for r in per_market:
        b = buckets.setdefault(r["report_bucket"], {
            "n_markets": 0, "pnl_best_case": 0.0,
            "pnl_with_markout_trades": 0.0, "pnl_with_markout_time": 0.0,
            "captured_notional": 0.0,
        })
        b["n_markets"] += 1
        b["pnl_best_case"] += r["pnl"]
        b["pnl_with_markout_trades"] += r["pnl_with_markout_trades"]
        b["pnl_with_markout_time"] += r["pnl_with_markout_time"]
        b["captured_notional"] += r["captured_notional"]
    out = {}
    for name, b in sorted(buckets.items(), key=lambda kv: -kv[1]["pnl_with_markout_time"]):
        out[name] = {
            "n_markets": b["n_markets"],
            "pnl_best_case": round(b["pnl_best_case"], 2),
            "pnl_with_markout_trades": round(b["pnl_with_markout_trades"], 2),
            "pnl_with_markout_time": round(b["pnl_with_markout_time"], 2),
            "captured_notional": round(b["captured_notional"], 2),
            "markout_time_pct_of_best_case": (
                round(b["pnl_with_markout_time"] / b["pnl_best_case"] * 100, 1)
                if b["pnl_best_case"] else None
            ),
        }
    return out


def resolution_epoch_seconds(resolution_time: str):
    """Parses the resolution_time string stored in trades_maker.csv (a
    pandas Timestamp's own str() form, e.g. "2023-04-21 23:03:21+00:00")
    into Unix epoch seconds, comparable directly against a trade's own
    epoch-seconds timestamp. Returns None if it can't be parsed (shouldn't
    happen for a market with a resolved_outcome_index, but never assume)."""
    if not resolution_time:
        return None
    try:
        return dt.datetime.fromisoformat(resolution_time).timestamp()
    except (ValueError, TypeError):
        return None


def filter_trades_excluding_near_resolution(sorted_trades: list[dict], resolution_epoch, exclusion_seconds: float) -> list[dict]:
    """Drops every trade within `exclusion_seconds` of resolution_epoch -- a
    way to test whether a market's modeled PnL is being driven by endgame /
    informational-cascade trading (the actual outcome becoming apparent)
    rather than steady-state market-making conditions. Removed trades are
    dropped from BOTH capture eligibility and markout context, not just
    capture -- the simplest correct reading of "assume every resting quote
    is pulled once the cutoff is crossed, and you've stopped watching this
    market." If resolution_epoch is None (unparseable) or exclusion_seconds
    is 0, no filtering is possible/requested -- returns sorted_trades
    unchanged (not a copy)."""
    if resolution_epoch is None or exclusion_seconds <= 0:
        return sorted_trades
    return [t for t in sorted_trades if resolution_epoch - t["timestamp"] > exclusion_seconds]


def volume_fraction_near_resolution(sorted_trades: list[dict], resolution_epoch, window_seconds: float):
    """What fraction of a market's TOTAL real trade volume (not our modeled
    capture -- the market's own actual activity) happened within
    `window_seconds` of resolution. Independent of any capture assumption,
    so it's a clean diagnostic for "is this market's trading concentrated
    right at the end" regardless of what fill_share/half_spread you'd model
    with. Returns None if resolution_epoch is unparseable or there's no
    volume to measure."""
    if resolution_epoch is None or not sorted_trades:
        return None
    total = sum(t["price"] * t["size"] for t in sorted_trades)
    if total <= 0:
        return None
    near = sum(t["price"] * t["size"] for t in sorted_trades if resolution_epoch - t["timestamp"] <= window_seconds)
    return near / total


def market_pace_seconds(sorted_trades: list[dict]):
    """Median seconds between consecutive real trades on the FULL tape (not
    just captured fills) -- an inherent property of how actively a market
    trades, independent of any assumed MM participation in it. This is the
    direct mechanism behind the category-level pattern (politics slow -> low
    adverse selection, sports fast -> high): category is a proxy for pace,
    pace is the actual thing driving it. Returns None for a market with
    fewer than 2 trades (no gap to measure)."""
    if len(sorted_trades) < 2:
        return None
    gaps = sorted(b["timestamp"] - a["timestamp"] for a, b in zip(sorted_trades, sorted_trades[1:]))
    n = len(gaps)
    mid = n // 2
    return gaps[mid] if n % 2 else (gaps[mid - 1] + gaps[mid]) / 2


def assign_quantile_buckets(per_market: list[dict], field: str, out_field: str,
                             n_quantiles: int = PACE_QUANTILES, prefix: str = "Q") -> None:
    """Generic equal-COUNT quantile bucketing (bucket 1 = lowest `field`,
    bucket n_quantiles = highest) -- data-driven edges over the actual
    population, not a fixed threshold picked by hand. Mutates `per_market` in
    place, writing `out_field`. Rows where `field` is None get "n/a" and are
    excluded from the split itself. Used for pace (assign_pace_buckets below)
    and, identically, for any other per-market numeric characteristic worth
    segmenting by -- e.g. trade volume."""
    eligible = [r for r in per_market if r[field] is not None]
    eligible.sort(key=lambda r: r[field])
    n = len(eligible)
    for idx, r in enumerate(eligible):
        r[out_field] = f"{prefix}{min(n_quantiles - 1, idx * n_quantiles // n) + 1}" if n else "n/a"
    for r in per_market:
        r.setdefault(out_field, "n/a")


def assign_pace_buckets(per_market: list[dict]) -> None:
    """Mutates `per_market` in place, adding a `pace_bucket` field (Q1..Q5,
    Q1 = fastest/most active market, Q5 = slowest) via data-driven quantile
    edges over median_inter_trade_s. Markets with no measurable pace
    (median_inter_trade_s is None, i.e. <2 real trades) get pace_bucket
    "n/a" and are excluded from the quantile split itself."""
    assign_quantile_buckets(per_market, "median_inter_trade_s", "pace_bucket", PACE_QUANTILES, "Q")


def quantile_breakdown(per_market: list[dict], bucket_field: str, range_field: str) -> dict:
    """Same shape as category_breakdown, keyed by whatever quantile bucket
    was assigned to `bucket_field` (by assign_quantile_buckets) instead of
    report_bucket, plus the actual range of `range_field` each bucket spans
    (for interpretability -- "Q5" alone doesn't say anything, "median gap
    38-420 minutes" does). Sorted best-to-worst by pnl_with_markout_time,
    same convention as category_breakdown, so callers can just take the
    first entry as "the best bucket" without assuming the bucketed
    characteristic and profitability are monotonic -- they're expected to
    correlate, not required to."""
    buckets = {}
    for r in per_market:
        if r[bucket_field] == "n/a":
            continue
        b = buckets.setdefault(r[bucket_field], {
            "n_markets": 0, "pnl_best_case": 0.0,
            "pnl_with_markout_trades": 0.0, "pnl_with_markout_time": 0.0,
            "captured_notional": 0.0, "range_samples": [],
        })
        b["n_markets"] += 1
        b["pnl_best_case"] += r["pnl"]
        b["pnl_with_markout_trades"] += r["pnl_with_markout_trades"]
        b["pnl_with_markout_time"] += r["pnl_with_markout_time"]
        b["captured_notional"] += r["captured_notional"]
        b["range_samples"].append(r[range_field])
    out = {}
    for name, b in sorted(buckets.items(), key=lambda kv: -kv[1]["pnl_with_markout_time"]):
        out[name] = {
            "n_markets": b["n_markets"],
            "pnl_best_case": round(b["pnl_best_case"], 2),
            "pnl_with_markout_trades": round(b["pnl_with_markout_trades"], 2),
            "pnl_with_markout_time": round(b["pnl_with_markout_time"], 2),
            "captured_notional": round(b["captured_notional"], 2),
            "markout_time_pct_of_best_case": (
                round(b["pnl_with_markout_time"] / b["pnl_best_case"] * 100, 1)
                if b["pnl_best_case"] else None
            ),
            f"{range_field}_range": [round(min(b["range_samples"]), 1), round(max(b["range_samples"]), 1)],
        }
    return out


def pace_breakdown(per_market: list[dict]) -> dict:
    """quantile_breakdown specialized to pace_bucket / median_inter_trade_s
    -- kept as a named function since it's the one used throughout main()."""
    return quantile_breakdown(per_market, "pace_bucket", "median_inter_trade_s")


def main():
    meta = load_market_meta()
    print(f"[mm-proxy] {len(meta)} markets in the reused Final-1% population, "
          f"all with cached trade tapes on disk already")

    sensitivity = {}
    for hs in HALF_SPREADS:
        for fs in FILL_SHARES:
            sensitivity[f"hs{hs}_fs{fs}"] = {
                "half_spread": hs, "fill_share": fs,
                "total_pnl_best_case": 0.0,
                "total_pnl_with_markout_trades": 0.0,
                "total_pnl_with_markout_time": 0.0,
                "n_markets_active": 0,
            }

    per_market_base = []  # for the base-case equity curve
    n_no_resolved_idx = 0
    n_no_trades = 0
    n_volume_capped = 0        # markets where the liquidity constraint actually bound
    volume_capped_notional = 0.0  # total notional trimmed off by the liquidity constraint (base case)
    window_span_samples = []  # per-market avg real-time span of the MARKOUT_WINDOW_TRADES window, base case
    window_grid_by_market = {}  # cid -> {window_seconds: pnl_with_markout_time}, base half_spread/fill_share only
    for i, (cid, m) in enumerate(meta.items()):
        if m["resolved_outcome_index"] is None:
            n_no_resolved_idx += 1
            continue
        raw_trades = pmf.fetch_market_trades(cid)
        if not raw_trades:
            n_no_trades += 1
            continue
        sorted_trades, total_market_volume = parse_and_sort_trades(raw_trades)
        if not sorted_trades:
            n_no_trades += 1
            continue

        for key, cfg in sensitivity.items():
            r = market_pnl(sorted_trades, total_market_volume, cfg["half_spread"], cfg["fill_share"])
            cfg["total_pnl_best_case"] += r["pnl_best_case"]
            cfg["total_pnl_with_markout_trades"] += r["pnl_with_markout_trades"]
            cfg["total_pnl_with_markout_time"] += r["pnl_with_markout_time"]
            if r["n_captured"] > 0:
                cfg["n_markets_active"] += 1

        base = market_pnl(sorted_trades, total_market_volume, BASE_HALF_SPREAD, BASE_FILL_SHARE)
        if base["n_captured"] > 0:
            uncapped_desired = sum(
                min(t["size"] * BASE_FILL_SHARE, MAX_NOTIONAL_PER_TRADE / t["price"]) * t["price"]
                for t in sorted_trades
            )
            if base["captured_notional"] < uncapped_desired - 1e-6:
                n_volume_capped += 1
                volume_capped_notional += uncapped_desired - base["captured_notional"]
            if base["avg_trades_window_span_s"] is not None:
                window_span_samples.append(base["avg_trades_window_span_s"])
            per_market_base.append({
                "condition_id": cid, "question": m["question"][:80],
                "resolution_time": m["resolution_time"],
                "report_bucket": m["report_bucket"],
                "pnl": round(base["pnl_best_case"], 4),
                "pnl_with_markout_trades": round(base["pnl_with_markout_trades"], 4),
                "pnl_with_markout_time": round(base["pnl_with_markout_time"], 4),
                "n_captured_trades": base["n_captured"],
                "captured_notional": round(base["captured_notional"], 2),
                "total_market_volume": round(total_market_volume, 2),
                "volume_share_captured": round(base["volume_share_captured"], 4) if base["volume_share_captured"] is not None else None,
                "avg_trades_window_span_s": round(base["avg_trades_window_span_s"], 2) if base["avg_trades_window_span_s"] is not None else None,
                "median_inter_trade_s": market_pace_seconds(sorted_trades),
            })
            window_grid_by_market[cid] = {
                w: market_pnl(sorted_trades, total_market_volume, BASE_HALF_SPREAD, BASE_FILL_SHARE,
                               markout_window_seconds=w)["pnl_with_markout_time"]
                for w in MARKOUT_TIME_GRID_SECONDS
            }
        if (i + 1) % 500 == 0:
            print(f"  [mm-proxy] {i+1}/{len(meta)} markets processed ...", flush=True)

    concentration_report = concentration_by_top_n(per_market_base, "pnl")
    concentration_report_markout_trades = concentration_by_top_n(per_market_base, "pnl_with_markout_trades", worst=True)
    concentration_report_markout_time = concentration_by_top_n(per_market_base, "pnl_with_markout_time", worst=True)
    n_markets_markout_trades_negative = sum(1 for r in per_market_base if r["pnl_with_markout_trades"] < 0)
    n_markets_markout_time_negative = sum(1 for r in per_market_base if r["pnl_with_markout_time"] < 0)
    category_report = category_breakdown(per_market_base)

    window_span_median = _percentile(window_span_samples, 50)
    window_span_p90 = _percentile(window_span_samples, 90)
    window_span_mean = sum(window_span_samples) / len(window_span_samples) if window_span_samples else None

    # "Focus on the best markets": bucket by trading pace and find the bucket
    # that actually survives adverse selection best (empirically, not by
    # assuming "slow = best" up front), then check whether restricting to it
    # holds up across the whole reaction-speed grid, not just at 15s.
    assign_pace_buckets(per_market_base)
    pace_report = pace_breakdown(per_market_base)
    best_pace_bucket = next(iter(pace_report), None)
    best_bucket_cids = {r["condition_id"] for r in per_market_base if r["pace_bucket"] == best_pace_bucket}
    best_bucket_categories = {}
    for r in per_market_base:
        if r["pace_bucket"] == best_pace_bucket:
            best_bucket_categories[r["report_bucket"]] = best_bucket_categories.get(r["report_bucket"], 0) + 1

    time_window_sensitivity = []
    for w in MARKOUT_TIME_GRID_SECONDS:
        total_all = sum(window_grid_by_market[r["condition_id"]][w] for r in per_market_base)
        total_best = sum(
            window_grid_by_market[cid][w] for cid in best_bucket_cids
        )
        time_window_sensitivity.append({
            "window_seconds": w,
            "total_pnl_all_markets": round(total_all, 2),
            "total_pnl_best_pace_bucket": round(total_best, 2),
            "n_markets_best_pace_bucket": len(best_bucket_cids),
        })

    per_market_base.sort(key=lambda r: r["resolution_time"])
    equity = START_BANKROLL
    equity_markout_trades = START_BANKROLL
    equity_markout_time = START_BANKROLL
    curve = [(
        per_market_base[0]["resolution_time"][:10] if per_market_base else None,
        equity, equity_markout_trades, equity_markout_time,
    )]
    for r in per_market_base:
        equity += r["pnl"]
        equity_markout_trades += r["pnl_with_markout_trades"]
        equity_markout_time += r["pnl_with_markout_time"]
        curve.append((
            r["resolution_time"][:10], round(equity, 2),
            round(equity_markout_trades, 2), round(equity_markout_time, 2),
        ))

    base_key = f"hs{BASE_HALF_SPREAD}_fs{BASE_FILL_SHARE}"
    base_cfg = sensitivity[base_key]

    summary = {
        "n_markets_total": len(meta),
        "n_markets_no_resolution": n_no_resolved_idx,
        "n_markets_no_trades": n_no_trades,
        "n_markets_with_captured_flow_base_case": base_cfg["n_markets_active"],
        "start_bankroll": START_BANKROLL,
        "max_notional_per_trade": MAX_NOTIONAL_PER_TRADE,
        "max_market_volume_share": MAX_MARKET_VOLUME_SHARE,
        "markout_window_trades": MARKOUT_WINDOW_TRADES,
        "markout_window_seconds": MARKOUT_WINDOW_SECONDS,
        "markout_trades_window_actual_span_seconds": {
            "median": round(window_span_median, 2) if window_span_median is not None else None,
            "p90": round(window_span_p90, 2) if window_span_p90 is not None else None,
            "mean": round(window_span_mean, 2) if window_span_mean is not None else None,
            "n_samples": len(window_span_samples),
        },
        "base_case": {"half_spread": BASE_HALF_SPREAD, "fill_share": BASE_FILL_SHARE},
        "final_equity_base_case": round(equity, 2),
        "final_equity_with_markout_trades": round(equity_markout_trades, 2),
        "final_equity_with_markout_time": round(equity_markout_time, 2),
        "total_pnl_base_case": round(base_cfg["total_pnl_best_case"], 2),
        "total_pnl_with_markout_trades": round(base_cfg["total_pnl_with_markout_trades"], 2),
        "total_pnl_with_markout_time": round(base_cfg["total_pnl_with_markout_time"], 2),
        "total_return_pct_base_case": round(base_cfg["total_pnl_best_case"] / START_BANKROLL * 100, 2),
        "total_return_pct_with_markout_trades": round(base_cfg["total_pnl_with_markout_trades"] / START_BANKROLL * 100, 2),
        "total_return_pct_with_markout_time": round(base_cfg["total_pnl_with_markout_time"] / START_BANKROLL * 100, 2),
        "liquidity_constraint": {
            "n_markets_volume_capped": n_volume_capped,
            "notional_trimmed_by_cap": round(volume_capped_notional, 2),
        },
        "sensitivity_grid": list(sensitivity.values()),
        "concentration_base_case": concentration_report,
        "concentration_with_markout_trades_worst": concentration_report_markout_trades,
        "concentration_with_markout_time_worst": concentration_report_markout_time,
        "n_markets_markout_trades_negative": n_markets_markout_trades_negative,
        "n_markets_markout_time_negative": n_markets_markout_time_negative,
        "pct_markets_markout_trades_negative": round(n_markets_markout_trades_negative / len(per_market_base) * 100, 1) if per_market_base else None,
        "pct_markets_markout_time_negative": round(n_markets_markout_time_negative / len(per_market_base) * 100, 1) if per_market_base else None,
        "category_breakdown": category_report,
        "pace_breakdown": pace_report,
        "best_pace_bucket": best_pace_bucket,
        "best_pace_bucket_category_composition": best_bucket_categories,
        "markout_time_grid_seconds": MARKOUT_TIME_GRID_SECONDS,
        "time_window_sensitivity": time_window_sensitivity,
        "equity_curve_base_case": curve,
        "per_market_base_case": per_market_base,
    }

    print(f"\n=== Stylized MM proxy, base case (half_spread=${BASE_HALF_SPREAD}, fill_share={BASE_FILL_SHARE:.0%}) ===")
    print(f"{base_cfg['n_markets_active']} of {len(meta)} markets had any captured flow")
    print(f"Best case (zero adverse selection):                    "
          f"${base_cfg['total_pnl_best_case']:>12,.2f}  ->  ${equity:,.2f} ({summary['total_return_pct_base_case']:+.2f}%, NOT compounded)")
    print(f"Markout, {MARKOUT_WINDOW_TRADES}-trade window:                          "
          f"${base_cfg['total_pnl_with_markout_trades']:>12,.2f}  ->  ${equity_markout_trades:,.2f} ({summary['total_return_pct_with_markout_trades']:+.2f}%, NOT compounded)")
    print(f"Markout, {MARKOUT_WINDOW_SECONDS}s time window (fast MM reaction, new):       "
          f"${base_cfg['total_pnl_with_markout_time']:>12,.2f}  ->  ${equity_markout_time:,.2f} ({summary['total_return_pct_with_markout_time']:+.2f}%, NOT compounded)")

    if window_span_samples:
        print(f"\nHow long does a {MARKOUT_WINDOW_TRADES}-trade window actually span in real time? "
              f"(measured per market, not assumed)")
        print(f"  median {window_span_median:,.1f}s   mean {window_span_mean:,.1f}s   p90 {window_span_p90:,.1f}s   "
              f"(n={len(window_span_samples)} markets with captured flow)")

    if base_cfg['total_pnl_best_case']:
        print(f"\nAdverse selection cost, {MARKOUT_WINDOW_SECONDS}s time window (the realistic fast-MM number): "
              f"${base_cfg['total_pnl_best_case'] - base_cfg['total_pnl_with_markout_time']:,.2f} "
              f"({(1 - base_cfg['total_pnl_with_markout_time']/base_cfg['total_pnl_best_case'])*100:.1f}% of the best-case number)")
    print(f"\nLiquidity constraint (cap captured notional at {MAX_MARKET_VOLUME_SHARE:.0%} of each market's own real volume): "
          f"bound in {n_volume_capped} market(s), trimming ${volume_capped_notional:,.2f} of assumed captured notional "
          f"that the per-trade cap alone would have allowed.")
    print("\nSensitivity grid (total PnL, $ -- best case | markout trades | markout time):")
    for key, cfg in sensitivity.items():
        print(f"  half_spread=${cfg['half_spread']:<6} fill_share={cfg['fill_share']:<6.0%}  "
              f"best_case=${cfg['total_pnl_best_case']:>10,.2f}  "
              f"markout_trades=${cfg['total_pnl_with_markout_trades']:>10,.2f}  "
              f"markout_time=${cfg['total_pnl_with_markout_time']:>10,.2f}  "
              f"active_markets={cfg['n_markets_active']}")
    print("\nPnL concentration, best case (this model is NOT diversified spread capture -- check before trusting the headline number):")
    for row in concentration_report["by_top_n"]:
        print(f"  top {row['n']:<3} market(s): ${row['pnl']:>10,.2f}  ({row['pct_of_total']}% of total)")
    if concentration_report["by_top_n"][0]["top_markets"]:
        top1 = concentration_report["by_top_n"][0]["top_markets"][0]
        print(f"  #1 contributor: {top1['question']!r} (${top1['pnl']:,.2f}, {top1['n_captured_trades']} captured trades)")

    print(f"\nBy market category, {MARKOUT_WINDOW_SECONDS}s markout PnL (sorted best to worst -- which markets are actually good for MM):")
    for name, b in category_report.items():
        pct = f"{b['markout_time_pct_of_best_case']}%" if b['markout_time_pct_of_best_case'] is not None else "n/a"
        print(f"  {name:<15} n_markets={b['n_markets']:<5} best_case=${b['pnl_best_case']:>10,.2f}  "
              f"markout_time=${b['pnl_with_markout_time']:>10,.2f}  ({pct} of best case)")

    print(f"\nBy trading PACE (median seconds between real trades, Q1=fastest .. Q5=slowest, "
          f"{MARKOUT_WINDOW_SECONDS}s markout PnL, sorted best to worst):")
    for name, b in pace_report.items():
        pct = f"{b['markout_time_pct_of_best_case']}%" if b['markout_time_pct_of_best_case'] is not None else "n/a"
        lo, hi = b["median_inter_trade_s_range"]
        print(f"  {name}  n_markets={b['n_markets']:<5} gap={lo:>8,.1f}s-{hi:>9,.1f}s  "
              f"best_case=${b['pnl_best_case']:>10,.2f}  markout_time=${b['pnl_with_markout_time']:>10,.2f}  ({pct} of best case)")
    if best_pace_bucket:
        cat_str = ", ".join(f"{k}={v}" for k, v in sorted(best_bucket_categories.items(), key=lambda kv: -kv[1]))
        print(f"  Best bucket: {best_pace_bucket} ({len(best_bucket_cids)} markets) -- category mix: {cat_str}")

    print(f"\nFocus on the best markets: total PnL, all markets vs. restricted to {best_pace_bucket} only, "
          f"across reaction-speed assumptions (this is the actual answer to 'does selecting better markets help, "
          f"and does that hold up regardless of how fast you assume you can react'):")
    for row in time_window_sensitivity:
        print(f"  window={row['window_seconds']:>4}s   all_markets=${row['total_pnl_all_markets']:>12,.2f}   "
              f"{best_pace_bucket}_only=${row['total_pnl_best_pace_bucket']:>12,.2f}  "
              f"(n={row['n_markets_best_pace_bucket']} markets)")

    print(f"\n{n_markets_markout_time_negative} of {len(per_market_base)} markets ({summary['pct_markets_markout_time_negative']}%) "
          f"have NEGATIVE {MARKOUT_WINDOW_SECONDS}s-markout PnL -- this is not just a few outliers, adverse selection hurts broadly:")
    print("PnL concentration, worst markout losses, time window (bottom N markets' share of the total markout loss):")
    for row in concentration_report_markout_time["by_top_n"]:
        print(f"  worst {row['n']:<3} market(s): ${row['pnl']:>12,.2f}  ({row['pct_of_total']}% of total loss)")
    if concentration_report_markout_time["by_top_n"][0]["top_markets"]:
        w1 = concentration_report_markout_time["by_top_n"][0]["top_markets"][0]
        print(f"  #1 worst: {w1['question']!r} (${w1['pnl']:,.2f}, {w1['n_captured_trades']} captured trades)")

    out_path = RESULTS_DIR / "mm_proxy_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
