"""Unit tests for the Polymarket final-1% backtest, on synthetic data.

No network calls: every test builds its own price series / market dicts
and exercises the signal, fee, backtest, and statistics functions in
isolation.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from polymarket_final_pct import (
    BacktestConfig,
    FillAssumptions,
    GasAssumptions,
    SignalConfig,
    _dedupe_by_id,
    categorize_flip,
    category_breakdown,
    classify_fee_category,
    classify_report_bucket,
    clopper_pearson_interval,
    compute_metrics,
    compute_with_vs_without_flips,
    days_to_resolution_distribution,
    detect_crossing,
    estimate_vwap_fill,
    maker_fee_frac_of_notional,
    max_days_to_resolution_variant,
    report_bucket_coverage,
    resolved_outcome_index,
    simulate_trade,
    stratified_sample_markets,
    taker_fee_frac_of_notional,
    wilson_interval,
    write_report,
)


def price_df(prices: list[float], start_t: int = 1_700_000_000, step: int = 60) -> pd.DataFrame:
    t = [start_t + i * step for i in range(len(prices))]
    return pd.DataFrame({"t": t, "p": prices})


# ---------------------------------------------------------------------------
# Crossing detection / no lookahead
# ---------------------------------------------------------------------------

def test_no_crossing_below_threshold():
    df = price_df([0.5, 0.8, 0.95, 0.97, 0.98])
    assert detect_crossing(df, threshold=0.99, n_consecutive=3) is None


def test_single_noisy_tick_does_not_trigger():
    df = price_df([0.5, 0.6, 0.995, 0.7, 0.8])  # one-off spike, not persistent
    assert detect_crossing(df, threshold=0.99, n_consecutive=3) is None


def test_crossing_requires_n_consecutive_and_uses_actual_price():
    df = price_df([0.5, 0.6, 0.991, 0.993, 0.997, 0.6])
    hit = detect_crossing(df, threshold=0.99, n_consecutive=3)
    assert hit is not None
    # confirmed on the 3rd consecutive qualifying snapshot (index 4), at ITS
    # actual price, not a hypothetical fill at exactly 0.99
    assert hit["entry_idx"] == 4
    assert hit["entry_price"] == pytest.approx(0.997)


def test_crossing_fires_at_earliest_qualifying_run_not_a_later_one():
    df = price_df([0.5, 0.991, 0.992, 0.993, 0.5, 0.994, 0.995, 0.996])
    hit = detect_crossing(df, threshold=0.99, n_consecutive=3)
    assert hit["entry_idx"] == 3  # first run, not the second later run at idx 7


def test_n_consecutive_is_configurable():
    df = price_df([0.995, 0.996])
    assert detect_crossing(df, threshold=0.99, n_consecutive=3) is None
    hit = detect_crossing(df, threshold=0.99, n_consecutive=2)
    assert hit is not None and hit["entry_idx"] == 1


def test_empty_series_returns_none():
    assert detect_crossing(price_df([]), threshold=0.99, n_consecutive=3) is None


# ---------------------------------------------------------------------------
# Multi-episode coarse-to-fine zoom (fetch_token_lifetime_prices)
# ---------------------------------------------------------------------------

def test_approach_episode_starts_finds_every_contiguous_run():
    import polymarket_final_pct as pmf
    coarse = pd.DataFrame({
        "t": [0, 3600, 7200, 10800, 14400, 18000],
        "p": [0.98, 0.99, 0.50, 0.30, 0.97, 0.98],
    })
    # two contiguous runs >= 0.97: [t=0,3600] and [t=14400,18000]
    assert pmf._approach_episode_starts(coarse, 0.97) == [0, 14400]


def test_approach_episode_starts_empty_when_never_approaches():
    import polymarket_final_pct as pmf
    coarse = pd.DataFrame({"t": [0, 3600], "p": [0.5, 0.6]})
    assert pmf._approach_episode_starts(coarse, 0.97) == []


def test_fetch_token_lifetime_prices_short_lifetime_uses_fine_direct(monkeypatch):
    import polymarket_final_pct as pmf
    fine = price_df([0.5, 0.99])
    monkeypatch.setattr(pmf, "fetch_price_series", lambda token_id, s, e, fidelity: fine)
    df, source = pmf.fetch_token_lifetime_prices("tok", 0, 5 * 86400)  # <= 15-day window
    assert source == "fine_direct"
    assert df.equals(fine)


def test_fetch_token_lifetime_prices_never_approaching_stays_coarse_only(monkeypatch):
    # No fine-grained call should be made at all when the coarse series never
    # gets close to the threshold -- verified by making any such call raise.
    import polymarket_final_pct as pmf
    coarse = pd.DataFrame({"t": [0, 86400, 2 * 86400], "p": [0.3, 0.4, 0.5]})

    def fake(token_id, s, e, fidelity):
        if fidelity == pmf.COARSE_FIDELITY_MIN:
            return coarse
        raise AssertionError("should never zoom in when coarse never approaches the threshold")

    monkeypatch.setattr(pmf, "fetch_price_series", fake)
    df, source = pmf.fetch_token_lifetime_prices("tok", 0, 40 * 86400)
    assert source == "coarse_only"
    assert df.equals(coarse)


def test_fetch_token_lifetime_prices_finds_crossing_in_a_later_approach_episode(monkeypatch):
    """Regression test: a market that approaches the threshold early, retreats,
    and only truly crosses much later in its life used to be silently missed,
    because the old zoom logic only ever looked at the FIRST approach episode's
    15-day window. Real, plausible shape: a long-running market that flirts
    with favorite status, cools off, then becomes the real favorite months
    later."""
    import polymarket_final_pct as pmf

    early_center = 1 * 86400
    late_center = 30 * 86400
    coarse = pd.DataFrame({
        "t": [early_center, 2 * 86400, late_center, late_center + 3600],
        "p": [0.98, 0.50, 0.98, 0.98],
    })
    early_zoom = price_df([0.98, 0.985, 0.97], start_t=early_center, step=60)  # approaches, never crosses
    late_zoom = price_df([0.991, 0.992, 0.993], start_t=late_center, step=60)  # the real crossing

    def fake(token_id, s, e, fidelity):
        if fidelity == pmf.COARSE_FIDELITY_MIN:
            return coarse
        if s <= early_center <= e:
            return early_zoom
        if s <= late_center <= e:
            return late_zoom
        raise AssertionError(f"unexpected fine-grained window [{s}, {e}]")

    monkeypatch.setattr(pmf, "fetch_price_series", fake)
    df, source = pmf.fetch_token_lifetime_prices("tok", 0, 40 * 86400)
    assert source == "fine_zoom"

    hit = detect_crossing(df, threshold=0.99, n_consecutive=3)
    assert hit is not None
    assert hit["entry_time_s"] == late_center + 2 * 60  # 3rd confirming snapshot of the LATE episode


def test_fetch_token_lifetime_prices_truncates_beyond_max_zoom_episodes(monkeypatch):
    import polymarket_final_pct as pmf
    n_episodes = pmf.MAX_ZOOM_EPISODES + 3
    coarse_rows = []
    for i in range(n_episodes):
        center = i * 5 * 86400
        coarse_rows.append({"t": center, "p": 0.98})
        coarse_rows.append({"t": center + 3600, "p": 0.30})  # retreat, so each is its own episode
    coarse = pd.DataFrame(coarse_rows)

    def fake(token_id, s, e, fidelity):
        if fidelity == pmf.COARSE_FIDELITY_MIN:
            return coarse
        return price_df([0.98], start_t=s)  # never actually crosses; only episode count matters here

    monkeypatch.setattr(pmf, "fetch_price_series", fake)
    df, source = pmf.fetch_token_lifetime_prices("tok", 0, n_episodes * 5 * 86400)
    assert source == "fine_zoom_truncated"


# ---------------------------------------------------------------------------
# Fees (confirmed formula: fee = shares * feeRate * p * (1-p); maker == 0)
# ---------------------------------------------------------------------------

def test_maker_fee_is_always_zero():
    assert maker_fee_frac_of_notional(0.99, "crypto") == 0.0
    assert maker_fee_frac_of_notional(0.50, "sports") == 0.0


def test_taker_fee_matches_confirmed_worked_examples():
    # docs' own worked examples: 100 shares @ $0.50 -> crypto $1.75, sports
    # $1.25, politics $1.00. fee_frac_of_notional = feeRate * (1-p); dollar
    # fee = frac * notional = frac * (100 * 0.50).
    for category, expected_dollar_fee in [("crypto", 1.75), ("sports", 1.25), ("politics", 1.00)]:
        frac = taker_fee_frac_of_notional(0.50, category)
        notional = 100 * 0.50
        assert frac * notional == pytest.approx(expected_dollar_fee, abs=1e-9)


def test_taker_fee_shrinks_near_the_extreme():
    frac_mid = taker_fee_frac_of_notional(0.50, "crypto")
    frac_extreme = taker_fee_frac_of_notional(0.99, "crypto")
    assert frac_extreme < frac_mid
    assert frac_extreme == pytest.approx(0.07 * 0.01)


def test_gas_sponsored_vs_non_relayed():
    assert GasAssumptions(relayer_sponsored=True).cost_usd() == 0.0
    g = GasAssumptions(relayer_sponsored=False, non_relayed_cost_usd_per_trade=0.0042)
    assert g.cost_usd() == pytest.approx(0.0042)


# ---------------------------------------------------------------------------
# Category classification
# ---------------------------------------------------------------------------

def test_classify_report_bucket_politics():
    m = {"question": "Will the Democrat nominee win the presidential election?", "slug": "x", "events": []}
    assert classify_report_bucket(m) == "politics"


def test_classify_report_bucket_sports():
    m = {"question": "Lakers vs Celtics: who wins Game 7?", "slug": "x", "events": []}
    assert classify_report_bucket(m) == "sports"


def test_classify_report_bucket_crypto_price():
    m = {"question": "Bitcoin Up or Down - August 25, 8:25AM-8:30AM ET", "slug": "btc-updown", "events": []}
    assert classify_report_bucket(m) == "crypto_price"


def test_classify_fee_category_maps_to_official_taxonomy():
    m = {"question": "Will Bitcoin hit $100k?", "slug": "x", "events": []}
    assert classify_fee_category(m) == "crypto"


def test_report_bucket_coverage_counts_every_bucket_including_zero():
    markets = [
        {"question": "Lakers vs Celtics: who wins Game 7?", "slug": "x", "events": []},
        {"question": "Will Bitcoin hit $100k?", "slug": "x", "events": []},
        {"question": "Will it rain in Boston tomorrow?", "slug": "x", "events": []},  # -> other
    ]
    coverage = report_bucket_coverage(markets)
    assert coverage == {"politics": 0, "sports": 1, "crypto_price": 1, "other": 1}
    assert sum(coverage.values()) == len(markets)


# ---------------------------------------------------------------------------
# Resolved outcome parsing (must never feed back into signal generation --
# tested structurally: detect_crossing above takes no market/outcome data)
# ---------------------------------------------------------------------------

def test_resolved_outcome_index_from_outcome_prices():
    assert resolved_outcome_index({"outcomePrices": json.dumps(["1", "0"])}) == 0
    assert resolved_outcome_index({"outcomePrices": json.dumps(["0", "1"])}) == 1


def test_resolved_outcome_index_none_when_ambiguous():
    assert resolved_outcome_index({"outcomePrices": json.dumps(["0", "0"])}) is None
    assert resolved_outcome_index({"outcomePrices": "[]"}) is None


# ---------------------------------------------------------------------------
# Trade simulation: fees, gas, depth cap, payout
# ---------------------------------------------------------------------------

def _market(question="Will Bitcoin hit $100k?", outcome_prices=("0", "1")):
    return {
        "id": "1", "conditionId": "0xabc", "question": question,
        "outcomePrices": json.dumps(list(outcome_prices)),
        "endDate": "2024-01-10T00:00:00Z", "closedTime": "2024-01-10T00:00:00Z",
        "slug": "x", "events": [],
    }


def _crossing(entry_price=0.995, outcome_index=1, entry_time_s=1_704_800_000):
    return {
        "token_id": "tok1", "outcome_index": outcome_index, "outcome_label": "Yes",
        "entry_time_s": entry_time_s, "entry_price": entry_price,
        "data_source": "fine_direct", "days_to_scheduled_end_at_entry": 1.0,
    }


def test_winning_trade_payout_and_pnl():
    cfg = BacktestConfig(signal=SignalConfig(), position_notional=100.0, gas=GasAssumptions(relayer_sponsored=True))
    fill = FillAssumptions(fill_type="maker")
    t = simulate_trade(_crossing(entry_price=0.99, outcome_index=1), _market(), fill, cfg, cap_shares=None)
    assert t["won"] is True
    shares = 100.0 / 0.99
    assert t["shares"] == pytest.approx(shares)
    assert t["payout"] == pytest.approx(shares * 1.0)
    assert t["fee_cost"] == 0.0  # maker
    assert t["pnl_gross"] == pytest.approx(shares - 100.0)
    assert t["pnl_net"] == pytest.approx(t["pnl_gross"])  # no fee, no gas


def test_flipped_trade_loses_full_notional():
    cfg = BacktestConfig(signal=SignalConfig(), position_notional=100.0, gas=GasAssumptions(relayer_sponsored=True))
    fill = FillAssumptions(fill_type="maker")
    # entered on outcome_index=0 but outcome_index=1 resolved -> flip
    t = simulate_trade(_crossing(entry_price=0.99, outcome_index=0), _market(outcome_prices=("0", "1")), fill, cfg, cap_shares=None)
    assert t["won"] is False
    assert t["payout"] == 0.0
    assert t["pnl_net"] == pytest.approx(-t["notional"])


def test_taker_fee_reduces_net_pnl_vs_maker():
    cfg = BacktestConfig(signal=SignalConfig(), position_notional=100.0, gas=GasAssumptions(relayer_sponsored=True))
    maker_t = simulate_trade(_crossing(entry_price=0.99, outcome_index=1), _market(), FillAssumptions("maker"), cfg, cap_shares=None)
    taker_t = simulate_trade(_crossing(entry_price=0.99, outcome_index=1), _market(), FillAssumptions("taker"), cfg, cap_shares=None)
    assert taker_t["fee_cost"] > 0
    assert taker_t["pnl_net"] < maker_t["pnl_net"]


def test_depth_cap_reduces_position_size():
    cfg = BacktestConfig(signal=SignalConfig(), position_notional=100.0, gas=GasAssumptions(relayer_sponsored=True))
    fill = FillAssumptions(fill_type="maker")
    desired_shares = 100.0 / 0.99
    t = simulate_trade(_crossing(entry_price=0.99, outcome_index=1), _market(), fill, cfg, cap_shares=desired_shares / 2)
    assert t["depth_capped"] is True
    assert t["shares"] == pytest.approx(desired_shares / 2)
    assert t["notional"] < 100.0


def test_non_relayed_gas_charged_per_trade():
    cfg = BacktestConfig(
        signal=SignalConfig(), position_notional=100.0,
        gas=GasAssumptions(relayer_sponsored=False, non_relayed_cost_usd_per_trade=0.01),
    )
    fill = FillAssumptions(fill_type="maker")
    t = simulate_trade(_crossing(entry_price=0.99, outcome_index=1), _market(), fill, cfg, cap_shares=None)
    assert t["gas_cost"] == pytest.approx(0.01)
    assert t["pnl_net"] == pytest.approx(t["pnl_gross"] - 0.01)


# ---------------------------------------------------------------------------
# Metrics: annualized return uses actual per-trade holding period
# ---------------------------------------------------------------------------

def _trades_df(rows):
    return pd.DataFrame(rows)


def test_annualized_return_uses_actual_holding_period_not_fixed_assumption():
    # two trades, same $ pnl, different holding periods -> different
    # annualized return (dollar-year-weighted), same total return.
    rows = [
        {"notional": 100.0, "pnl_net": 1.0, "pnl_gross": 1.0, "holding_days": 1.0, "won": True},
        {"notional": 100.0, "pnl_net": 1.0, "pnl_gross": 1.0, "holding_days": 100.0, "won": True},
    ]
    m = compute_metrics(_trades_df(rows))
    assert m["total_return"] == pytest.approx(2.0 / 200.0)
    # short-holding trade contributes much more annualized return per dollar-year
    dollar_years = (100 * 1 / 365.0) + (100 * 100 / 365.0)
    assert m["annualized_return"] == pytest.approx(2.0 / dollar_years)


def test_win_rate_and_flip_rate():
    rows = [
        {"notional": 100.0, "pnl_net": 1.0, "pnl_gross": 1.0, "holding_days": 1.0, "won": True},
        {"notional": 100.0, "pnl_net": 1.0, "pnl_gross": 1.0, "holding_days": 1.0, "won": True},
        {"notional": 100.0, "pnl_net": -100.0, "pnl_gross": -100.0, "holding_days": 1.0, "won": False},
    ]
    m = compute_metrics(_trades_df(rows))
    assert m["n_trades"] == 3
    assert m["n_flips"] == 1
    assert m["win_rate"] == pytest.approx(2 / 3)
    assert m["flip_rate"] == pytest.approx(1 / 3)


def test_with_vs_without_flips_isolates_flip_damage():
    rows = [
        {"notional": 100.0, "pnl_net": 1.0, "pnl_gross": 1.0, "holding_days": 1.0, "won": True},
        {"notional": 100.0, "pnl_net": -100.0, "pnl_gross": -100.0, "holding_days": 1.0, "won": False},
    ]
    wv = compute_with_vs_without_flips(_trades_df(rows))
    assert wv["with_flips"]["total_pnl"] == pytest.approx(-99.0)
    assert wv["without_flips_ie_winners_only"]["total_pnl"] == pytest.approx(1.0)


def test_empty_trades_df_handled_gracefully():
    m = compute_metrics(pd.DataFrame())
    assert m["n_trades"] == 0
    assert math.isnan(m["win_rate"])


# ---------------------------------------------------------------------------
# Confidence intervals on the flip rate
# ---------------------------------------------------------------------------

def test_ci_is_wide_for_small_sample_zero_flips():
    lo, hi = wilson_interval(0, 50)
    assert lo == pytest.approx(0.0, abs=1e-9)
    assert hi > 0.05  # can't rule out a meaningfully nonzero true rate from 0/50


def test_ci_narrows_with_more_trades_same_rate():
    lo_small, hi_small = wilson_interval(2, 100)
    lo_big, hi_big = wilson_interval(20, 1000)
    assert (hi_small - lo_small) > (hi_big - lo_big)


def test_clopper_pearson_is_at_least_as_wide_as_wilson_for_rare_events():
    # exact CI is the conservative standard choice for small counts
    w_lo, w_hi = wilson_interval(2, 3000)
    cp_lo, cp_hi = clopper_pearson_interval(2, 3000)
    assert cp_hi - cp_lo >= w_hi - w_lo - 1e-6


def test_ci_bounds_are_valid_probabilities():
    for k, n in [(0, 10), (5, 5), (2, 3000), (1500, 3000)]:
        for lo, hi in [wilson_interval(k, n), clopper_pearson_interval(k, n)]:
            assert 0.0 <= lo <= hi <= 1.0


# ---------------------------------------------------------------------------
# Flip categorization heuristic
# ---------------------------------------------------------------------------

def test_categorize_flip_flags_disputed_status():
    m = {"id": "1", "question": "x", "umaResolutionStatus": "disputed", "umaResolutionStatuses": "[]"}
    assert categorize_flip(m)["heuristic_category"] == "disputed_resolution"


def test_categorize_flip_defaults_to_manual_review():
    m = {"id": "1", "question": "x", "umaResolutionStatus": "resolved", "umaResolutionStatuses": "[]"}
    assert categorize_flip(m)["heuristic_category"] == "needs_manual_review"


# ---------------------------------------------------------------------------
# Stratified sampling -- stability under a growing census
# ---------------------------------------------------------------------------
# Regression coverage for a real bug found by hand: two threshold-sweep runs
# a day apart, both with the documented fixed seed=42, shared only ~1% of
# their sampled markets because (a) fetch_resolved_markets_census's dedup
# preserved non-deterministic ThreadPoolExecutor completion order, and (b) a
# single RNG stream shared sequentially across the groupby loop meant any
# upstream group's size changing (new markets resolving elsewhere) desynced
# every later group's draw. The fix: dedupe sorted by id, and a per-group
# seed with shuffle-then-slice (prefix-stable as each group's own k drifts).

def _mk_market(i: int, end_date: str, bucket_hint: str) -> dict:
    return {
        "id": str(i), "question": f"{bucket_hint} market {i}", "slug": f"m-{i}",
        "endDate": end_date, "startDate": "2023-01-01T00:00:00Z", "events": [],
    }


def test_dedupe_by_id_is_order_independent():
    a = {"id": "1", "v": "a"}
    b = {"id": "2", "v": "b"}
    assert _dedupe_by_id([a, b]) == _dedupe_by_id([b, a])


def test_stratified_sample_is_deterministic_for_a_fixed_census():
    census = [_mk_market(i, "2024-02-15T00:00:00Z", "nba game") for i in range(200)]
    s1 = sorted(m["id"] for m in stratified_sample_markets(census, n_target=50))
    s2 = sorted(m["id"] for m in stratified_sample_markets(census, n_target=50))
    assert s1 == s2


def test_estimate_vwap_fill_walks_the_tape_in_price_order(monkeypatch):
    import polymarket_final_pct as pmf
    trades = [
        {"asset": "tok", "side": "BUY", "timestamp": 1000, "price": 0.99, "size": 40.0},
        {"asset": "tok", "side": "BUY", "timestamp": 1010, "price": 0.995, "size": 40.0},
        {"asset": "tok", "side": "BUY", "timestamp": 1020, "price": 0.999, "size": 40.0},
        {"asset": "other-tok", "side": "BUY", "timestamp": 1005, "price": 0.5, "size": 1000.0},  # different token, ignored
        {"asset": "tok", "side": "SELL", "timestamp": 1005, "price": 0.98, "size": 1000.0},  # sell side, ignored
        {"asset": "tok", "side": "BUY", "timestamp": 990, "price": 0.98, "size": 1000.0},  # before entry, ignored
    ]
    monkeypatch.setattr(pmf, "fetch_market_trades", lambda condition_id: trades)

    r = estimate_vwap_fill("cid", "tok", entry_time_s=1000, entry_price=0.99, desired_shares=60.0)
    assert r is not None
    # fills 40 @ 0.99 then 20 @ 0.995, in chronological (not price) order
    expected_vwap = (40 * 0.99 + 20 * 0.995) / 60
    assert r["vwap"] == pytest.approx(expected_vwap)
    assert r["filled_shares"] == pytest.approx(60.0)
    assert r["fill_ratio"] == pytest.approx(1.0)


def test_estimate_vwap_fill_reports_partial_fill_ratio(monkeypatch):
    import polymarket_final_pct as pmf
    trades = [{"asset": "tok", "side": "BUY", "timestamp": 1000, "price": 0.99, "size": 10.0}]
    monkeypatch.setattr(pmf, "fetch_market_trades", lambda condition_id: trades)

    r = estimate_vwap_fill("cid", "tok", entry_time_s=1000, entry_price=0.99, desired_shares=100.0)
    assert r is not None
    assert r["filled_shares"] == pytest.approx(10.0)
    assert r["fill_ratio"] == pytest.approx(0.10)
    assert r["vwap"] == pytest.approx(0.99)


def test_estimate_vwap_fill_returns_none_with_no_matching_trades(monkeypatch):
    import polymarket_final_pct as pmf
    monkeypatch.setattr(pmf, "fetch_market_trades", lambda condition_id: [])
    assert estimate_vwap_fill("cid", "tok", entry_time_s=1000, entry_price=0.99, desired_shares=10.0) is None


def test_estimate_vwap_fill_stops_at_a_price_jump_from_new_information(monkeypatch):
    # Regression test for a real bug: an earlier version measured 38% "slippage"
    # on a sports market where the entry print was $0.71 and the next real BUY
    # print, 89 seconds later, was $0.98 -- the game had resolved in the
    # meantime (a goal), not an order consuming liquidity. The guard must stop
    # the walk at that jump rather than average through it.
    import polymarket_final_pct as pmf
    trades = [
        {"asset": "tok", "side": "BUY", "timestamp": 1000, "price": 0.71, "size": 20.0},
        {"asset": "tok", "side": "BUY", "timestamp": 1089, "price": 0.98, "size": 99.0},  # new info regime
        {"asset": "tok", "side": "BUY", "timestamp": 1090, "price": 0.99, "size": 5.0},
    ]
    monkeypatch.setattr(pmf, "fetch_market_trades", lambda condition_id: trades)

    r = estimate_vwap_fill("cid", "tok", entry_time_s=1000, entry_price=0.71, desired_shares=100.0,
                            window_s=300, max_price_deviation=0.03)
    assert r is not None
    assert r["vwap"] == pytest.approx(0.71)  # only the first print is within the deviation guard
    assert r["filled_shares"] == pytest.approx(20.0)
    assert r["fill_ratio"] == pytest.approx(0.20)


def test_stratified_sample_is_subset_stable_as_census_grows():
    old = [_mk_market(i, "2024-02-15T00:00:00Z", "nba game") for i in range(500)] + \
          [_mk_market(i, "2024-05-15T00:00:00Z", "election") for i in range(500, 1000)]
    grown = old + [_mk_market(i, "2026-08-15T00:00:00Z", "bitcoin price") for i in range(1000, 1300)]

    before = set(m["id"] for m in stratified_sample_markets(old, n_target=200))
    after = set(m["id"] for m in stratified_sample_markets(grown, n_target=200))
    old_ids_after = {i for i in after if int(i) < 1000}

    # a group untouched by the new markets shrinks (frac drops as the total
    # census grows) but must never scramble into an unrelated random subset
    assert old_ids_after.issubset(before)
    assert len(old_ids_after) > 0


# ---------------------------------------------------------------------------
# Report generation -- pins write_report's call signature (a mismatch here
# would only otherwise surface at the end of a full, hours-long live run)
# ---------------------------------------------------------------------------

def test_write_report_runs_and_surfaces_other_bucket_coverage(tmp_path):
    cfg = BacktestConfig(signal=SignalConfig(), position_notional=100.0, gas=GasAssumptions(relayer_sponsored=True))
    fill = FillAssumptions(fill_type="maker")
    tdf = pd.DataFrame([
        simulate_trade(_crossing(entry_price=0.99, outcome_index=1), _market(), fill, cfg, cap_shares=None),
    ])
    trades_by_fill = {"maker": tdf}
    days_dist = days_to_resolution_distribution(tdf)
    category_tables = {"maker fills, net of fees": category_breakdown(tdf, "pnl_net")}
    max_days_tables = {"maker fills, net of fees": max_days_to_resolution_variant(tdf, 7.0, "pnl_net")}
    bucket_coverage = report_bucket_coverage([
        {"question": "Will it rain in Boston tomorrow?", "slug": "x", "events": []},  # -> other
    ])

    out_path = tmp_path / "report.md"
    write_report(
        out_path=out_path,
        census_size=1,
        sample_size=1,
        trades_by_fill=trades_by_fill,
        days_dist=days_dist,
        category_tables=category_tables,
        max_days_tables=max_days_tables,
        sensitivity_tables={},
        depth_cap_flags=pd.DataFrame(),
        signal_cfg=SignalConfig(),
        gas_estimate_usd=0.005,
        bucket_coverage=bucket_coverage,
    )

    text = out_path.read_text()
    assert "100.0% other" in text
    assert "other=1" in text
