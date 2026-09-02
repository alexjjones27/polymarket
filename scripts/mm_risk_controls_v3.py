"""Follow-up to market_pnl_advanced (run_mm_proxy_backtest.py): tests whether
the specific gaps flagged in that work -- VPIN buckets too small (mean VPIN
0.88, should center near 0.5), a linear/binary inventory skew, a flat
per-trade notional cap regardless of volatility, and no reaction to a fill
that immediately goes wrong -- can be narrowed further. Every mechanism here
is causal (a fill's terms depend only on trades strictly before it, verified
in tests the same way compute_vpin_series/market_pnl_advanced already are)
and switches off to reproduce market_pnl_advanced's own numbers, never
silently replacing them -- same discipline as every prior increment in this
file's lineage.

Four independent additions, each individually toggleable:

1. DYNAMIC VPIN BUCKETS (compute_vpin_series_dynamic): bucket notional is
   recalibrated per market (and over time within a market) from a rolling
   median trade size, instead of one flat $500 for every market regardless
   of its own typical trade size -- the direct fix for the 0.88 mean-VPIN
   calibration problem flagged in Section 9.

2. SIGMOID INVENTORY SKEW (sigmoid_skew_headroom): replaces the linear
   headroom ramp with a logistic curve centered on the inventory limit, so
   small positions are barely derated and derating accelerates near the
   limit, plus an exponential half-life mean-reversion of the tracked
   inventory itself between fills (a proxy for real desks doing other
   flattening work over time, not just waiting for an opposite fill).

3. VOLATILITY-SCALED POSITION CAP (volatility_scaled_notional_cap): the
   flat $25 per-trade cap shrinks as recent realized price volatility rises.

4. ORDER-IMBALANCE TOXICITY (compute_order_imbalance_series) as a simpler,
   single-parameter alternative to VPIN, plus a POST-FILL COOLDOWN
   (cooldown-related helpers) that retroactively checks whether a captured
   fill went on to look toxic and, if so, temporarily widens spread /
   shrinks size with an exponentially decaying "heat" state.
"""
import math
import statistics

# ---------------------------------------------------------------------------
# 1. Dynamic (volume-synchronized) VPIN bucket sizing
# ---------------------------------------------------------------------------
# The fixed VPIN_BUCKET_NOTIONAL=$500 in run_mm_proxy_backtest.py produced a
# mean VPIN of 0.88 across the unbiased population -- a measure that should
# center near 0.5 for typical flow was instead pinned near its 1.0 ceiling,
# meaning most buckets held only one or two trades and looked one-sided from
# small-sample noise, not genuine informed trading (flagged, not fixed, in
# Section 9). The fix: size each bucket from that MARKET's own recent trade
# sizes, not one number assumed to fit every market.

VPIN_ROLLING_WINDOW_TRADES = 50  # how many past trades the rolling median is estimated over
VPIN_MIN_OBSERVATIONS = 5        # fewer than this and the rolling median is considered unreliable
MIN_DYNAMIC_BUCKET_NOTIONAL = 20.0  # floor so a burst of tiny trades can't collapse the bucket to ~$0


def rolling_median_trade_notional(sorted_trades: list[dict], i: int,
                                   window: int = VPIN_ROLLING_WINDOW_TRADES,
                                   min_observations: int = VPIN_MIN_OBSERVATIONS):
    """Median notional (price*size) of the up-to-`window` trades STRICTLY
    BEFORE index i -- causal, usable as a per-trade, adaptive "what does a
    typical trade in this market look like right now" estimate. None if
    fewer than `min_observations` prior trades exist (too early in the tape
    to trust the estimate)."""
    lo = max(0, i - window)
    obs = [t["price"] * t["size"] for t in sorted_trades[lo:i]]
    if len(obs) < min_observations:
        return None
    return statistics.median(obs)


def compute_vpin_series_dynamic(sorted_trades: list[dict],
                                 bucket_trade_target: float = 10.0,
                                 rolling_window: int = VPIN_ROLLING_WINDOW_TRADES,
                                 min_bucket_notional: float = MIN_DYNAMIC_BUCKET_NOTIONAL,
                                 fallback_bucket_notional: float = 500.0,
                                 window_buckets: int = 20) -> list:
    """Same causal construction as compute_vpin_series (run_mm_proxy_backtest.py)
    -- vpin_series[i] uses only buckets fully formed from trades strictly
    before i -- but the notional threshold that completes a bucket is
    recomputed at every step as `rolling_median_trade_notional(...) *
    bucket_trade_target`, floored at `min_bucket_notional`: "a bucket should
    hold about `bucket_trade_target` typically-sized recent trades," instead
    of one flat dollar amount assumed to fit every market's trade-size
    distribution. Before enough trades exist to estimate a rolling median
    (see VPIN_MIN_OBSERVATIONS), falls back to `fallback_bucket_notional`
    (the original fixed default) rather than leaving the market's opening
    stretch with no VPIN estimate at all."""
    n = len(sorted_trades)
    vpin_series = [None] * n
    completed_bucket_imbalances = []
    bucket_buy = 0.0
    bucket_sell = 0.0
    bucket_notional_accum = 0.0
    for i, t in enumerate(sorted_trades):
        if completed_bucket_imbalances:
            recent = completed_bucket_imbalances[-window_buckets:]
            vpin_series[i] = sum(recent) / len(recent)

        notional = t["price"] * t["size"]
        if t["side"] == "BUY":
            bucket_buy += notional
        else:
            bucket_sell += notional
        bucket_notional_accum += notional

        median_trade = rolling_median_trade_notional(sorted_trades, i, rolling_window)
        threshold = (max(min_bucket_notional, median_trade * bucket_trade_target)
                     if median_trade is not None else fallback_bucket_notional)
        if bucket_notional_accum >= threshold:
            imbalance = abs(bucket_buy - bucket_sell) / bucket_notional_accum
            completed_bucket_imbalances.append(imbalance)
            bucket_buy = bucket_sell = bucket_notional_accum = 0.0
    return vpin_series


# ---------------------------------------------------------------------------
# 2. Sigmoid inventory skew with a target and a half-life
# ---------------------------------------------------------------------------
# market_pnl_advanced's headroom = max(0, 1 - |inventory|/limit) is LINEAR:
# a position at 10% of the limit is derated exactly as proportionally as one
# at 90%. Real risk-limit behavior is usually closer to "barely notice a
# small position, clamp down hard as the limit approaches" -- a sigmoid
# captures that shape and gives a tunable steepness (skew_strength) instead
# of one fixed slope.

DEFAULT_SKEW_STRENGTH = 6.0  # logistic steepness; higher = sharper transition near the limit
DEFAULT_INVENTORY_HALF_LIFE_SECONDS = None  # None = no time decay (reduces to market_pnl_advanced's static inventory)


def sigmoid_skew_headroom(displacement_notional: float, limit_notional: float,
                           skew_strength: float = DEFAULT_SKEW_STRENGTH) -> float:
    """Fraction (0..1) of fill_share retained on the side moving further from
    target, as a function of |displacement| relative to `limit_notional`.
    u = |displacement|/limit: headroom(0) ~ 1 (barely derated when flat),
    headroom(1) = 0.5 (half-derated exactly at the nominal limit),
    headroom(u->inf) -> 0. `skew_strength` controls how sharply the
    transition happens around u=1 -- low values approach the old linear
    behavior's gentleness, high values approach a hard cutoff right at the
    limit. Returns 1.0 (no derating) if limit_notional <= 0 is not a valid
    limit to reason about (treated as "no limit configured" -- callers pass
    float('inf') for genuinely no limit, which also yields u=0 and headroom
    close to 1 with no special-casing needed)."""
    if limit_notional <= 0 or math.isinf(limit_notional):
        return 1.0
    u = abs(displacement_notional) / limit_notional
    x = skew_strength * (u - 1.0)
    # A market that runs inventory far past the nominal limit (the gentle
    # near-zero slope is the whole point of the sigmoid, but it means a
    # position CAN overshoot before derating bites hard) can push x well
    # past exp()'s ~709 overflow point. Clamp rather than let it raise --
    # both tails are already exactly 0.0/1.0 to double precision by then.
    if x > 700:
        return 0.0
    if x < -700:
        return 1.0
    return 1.0 / (1.0 + math.exp(x))


def decay_inventory_toward_target(inventory_shares: float, dt_seconds: float,
                                   half_life_seconds, target_shares: float = 0.0) -> float:
    """Exponentially decays a running inventory estimate toward `target_shares`
    over `dt_seconds` of elapsed real time, at the given half-life -- models a
    desk doing other flattening work between prints (hedging, working an
    aggressive order, manual intervention) rather than the position only ever
    changing on the next opposite-side fill. half_life_seconds of None or a
    non-positive value disables decay entirely (returns inventory_shares
    unchanged), so this reduces to market_pnl_advanced's static-inventory
    behavior when not configured."""
    if half_life_seconds is None or half_life_seconds <= 0 or dt_seconds <= 0:
        return inventory_shares
    decay = 0.5 ** (dt_seconds / half_life_seconds)
    return target_shares + (inventory_shares - target_shares) * decay


# ---------------------------------------------------------------------------
# 3. Volatility-scaled position cap
# ---------------------------------------------------------------------------
# MAX_NOTIONAL_PER_TRADE ($25) is flat regardless of how much the market has
# recently been moving. Scaling it down when realized volatility is elevated
# is standard risk practice: a resting quote is more exposed to being run
# over by a fast-moving, high-volatility print than a slow, stable one.

VOL_WINDOW_TRADES = 50           # realized vol measured over the last N trades, same order as VPIN_ROLLING_WINDOW_TRADES
VOL_MIN_OBSERVATIONS = 5
VOL_REFERENCE = 0.02             # a "typical" trade-to-trade price stdev for this dataset; cap is unscaled (1x) at this level
VOL_CAP_SENSITIVITY = 1.0        # how strongly the cap shrinks per multiple of VOL_REFERENCE
MIN_NOTIONAL_CAP_FRACTION = 0.2  # cap can never shrink below this fraction of the base cap


def realized_price_vol(sorted_trades: list[dict], i: int, window: int = VOL_WINDOW_TRADES,
                        min_observations: int = VOL_MIN_OBSERVATIONS):
    """Sample standard deviation of trade PRICES (not returns -- prices here
    are already bounded probabilities in [0,1], so a raw price stdev is a
    directly interpretable, unit-consistent volatility proxy) over the last
    `window` trades strictly before index i. None if fewer than
    `min_observations` prior trades exist."""
    lo = max(0, i - window)
    prices = [t["price"] for t in sorted_trades[lo:i]]
    n = len(prices)
    if n < min_observations:
        return None
    # Plain-float sample stdev, not statistics.stdev: the stdlib version
    # routes through exact Fraction arithmetic for precision, which is
    # 10-20x slower and unnecessary at this dataset's float precision --
    # measured to turn a ~2.5s full-population pass into ~48s when this
    # runs inside market_pnl_v3's per-trade hot loop.
    mean = sum(prices) / n
    variance = sum((p - mean) ** 2 for p in prices) / (n - 1)
    return variance ** 0.5


def volatility_scaled_notional_cap(base_cap: float, vol,
                                    vol_reference: float = VOL_REFERENCE,
                                    sensitivity: float = VOL_CAP_SENSITIVITY,
                                    min_cap_fraction: float = MIN_NOTIONAL_CAP_FRACTION) -> float:
    """Shrinks `base_cap` as `vol` (realized_price_vol's output) rises above
    `vol_reference`: multiplier = 1 / (1 + sensitivity * max(0, vol/vol_reference - 1)),
    floored at `min_cap_fraction` of base_cap so the cap never collapses to
    zero. vol=None (not enough history yet) or vol_reference<=0 returns
    base_cap unscaled."""
    if vol is None or vol_reference <= 0:
        return base_cap
    excess = max(0.0, vol / vol_reference - 1.0)
    multiplier = 1.0 / (1.0 + sensitivity * excess)
    return base_cap * max(min_cap_fraction, multiplier)


# ---------------------------------------------------------------------------
# 4. Order imbalance (simpler VPIN alternative) + post-fill cooldown
# ---------------------------------------------------------------------------

ORDER_IMBALANCE_WINDOW_TRADES = 20


def compute_order_imbalance_series(sorted_trades: list[dict],
                                    window_trades: int = ORDER_IMBALANCE_WINDOW_TRADES) -> list:
    """Causal, single-parameter alternative to VPIN: signed net volume (buy
    notional minus sell notional) over the last `window_trades` trades
    STRICTLY BEFORE index i, normalized by their total notional -- in
    [-1, 1]. Unlike VPIN this needs no volume-clock/bucket construction at
    all, just a trade-count lookback, at the cost of being noisier for
    lopsided trade-SIZE (rather than trade-COUNT) flow. None where fewer
    than 1 prior trade with nonzero notional exists."""
    n = len(sorted_trades)
    series = [None] * n
    for i in range(n):
        lo = max(0, i - window_trades)
        window = sorted_trades[lo:i]
        if not window:
            continue
        buy = sum(t["price"] * t["size"] for t in window if t["side"] == "BUY")
        sell = sum(t["price"] * t["size"] for t in window if t["side"] == "SELL")
        total = buy + sell
        if total <= 0:
            continue
        series[i] = (buy - sell) / total
    return series


# A fill is judged "toxic" (retroactively, once enough real time has passed
# to have a VWAP to judge it against) if price drifted against it by more
# than TOXIC_ADVERSE_SPREAD_MULTIPLE times the half-spread that fill earned.
# Once triggered, cooldown "heat" starts at 1.0 and decays with the given
# half-life; while heat > 0 it multiplicatively widens the effective spread
# and derates fill_share on EVERY side (not just the side that caused it --
# a toxic print is evidence the whole market just got more dangerous to
# quote in, not just one side of it).
TOXIC_EVAL_DELAY_SECONDS = 15.0       # matches MARKOUT_WINDOW_SECONDS's own reaction-latency assumption
TOXIC_ADVERSE_SPREAD_MULTIPLE = 2.0
COOLDOWN_HALF_LIFE_SECONDS = 30.0
COOLDOWN_MAX_SPREAD_BOOST = 1.0       # heat=1.0 -> spread multiplier up to 2x
COOLDOWN_MAX_SIZE_CUT = 0.5           # heat=1.0 -> fill_share cut by up to 50%


def cooldown_heat(elapsed_seconds, half_life_seconds: float = COOLDOWN_HALF_LIFE_SECONDS) -> float:
    """1.0 right at a toxic-fill trigger, decaying by half every
    `half_life_seconds` of elapsed real time since. 0.0 if no trigger has
    happened yet (elapsed_seconds is None)."""
    if elapsed_seconds is None or elapsed_seconds < 0:
        return 0.0
    return 0.5 ** (elapsed_seconds / half_life_seconds)


TOXIC_EVAL_MAX_SCAN = 500  # safety cap on the backward VWAP scan, same idea as base.MAX_TIME_WINDOW_SCAN


def _vwap_since(sorted_trades: list[dict], fill_idx: int, now_idx: int, max_scan: int = TOXIC_EVAL_MAX_SCAN):
    """VWAP of trades strictly between fill_idx and now_idx (both indices
    into sorted_trades), scanning at most `max_scan` trades back from
    now_idx as a safety valve. Returns None if there are no trades in that
    span."""
    lo = max(fill_idx + 1, now_idx - max_scan)
    window = sorted_trades[lo:now_idx]
    if not window:
        return None
    w_shares = sum(w["size"] for w in window)
    if w_shares <= 0:
        return None
    return sum(w["size"] * w["price"] for w in window) / w_shares


# ---------------------------------------------------------------------------
# The composed model
# ---------------------------------------------------------------------------

def market_pnl_v3(sorted_trades, total_market_volume, half_spread, fill_share,
                   markout_window_seconds: float = 15.0,
                   base_notional_cap: float = 25.0,
                   max_relative_spread: float = 0.3,
                   max_market_volume_share: float = 0.20,
                   # 1. toxicity signal (spread widening)
                   toxicity_mode: str = "vpin_fixed",  # "vpin_fixed" | "vpin_dynamic" | "order_imbalance" | "none"
                   vpin_bucket_notional: float = 500.0,
                   vpin_bucket_trade_target: float = 10.0,
                   vpin_window_buckets: int = 20,
                   order_imbalance_window_trades: int = ORDER_IMBALANCE_WINDOW_TRADES,
                   spread_multiplier_max: float = 2.0,
                   # 2. inventory skew
                   skew_mode: str = "linear",  # "linear" (market_pnl_advanced's own formula) | "sigmoid"
                   inventory_limit_notional: float = 100.0,
                   target_inventory_notional: float = 0.0,
                   skew_strength: float = DEFAULT_SKEW_STRENGTH,
                   inventory_half_life_seconds=None,
                   # 3. volatility-scaled position cap
                   use_volatility_cap: bool = False,
                   vol_window_trades: int = VOL_WINDOW_TRADES,
                   vol_reference: float = VOL_REFERENCE,
                   vol_cap_sensitivity: float = VOL_CAP_SENSITIVITY,
                   # 4. post-fill cooldown
                   enable_cooldown: bool = False,
                   toxic_eval_delay_seconds: float = TOXIC_EVAL_DELAY_SECONDS,
                   toxic_adverse_spread_multiple: float = TOXIC_ADVERSE_SPREAD_MULTIPLE,
                   cooldown_half_life_seconds: float = COOLDOWN_HALF_LIFE_SECONDS,
                   cooldown_max_spread_boost: float = COOLDOWN_MAX_SPREAD_BOOST,
                   cooldown_max_size_cut: float = COOLDOWN_MAX_SIZE_CUT):
    """market_pnl_advanced's spread-capture/time-markout core, generalized
    with four independently-toggleable risk controls (see module docstring).
    Every toggle has a default that reproduces market_pnl_advanced's own
    behavior: toxicity_mode="vpin_fixed" with the same bucket size, skew_mode
    "linear" with the same limit, use_volatility_cap=False, enable_cooldown=
    False -- verified directly in tests, not just asserted here. Returns the
    same core fields as market_pnl_advanced (pnl_best_case,
    pnl_with_markout_time, n_captured, captured_notional,
    volume_share_captured, avg_vpin, max_abs_inventory_notional,
    n_inventory_capped) plus n_cooldown_triggers."""
    if toxicity_mode == "vpin_fixed":
        toxicity_series = compute_vpin_series_fixed_adapter(sorted_trades, vpin_bucket_notional, vpin_window_buckets)
    elif toxicity_mode == "vpin_dynamic":
        toxicity_series = compute_vpin_series_dynamic(
            sorted_trades, bucket_trade_target=vpin_bucket_trade_target,
            fallback_bucket_notional=vpin_bucket_notional, window_buckets=vpin_window_buckets)
    elif toxicity_mode == "order_imbalance":
        raw = compute_order_imbalance_series(sorted_trades, order_imbalance_window_trades)
        toxicity_series = [abs(x) if x is not None else None for x in raw]
    elif toxicity_mode == "none":
        toxicity_series = [None] * len(sorted_trades)
    else:
        raise ValueError(f"unknown toxicity_mode: {toxicity_mode!r}")

    pnl_best_case = 0.0
    pnl_with_markout_time = 0.0
    n_captured = 0
    captured_notional = 0.0
    inventory_shares = 0.0
    max_abs_inventory_notional = 0.0
    n_inventory_capped = 0
    n_cooldown_triggers = 0
    toxicity_samples = []
    volume_cap = max_market_volume_share * total_market_volume
    last_ts = None
    cooldown_trigger_ts = None
    pending_fills = []  # list of (fill_idx, fill_ts, fill_price, fill_side, fill_shares, eff_half_spread)

    for i, t in enumerate(sorted_trades):
        price, size, side, ts = t["price"], t["size"], t["side"], t["timestamp"]
        base_eff_half_spread = min(half_spread, max_relative_spread * price, max_relative_spread * (1 - price))
        if base_eff_half_spread <= 0:
            last_ts = ts
            continue

        # --- retroactively evaluate any pending fills old enough to judge ---
        # (skipped entirely when cooldown is off -- both for performance and
        # so a disabled control can never leave a trace in the output, same
        # discipline as every other toggle here)
        if enable_cooldown:
            still_pending = []
            for (f_idx, f_ts, f_price, f_side, f_shares, f_half_spread) in pending_fills:
                if ts - f_ts < toxic_eval_delay_seconds:
                    still_pending.append((f_idx, f_ts, f_price, f_side, f_shares, f_half_spread))
                    continue
                vwap = _vwap_since(sorted_trades, f_idx, i)
                if vwap is not None:
                    adverse = (f_price - vwap) if f_side == "SELL" else (vwap - f_price)
                    if adverse > toxic_adverse_spread_multiple * f_half_spread:
                        cooldown_trigger_ts = ts
                        n_cooldown_triggers += 1
            pending_fills = still_pending

        # --- toxicity-driven spread widening ---
        toxicity = toxicity_series[i]
        spread_multiplier = 1.0 + toxicity * (spread_multiplier_max - 1.0) if toxicity is not None else 1.0
        if toxicity is not None:
            toxicity_samples.append(toxicity)

        # --- cooldown-driven extra widening / size cut ---
        cooldown_mult = 1.0
        cooldown_size_mult = 1.0
        if enable_cooldown and cooldown_trigger_ts is not None:
            heat = cooldown_heat(ts - cooldown_trigger_ts, cooldown_half_life_seconds)
            cooldown_mult = 1.0 + cooldown_max_spread_boost * heat
            cooldown_size_mult = max(0.0, 1.0 - cooldown_max_size_cut * heat)

        eff_half_spread = min(base_eff_half_spread * spread_multiplier * cooldown_mult,
                               max_relative_spread * price, max_relative_spread * (1 - price))

        # --- inventory decay + skew ---
        if inventory_half_life_seconds is not None and last_ts is not None:
            inventory_shares = decay_inventory_toward_target(
                inventory_shares, ts - last_ts, inventory_half_life_seconds,
                target_shares=target_inventory_notional / price if price else 0.0)
        last_ts = ts

        inventory_notional = inventory_shares * price
        displacement = inventory_notional - target_inventory_notional
        moving_away_from_flat = (side == "SELL" and displacement >= 0) or (side == "BUY" and displacement <= 0)
        skewed_fill_share = fill_share * cooldown_size_mult
        if moving_away_from_flat and inventory_limit_notional > 0:
            if skew_mode == "linear":
                headroom = max(0.0, 1.0 - abs(displacement) / inventory_limit_notional)
            elif skew_mode == "sigmoid":
                headroom = sigmoid_skew_headroom(displacement, inventory_limit_notional, skew_strength)
            else:
                raise ValueError(f"unknown skew_mode: {skew_mode!r}")
            skewed_fill_share *= headroom
            if headroom < 1.0:
                n_inventory_capped += 1

        notional_cap = base_notional_cap
        if use_volatility_cap:
            vol = realized_price_vol(sorted_trades, i, vol_window_trades)
            notional_cap = volatility_scaled_notional_cap(base_notional_cap, vol, vol_reference, vol_cap_sensitivity)

        shares = min(size * skewed_fill_share, notional_cap / price)
        if shares <= 0:
            continue
        notional = shares * price
        if captured_notional + notional > volume_cap:
            remaining = volume_cap - captured_notional
            if remaining <= 1e-9:
                break
            shares = remaining / price
            notional = remaining

        pnl_best_case += shares * eff_half_spread

        def _adverse(markout_price):
            return (price - markout_price) * shares if side == "SELL" else (markout_price - price) * shares

        time_markout_price, _ = _time_window_vwap_local(sorted_trades, i, markout_window_seconds)
        if time_markout_price is not None:
            pnl_with_markout_time += shares * eff_half_spread - _adverse(time_markout_price)
        else:
            pnl_with_markout_time += shares * eff_half_spread

        if enable_cooldown:
            pending_fills.append((i, ts, price, side, shares, eff_half_spread))
        inventory_shares += shares if side == "SELL" else -shares
        max_abs_inventory_notional = max(max_abs_inventory_notional, abs(inventory_shares * price))

        n_captured += 1
        captured_notional += notional
        if captured_notional >= volume_cap:
            break

    return {
        "pnl_best_case": pnl_best_case,
        "pnl_with_markout_time": pnl_with_markout_time,
        "n_captured": n_captured,
        "captured_notional": captured_notional,
        "volume_share_captured": captured_notional / total_market_volume if total_market_volume else None,
        "avg_vpin": sum(toxicity_samples) / len(toxicity_samples) if toxicity_samples else None,
        "max_abs_inventory_notional": max_abs_inventory_notional,
        "n_inventory_capped": n_inventory_capped,
        "n_cooldown_triggers": n_cooldown_triggers,
    }


def compute_vpin_series_fixed_adapter(sorted_trades, bucket_notional, window_buckets):
    """Thin wrapper around run_mm_proxy_backtest.compute_vpin_series so this
    module doesn't need a hard import-time dependency on that script (avoids
    a circular import: run_mm_proxy_v3.py imports both). Imported lazily."""
    import run_mm_proxy_backtest as base
    return base.compute_vpin_series(sorted_trades, bucket_notional, window_buckets)


def _time_window_vwap_local(sorted_trades, i, seconds, max_scan=500):
    """Local copy of run_mm_proxy_backtest._time_window_vwap (kept private
    there) -- VWAP of trades within `seconds` after sorted_trades[i]."""
    t0 = sorted_trades[i]["timestamp"]
    w_notional = 0.0
    w_shares = 0.0
    n = 0
    j = i + 1
    limit = min(len(sorted_trades), i + 1 + max_scan)
    while j < limit and sorted_trades[j]["timestamp"] - t0 <= seconds:
        w = sorted_trades[j]
        w_notional += w["size"] * w["price"]
        w_shares += w["size"]
        n += 1
        j += 1
    if w_shares <= 0:
        return None, 0
    return w_notional / w_shares, n
