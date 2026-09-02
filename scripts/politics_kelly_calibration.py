"""Pure functions for the politics price-calibrated Kelly backtest
(scripts/run_politics_kelly_backtest.py). No network calls.

The strategy's edge does NOT come from trusting a market's own price as the
true probability -- betting p_estimate == market_price makes the Kelly
formula return exactly zero every time (b*p - q = 0 identically when
p == price, before any fee is even applied). The edge instead comes from
checking, walk-forward and empirically, whether markets whose YES price
sits in a given range actually resolve YES more or less often than that
price range implies -- the same idea as a sports-book calibration study,
applied to Polymarket politics markets across the whole price spectrum
instead of the Final-1% strategy's own near-100% tail.
"""

BUCKET_WIDTH = 0.10  # 10 buckets spanning [0, 1)


def price_bucket(yes_price: float, bucket_width: float = BUCKET_WIDTH) -> int:
    """Which bucket index (0-based) a YES price falls into. Bucket k spans
    [k*width, (k+1)*width); price 1.0 is clamped into the top bucket rather
    than overflowing into a nonexistent extra one."""
    if not (0.0 <= yes_price <= 1.0):
        raise ValueError(f"yes_price must be in [0, 1], got {yes_price}")
    n_buckets = round(1.0 / bucket_width)
    idx = int(yes_price / bucket_width)
    return min(idx, n_buckets - 1)


def bucket_midpoint(bucket_idx: int, bucket_width: float = BUCKET_WIDTH) -> float:
    return (bucket_idx + 0.5) * bucket_width


def bucket_range(bucket_idx: int, bucket_width: float = BUCKET_WIDTH) -> tuple[float, float]:
    return (bucket_idx * bucket_width, (bucket_idx + 1) * bucket_width)


def midpoint_centered_prior(bucket_idx: int, prior_strength: float, bucket_width: float = BUCKET_WIDTH) -> tuple[float, float]:
    """Beta(prior_a, prior_b) prior for a price bucket, centered on the
    bucket's OWN midpoint rather than one fixed prior for every bucket --
    "assume the market is calibrated until this bucket's own resolved
    history says otherwise," with `prior_strength` pseudo-observations of
    weight (higher = slower to react to real data, lower = noisier early
    estimates). This is the key generalization from the Final-1% strategy's
    single near-zero-flip-rate prior (appropriate only for its own
    near-100% tail) to a prior that makes sense across the full price range."""
    mid = bucket_midpoint(bucket_idx, bucket_width)
    return (mid * prior_strength, (1.0 - mid) * prior_strength)


def calibrated_p_yes(prior_a: float, prior_b: float, observed_wins: int, observed_n: int) -> float:
    """Posterior mean of a Beta(prior_a + wins, prior_b + losses) --
    the walk-forward empirical P(YES) estimate for a bucket, given
    everything resolved in it so far."""
    return (prior_a + observed_wins) / (prior_a + prior_b + observed_n)


def kelly_fraction(p: float, price: float, fee_frac: float) -> float:
    """Full-Kelly fraction of bankroll to stake on YES at `price`, believing
    the true probability is `p`, paying `fee_frac` of notional in fees.
    Same formula as run_kelly_backtest.py's run_sim: b = net odds per dollar
    staked, L = total capital at risk per dollar staked (>1 because a losing
    stake's fee still has to come from somewhere -- modeled the same way as
    the existing Kelly backtest for consistency). Returns 0 (not negative)
    when there's no edge -- a negative Kelly fraction means "bet the other
    side," which this strategy doesn't do (see module docstring: it only
    ever evaluates buying YES, since the YES-priced calibration curve
    already encodes the NO side symmetrically)."""
    if price <= 0 or price >= 1:
        return 0.0
    b = (1.0 - price) / price - fee_frac
    if b <= 0:
        return 0.0
    q = 1.0 - p
    L = 1.0 + fee_frac
    f = (p * b - q * L) / (b * L)
    return max(0.0, f)
