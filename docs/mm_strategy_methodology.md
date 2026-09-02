# Polymarket Market-Making: Strategy Methodology & Validation

**Status: NOT VALIDATED FOR CAPITAL DEPLOYMENT.** The headline result of the
work documented here is negative: once evaluated under a rigorous,
walk-forward, out-of-sample methodology on an unbiased market population,
this strategy shows no statistically robust edge. Every market-selection
filter that looked attractive in exploratory (in-sample) analysis failed
decisively when re-tested on held-out data. That is the most important
finding in this document, and it is reported as such rather than as a
qualified success — catching this before capital moves is the entire point
of running the validation in the first place.

This document exists to make that conclusion checkable: every number in it
is reproducible from a script in `scripts/` and a result file in
`results/polymarket_final_pct/`, cited by name throughout.

---

## 1. What "market making" means in this codebase

There is no real market-making backtest possible here, and that limitation
shapes everything downstream. Polymarket's `/book` endpoint 404s on resolved
markets — there is no historical order-book depth to determine what price a
resting quote would actually have been filled at, or how often. The only
historical signal available is the public trade-print tape
(`data-api /trades`, via `fetch_market_trades` in `src/polymarket_final_pct.py`).

The model built on top of that tape (`scripts/run_mm_proxy_backtest.py`) is
an explicitly stylized proxy:

- For an assumed half-spread and an assumed "fill share" (the fraction of
  each real trade's size a resting maker quote is assumed to capture), every
  captured unit of every real trade earns exactly the half-spread.
- No inventory-direction term in the base case — this is a deliberate BEST
  CASE (zero adverse selection): perfect, instant, costless flattening of
  every position. Two earlier, more elaborate attempts to add one were
  tried and rejected (see the module docstring) because both let the model
  double-count information that was already priced into the trade tape.
- Adverse selection is instead measured via **markout**: each assumed fill
  is marked against the volume-weighted average price of the trades that
  follow it, not a single next print (which is pure bid-ask bounce, not
  information). Two markout windows are computed side by side:
  - a **trade-count window** (next 20 real prints, whatever real time that
    spans — empirically found to range from a median of ~3.3 hours to a
    mean of ~22 hours across the original population, which is nowhere
    close to a real market maker's reaction speed);
  - a **time window** (15 seconds, matching the same near-immediate-
    execution assumption used elsewhere in this codebase), which is the
    more realistic proxy for "how long does it take to notice adverse
    movement and pull or reprice a quote."
- Constraints: a per-trade notional cap ($25) so one whale print can't
  dominate a market's numbers; a relative-spread cap (half-spread ≤ 30% of
  the distance to 0 or 1) so the flat dollar spread doesn't imply absurd
  percentage spreads near price extremes; and a liquidity/volume-share cap
  (cumulative captured notional per market ≤ 20% of that market's own real
  trade volume) so the model can't implicitly claim to have been the
  dominant liquidity source in a market it never actually quoted in.

Every one of these is an assumption, not a measurement, and is documented as
such in the code. Sensitivity to the two free parameters (half-spread,
fill-share) is checked via a 3×3 grid throughout.

---

## 2. Population construction — the bias that was found and fixed

### 2.1 The original (biased) population

The MM proxy backtest was originally built to reuse the **same market
population and cached trade tapes as the "Final-1%" longshot strategy**
(`trades_maker.csv`, 3,374 markets) — a reasonable-sounding "zero extra
fetches" shortcut. It is also a serious selection-bias problem: every market
in that population was selected because some outcome **crossed the 99%+
threshold** at some point in its life. That is the Final-1% strategy's own
entry signal, not a property a market-making desk would filter on. A fund
quoting spreads across the general market doesn't only quote in markets that
happen to approach a near-certain outcome eventually — most markets never
do.

This was not caught until directly asked whether the MM analysis was
representative. Every conclusion reached before Section 4 below (the pace
segmentation, the Q3 deep dive, the volume and resolution-proximity
follow-ups — see `results/polymarket_final_pct/mm_proxy_results.json` and
`mm_proxy_q3_deep_dive.json`) was validated only on this skewed population.

### 2.2 The fix: an unbiased, stratified sample from the full census

`scripts/build_mm_unbiased_population.py` draws a fresh population
independent of the Final-1% strategy's own selection:

1. Reuses the Final-1% strategy's own **census infrastructure**
   (`fetch_resolved_markets_census`) — a complete, uncurated crawl of every
   resolved Polymarket market since the CLOB-era cutoff (2022-09-01),
   already fully cached on disk from prior work. **844,529** cleanly-resolved
   markets, no probability/extremity filter of any kind.
2. Draws a **stratified random sample** (by resolution-quarter × report
   category, proportional allocation — the same subset-stable sampling
   design as the Final-1% strategy's own `stratified_sample_markets`) of
   **1,517 markets**, seeded independently (seed `20260902` vs. the original
   strategy's `42`) so the two populations share no selection logic.
3. Fetches (and caches) each sampled market's trade tape.

**Memory note, documented because it nearly broke the run:** materializing
the full census as ~800k full Gamma market dicts in one process (as the
existing `fetch_resolved_markets_census()` does) pushed this environment to
9GB+ resident memory and was killed mid-run to avoid an OOM crash on a 15GB
box with no swap. The fix was to stream each cached leaf file one at a time,
reduce every market immediately to the ~6 fields actually needed for
stratification, and discard the rest — peak memory dropped to ~1.4GB.

**What the bias actually looked like, once measured:** the unbiased
sample's category mix is politics=34 (2.2%), sports=657 (43.3%),
crypto_price=289 (19.1%), other=537 (35.4%). Politics — heavily
overrepresented in the original Final-1% population (elections and
appointments frequently do resolve from a near-certain state) — is a small
minority of the true market population. Every "politics survives markout
best" conclusion from the earlier, biased analysis was drawn from a category
that barely exists in Polymarket's actual market mix.

---

## 3. Exploratory analysis (in-sample — see Section 4 for why this is not the final answer)

This section summarizes what was found by directly inspecting the biased
population before the bias was caught. **These findings are reported for
completeness and because building the tooling was not wasted work — the
validation methodology in Section 4 reuses every one of these functions —
but the specific numeric conclusions below did not survive out-of-sample
testing (Section 5) and should not be treated as a trading signal.**

### 3.1 Whole-population result (`mm_proxy_results.json`)

3,374 markets, 3,086 with captured flow at the base config (half-spread
$0.01, fill-share 15%):

- Best case (zero adverse selection): **+$90,498.70**
- 15-second markout: **-$246,871.01**

The gap is the entire finding: a naive spread-capture number looks strongly
profitable and is not, once adverse selection is priced in at a realistic
reaction speed.

### 3.2 Pace segmentation

Markets were bucketed into pace quintiles (Q1=fastest .. Q5=slowest, by
median seconds between real trades — a data-driven, equal-count quantile
split). Selection was by *ranking*, not by assuming "slower is better": Q3
(56–280s between trades) ranked best, **not** Q5 — Q5 "survives" markout in
percentage terms but has almost no real volume to capture. Q1 (<8s) was
worst by a wide margin (-645.5% of its own best-case PnL).

### 3.3 Q3 deep dive (`mm_proxy_q3_deep_dive.json`)

Restricted to the 573 Q3-bucket markets: best case $18,188.71, 15s markout
$8,237.60. Follow-up checks on this subset:

- **Volume segmentation**: the top volume quintile (V5) carried essentially
  all of the bucket's profit ($9,350.71 of $8,237.60 net); V1–V4 roughly
  washed out. But concentration *within* V5 was just as extreme as the whole
  bucket (top 5 of 114 V5 markets = 146% of V5's own total) — a volume floor
  relocated the concentration problem rather than fixing it.
- **Resolution proximity**: trading in the literal final hour before
  resolution barely moved the numbers. But 27% of all Q3 trade volume sat in
  the last 24 hours before resolution, and for the median market, 98.2% of
  its *entire* lifetime volume did — most Q3 markets are effectively
  single-day "flash" markets. Excluding the last 24h entirely dropped the
  population from 573 to 363 markets and *raised* markout PnL to $15,968.19.

Each of these was a legitimate, carefully-built analysis on its own terms.
The problem is procedural, not computational: three sequential cuts (pace,
then volume, then resolution-history) were each selected by looking at
results on the same data being reported. That is exactly the data-snooping
pattern Section 4 exists to catch.

---

## 4. Validation methodology

Two standard quantitative-research safeguards were added, neither of which
existed before this review, and both of which a real fund's risk/validation
function would require before touching capital:

### 4.1 Walk-forward, out-of-sample testing

`scripts/run_mm_walkforward_validation.py`, run on the **unbiased**
population from Section 2:

1. `market_pnl` computed **once** per market at the base config — filter
   combinations are pure in-memory aggregation over precomputed per-market
   stats, not repeated backtesting.
2. Chronological split by `resolution_time`: TRAIN = earlier 70% (932
   markets, 2023-09-10 to 2026-04-17), TEST = later 30% (399 markets,
   2026-04-18 to 2026-09-02, held out entirely from selection).
3. Grid search **on TRAIN ONLY** over candidate (pace range, volume floor,
   minimum pre-resolution trading history) combinations — the same three
   characteristics discovered in Section 3, but now with concrete numeric
   boundaries fit only on data that, in real deployment, would already have
   happened before the test period began. Selection criterion: TRAIN
   **per-market median** markout PnL (median, not total — the total is
   exactly what let a handful of tail markets dominate the Section 3
   conclusions), subject to a minimum surviving-market count.
4. The winning combination applied **unchanged** to TEST.

### 4.2 Sensitivity to the selection procedure itself

A single minimum-market-count threshold is itself a researcher choice that
could bias the grid search toward small, extreme corners of parameter space.
`run_at_min_markets` sweeps this threshold across **[30, 75, 150, 250]** and
reports the out-of-sample result at each — if the conclusion is stable
across this sweep, it isn't an artifact of one arbitrary choice.

### 4.3 Bootstrap confidence intervals and drawdown

For the TEST-period markets that pass the winning filter, `bootstrap_ci`
resamples them with replacement 2,000 times to build an empirical
distribution of total PnL — quantifying how much of a headline number is
real edge versus sampling noise from a small number of markets.
`compute_drawdown` reports the chronological equity curve's peak-to-trough
max drawdown, a basic risk metric no prior version of this analysis
reported.

---

## 5. Results — the out-of-sample answer

### 5.1 Unfiltered baseline (no market selection at all)

On the 345 TEST-period markets with captured flow: best case **+$8,398.91**,
15s markout **-$13,238.51**. The same qualitative pattern as Section 3.1
reproduces on a completely different, unbiased population — best-case
spread capture looks profitable, and adverse selection reverses it. This
part of the finding is robust across both populations tested.

### 5.2 Filtered result, swept across selection strictness

| min_markets | TRAIN-selected filter | TRAIN n / median | TEST n | TEST total markout | TEST median | TEST %positive | P(TEST PnL > 0) |
|---|---|---|---|---|---|---|---|
| 30 | pace<6s, vol≥$22,285, hist≥6h | 53 / $87.32 | 14 | **-$2,858.36** | -$151.19 | 28.6% | **1.1%** |
| 75 | pace<6s, vol≥$4,703, hist≥6h | 81 / $43.68 | 20 | **-$4,177.44** | -$59.61 | 30.0% | **0.2%** |
| 150 | pace<6s, vol≥$908, hist≥0h | 164 / $23.45 | 26 | **-$4,180.20** | -$26.14 | 42.3% | **0.2%** |
| 250 | — | no combination met the threshold | — | — | — | — |

Full detail in `results/polymarket_final_pct/mm_walkforward_validation.json`.

Three things stand out:

1. **The grid search converges on the same regime every time**: markets
   trading faster than 6 seconds apart — the fastest, most active tier.
   This is the *opposite* of the Section 3.2 conclusion (Q3, a *moderate*
   pace, was "best" on the biased in-sample population). The two studies
   disagree with each other, which is itself evidence that neither should
   be trusted without the validation this section provides.
2. **It looks good on TRAIN and fails on TEST, consistently.** TRAIN median
   PnL per market: $87, $44, $23 (all positive, all looking like a real
   edge). TEST result: negative total PnL at all three thresholds, with
   bootstrap-estimated probability of a positive outcome between **0.2% and
   1.1%**. This is not narrowly missing — it is a decisive, repeated
   failure.
3. **The finding is stable across the sensitivity sweep**, which is what
   makes it trustworthy rather than a one-off unlucky split: three
   independently-selected filters (different volume floors, different
   history requirements, all converging on the pace regime) all fail
   out-of-sample by a wide margin.

### 5.3 Headline conclusion

**This stylized MM proxy strategy, evaluated rigorously, shows no
statistically robust edge.** The Q3 pace bucket, the V5 volume quintile, and
the 24-hour flash-market filter — all discovered in Section 3 — were
artifacts of in-sample pattern-matching on a market population that was
never representative to begin with. When the same style of analysis is
repeated correctly (unbiased population, TRAIN-only selection, held-out
TEST, multiple selection-strictness settings, bootstrap confidence
intervals), nothing survives.

---

## 6. Known limitations (beyond what Sections 1–5 already state)

These apply regardless of the validation result and would need to be
addressed before any live deployment, even if a future iteration did find a
validated edge:

- **No real order-book depth, ever.** The entire model is a reconstruction
  from trade prints. It cannot know true resting-quote fill probability,
  queue position, or whether the assumed "fill share" of each print is
  remotely achievable by a real resting order. This is the single largest
  source of uncertainty in every number in this document and cannot be
  fixed with more of this kind of backtesting — it needs live paper-trading
  against the real order book.
- **No capital-scaling / market-impact model.** The 20% volume-share cap
  stops the model from claiming to be the dominant liquidity source in one
  market, but there is no model of what happens to spreads and adverse
  selection as *real* capital is deployed and other participants react.
- **No portfolio-level / correlation risk.** Every result here is a sum of
  independent per-market PnLs. Real concurrent exposure across, say, many
  same-day sports markets or crypto-price markets carries correlation risk
  (a single macro move or a single day's sports outcomes hitting many
  positions at once) that this model does not represent at all.
- **Latency assumptions are still assumptions.** The 15-second markout
  window is borrowed from an unrelated part of this codebase (a
  taker-slippage assumption for a different strategy), not derived from
  anything MM-specific. Section 5's TEST result would need to be re-checked
  under a realistic infrastructure-latency model (order placement +
  confirmation + repricing loop time on Polymarket's actual API) before it
  could be trusted even directionally.
- **Small TEST-period sample.** 399 TEST markets, of which only 14–26
  survive any given filter, is a small base for a bootstrap CI — the
  *direction* of the result (consistently, decisively negative across three
  independent thresholds) is trustworthy; the exact dollar figures carry
  wide uncertainty.
- **Multiple-comparisons risk was mitigated, not eliminated.** The TRAIN
  grid search still tries many candidate combinations (5 pace ranges × 4
  volume floors × 4 history thresholds = 80, per threshold) without a formal
  multiple-testing correction. The walk-forward split is the primary
  defense (a combination that's real should survive contact with unseen
  data regardless of how it was found), and it did not — but a more
  disciplined pre-registered hypothesis (one filter, decided in advance,
  not found via search) would be a stronger test still.
- **Fee and gas modeling is absent from the MM proxy specifically** (the
  Final-1% strategy's own backtest models both; this proxy does not,
  because it never reaches the order-construction stage the fee model
  attaches to). Any live pilot would need real Polymarket maker/taker fee
  and gas costs layered in, which would only make the already-negative
  result worse.

---

## 7. Recommendation

**Do not deploy capital against this strategy as currently specified.** The
out-of-sample evidence is consistent and points the same direction across
every robustness check performed. This is not a "the numbers are marginal,
proceed with caution" result — it is a decisive rejection under the fund's
own validation process.

If this were a live desk deciding what to do next, in priority order:

1. **Get real order-book / quote data**, even a short live-recorded sample,
   to replace the trade-print reconstruction. Everything in this document is
   downstream of not having this, and no amount of further backtesting on
   trade prints alone can fix it.
2. **Run a small, capital-limited live pilot** instead of further backtest
   iteration, specifically to measure real fill rates and real latency —
   the two inputs this model has been guessing at from the start (fill_share
   and the 15s reaction window).
3. **If a future iteration does find a filter that survives walk-forward
   testing**, re-validate it with a pre-registered hypothesis (decide the
   filter *before* looking at held-out data, not via a grid search that then
   gets walk-forward-checked after the fact) and layer in real fees, gas,
   and a market-impact model before sizing any capital against it.
4. **Do not resume trusting the Section 3 findings** (Q3/V5/24h) for
   anything beyond illustrating what in-sample pattern-matching produces —
   they were superseded by Section 5, not refined by it.

---

## 8. Response to external review

An external review of this document raised three "critical errors" and a
list of omissions. Each factual claim was independently re-verified — two
live against Polymarket's own systems and official docs, one against
multiple independent news/social sources — rather than accepted or
dismissed on the reviewer's say-so. Two of the three corrections are real
and material; one is a mischaracterization; both real corrections were then
tested empirically rather than just noted, and neither changes Section 5's
conclusion.

### 8.1 Confirmed correct: the Feb 18, 2026 regime change

Polymarket removed the 500ms taker-order execution delay on 2026-02-18
without prior notice — confirmed via multiple independent, contemporaneous
sources (X/Twitter posts from several accounts, BlockBeats, Odaily, HTX,
WEEX). Before this date, resting maker orders had a brief window to be
cancelled before an adverse fill landed; after, fills are immediate. This
event postdates this analysis's own knowledge cutoff and was correctly
flagged as missing.

**Why it doesn't invalidate Section 5**, tested rather than assumed: the
walk-forward validation's TRAIN period (2023-09 to 2026-04) blends mostly
pre-change data with a post-change tail, while TEST (2026-04 to 2026-09) is
100% post-change — raising a real concern that the TEST failure reflected a
regime mismatch rather than a genuine lack of edge. `scripts/
run_mm_regime_and_rebate_check.py` isolates this by splitting every
market's trade tape on the regime-change timestamp (not resolution date,
since 36 markets straddle it) and computing best-case / 15s-markout
independently for each side:

| | n active markets | Best case | 15s markout | Markout gap |
|---|---|---|---|---|
| Pre-regime-change | 733 | +$34,239.20 | -$76,164.74 | 322.4% |
| Post-regime-change | 634 | +$15,651.85 | -$36,170.33 | 331.1% |

The gap — how much of the naive best-case number adverse selection erases —
is essentially identical in both regimes (322% vs. 331%). The regime change
does not explain Section 5's result; the same fundamental dynamic holds
before and after it.

### 8.2 Confirmed correct (and now modeled): Maker Rebates

Also confirmed directly against Polymarket's own documentation
(`docs.polymarket.com/programs/maker-rebates`, fetched live): makers pay 0%
fees (already correctly modeled in this codebase's Final-1% fee schedule
before this review) and the exchange redistributes a category-specific
share of taker fees back to makers whose resting orders get filled — 20%
for crypto, 15% for sports, 25% for everything else, geopolitics fee-free.
The real payout is a competitive pool split among every maker active in a
market that day (`your_fee_equivalent / everyone's_fee_equivalent × pool`),
which trade-tape data cannot observe — there's no way to know how many
other makers were quoting alongside a hypothetical position.

`market_pnl` in `scripts/run_mm_proxy_backtest.py` now computes an explicit
**upper bound** (`rebate_upper_bound`): what the rebate would be if our
hypothetical maker captured 100% of the pool — i.e. was the sole liquidity
provider, the most generous assumption possible, clearly labeled as a
ceiling and never merged into the existing PnL figures. Measured on the
same unbiased population:

| | Rebate upper bound | vs. the markout gap |
|---|---|---|
| All trades | $8,319.51 | 5.13% |
| Pre-regime | $5,811.33 | 5.26% |
| Post-regime | $2,508.18 | 4.84% |

Even under the most generous possible assumption, the rebate program closes
roughly **5% of the gap** between best-case spread capture and realistic
adverse-selection-adjusted PnL, consistently across both regimes. It is
real, additive income — and nowhere near large enough to change Section 5's
conclusion.

A second program, **Liquidity Rewards** (paid for resting orders near the
midpoint regardless of fill, via a competitive weekly-epoch scoring formula
— also confirmed against Polymarket's own docs), is real but not modeled:
its per-market reward-pool allocation and the competing makers' activity
are not observable from trade-tape data at all, so no defensible number can
be attached to it here. It remains a genuine, flagged upside for a live
pilot to measure directly, not a backtest correction.

### 8.3 Mischaracterized: historical order-book availability

The reviewer's central claim — that Polymarket provides historical L2
order-book data and this analysis's data foundation is therefore invalid —
was re-tested live, not just re-read. `GET /book` on Polymarket's own CLOB
API, called against a market that resolved yesterday (as of this
re-verification) and one from 2023, both return HTTP 404, `"No orderbook
exists for the requested token id"` — the same result this analysis's
original code comments recorded when first confirmed live months earlier.
Polymarket's own systems do not provide this. That specific sentence in the
client report is accurate as written.

What is real: third-party paid services (e.g. a service branded "PMData")
have been archiving L2 snapshots from the public websocket feed — but only
from **2026-02-01 onward** per that service's own documentation. This
analysis's population spans 2023-09 to 2026-09; a service with roughly
seven months of history cannot retroactively provide order-book depth for
the other two-plus years, so it does not "invalidate the entire data
foundation" as claimed. It is, however, a genuinely useful lead for a
narrowly-scoped recent-period study or a live pilot, and is worth
evaluating on its own terms (cost, completeness, reliability) if this
research continues.

The reviewer's specific fee formula (`Fee = C × 0.25 × (p×(1-p))²`) also
does not match Polymarket's own documented formula, re-confirmed live in
the same pass: `fee = C × feeRate × p × (1-p)` (not squared; feeRate is
category-specific, not a flat 0.25) — identical to the formula already
implemented in this codebase's Final-1% fee schedule (`src/
polymarket_final_pct.py`, confirmed live 2026-08-25, independently of this
review).

### 8.4 Not evaluated

The review's remaining suggestions — VPIN/toxicity detection, YES/NO and
cross-platform arbitrage, inventory-aware quoting, and specific open-source
reference implementations — are reasonable directions for a **future**
phase of research, not corrections to the current finding. None were
independently verified in this pass (several of the cited figures, e.g. a
specific aggregate arbitrage-profit total, were not checked against a
primary source and should not be treated as confirmed); they are noted here
as candidate next steps, not as claims this document relies on.

---

## 9. Reproducibility index

| Question | Script | Output |
|---|---|---|
| Base MM proxy model, whole (biased) population | `scripts/run_mm_proxy_backtest.py` | `mm_proxy_results.json` |
| Deep dive on the Q3 pace bucket (biased population) | `scripts/run_mm_proxy_q3_deep_dive.py` | `mm_proxy_q3_deep_dive.json` |
| Unbiased population construction | `scripts/build_mm_unbiased_population.py` | `mm_unbiased_population.csv` |
| Walk-forward validation + sensitivity sweep | `scripts/run_mm_walkforward_validation.py` | `mm_walkforward_validation.json` |
| Regime-change split + Maker Rebate upper bound | `scripts/run_mm_regime_and_rebate_check.py` | `mm_regime_and_rebate_check.json` |
| Unit tests for all pure functions above | `tests/test_mm_proxy_backtest.py`, `tests/test_mm_proxy_q3_deep_dive.py`, `tests/test_mm_walkforward_validation.py`, `tests/test_mm_regime_and_rebate_check.py` | `pytest tests/` (163 passing at time of writing) |

All scripts reuse already-cached data wherever possible (trade tapes,
census leaf files) and are safe to re-run.
