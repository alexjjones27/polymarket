# Polymarket Market-Making Strategy — Research Report

**Prepared for client review**
**Subject: mechanics of the model, research process, and current findings**

---

## Summary

This report explains a market-making strategy researched for Polymarket
(the prediction-market exchange): what the strategy is trying to do, exactly
how the model that estimates its profitability works, and what our
validation process found.

**Current status: not recommended for capital deployment.** The strategy
looked profitable under an initial, simpler analysis. When we subjected it
to a more rigorous test — one designed specifically to catch results that
look good only because of how the data happened to be chosen or examined —
it failed. We are reporting that outcome directly, because the value we are
providing here is as much the discipline that caught the problem as the
research itself. A strategy that survives real out-of-sample testing is
worth funding; a strategy that only looks good until you test it properly
is worth knowing about before capital moves, not after.

The rest of this report explains the mechanics in full, then walks through
what we found and why we don't recommend proceeding on the current basis.

---

## 1. The strategy, in plain terms

**Market making** means continuously offering to both buy and sell an asset
at slightly different prices — a "bid" (what you'll pay) a little below the
market, and an "ask" (what you'll sell for) a little above it. Every time
someone trades against one of your standing orders, you earn the difference
between your quote and the "true" price — the **spread**. You do this over
and over, across many trades, and the accumulated spread is the profit.

The risk is **adverse selection**: sometimes a trade happens against you
specifically *because* the trader knows something you don't yet — the price
is about to move against the position you just took on. If that happens
often enough, or badly enough, the losses from being run over by informed
trading exceed the spread you collected. Real market-making profitability
is the net of those two effects, not just the spread income in isolation.

Polymarket is a prediction-market exchange: every market resolves to $1.00
(if the outcome happened) or $0.00 (if it didn't). A market maker there is
quoting a spread on the *probability* of an event, and gets paid to provide
liquidity to traders who want to bet on it. The question this research
project set out to answer: **can this be done profitably on Polymarket, and
if so, in which markets?**

---

## 2. Why this can't be tested the "normal" way

The standard way to backtest a market-making strategy is to replay a
historical, tick-by-tick record of the order book — every price level, at
every moment, showing exactly what a resting order would have been filled
against and when. **Polymarket does not make this available for markets
that have already resolved.** Once a market closes, its order-book history
is gone; only the record of completed trades (the "trade tape" — every
executed trade, its price, size, and timestamp) remains accessible.

This is a real limitation, and it means every number in this report is an
**estimate built from a proxy model**, not a measurement from a true
order-book replay. We were explicit about this limitation from the start
and built the model to make its own assumptions visible rather than
papering over them — every assumption below is named, and every result is
reported alongside the assumption it depends on.

---

## 3. How the model works, step by step

### 3.1 The core idea

For every real trade that happened on a market, the model asks a
hypothetical question: *"If we had been resting a limit order priced a
little better than this trade's price, would some of this trade have
filled against us?"*

Two numbers govern the answer:

- **Half-spread** — how far inside the trade price our hypothetical order
  sits. A tighter (smaller) half-spread fills more often but earns less per
  fill; a wider one earns more per fill but is less likely to have been
  competitive.
- **Fill share** — what fraction of that trade's size we assume would have
  gone to our order rather than someone else's. Real limit order books have
  many participants queued at similar prices; no single resting order
  captures 100% of the flow at its price level.

Both are treated as **assumptions to be tested across a range**, not fixed
inputs. We ran the model across a 3×3 grid of half-spread and fill-share
values (half-spreads of 0.5¢, 1¢, and 2¢; fill shares of 5%, 15%, and 30%)
to see how sensitive the results are to these assumptions rather than
reporting one cherry-picked combination.

### 3.2 Best-case P&L (the starting point, not the answer)

If we assume every hypothetical fill is captured and then instantly,
perfectly, costlessly closed out at no further risk, every captured unit
earns exactly the half-spread — full stop. This is the **best case**: it
is a deliberate ceiling, not a realistic estimate, because it contains **no
adverse-selection cost at all**. We report it because it is the natural
first thing anyone asks ("what's the theoretical upside"), but it is not
the number a real trading decision should be based on.

Two earlier, more sophisticated attempts to add a realistic cost term to
this model were tried and discarded, and it's worth explaining why, because
both looked plausible before the numbers exposed the problem:

1. **Marking the position to the market's final outcome (win/loss).** This
   let a handful of long-shot markets that decayed to zero dominate the
   entire result with directional betting profit that has nothing to do
   with market-making — one single market accounted for nearly a third of
   the total in an early version of this test.
2. **Marking the position to the very next trade print.** This inflated
   results, because consecutive real trade prints naturally alternate
   between hitting the bid and the ask even when the underlying probability
   hasn't moved at all (ordinary "bid-ask bounce"). Using the very next
   print as a mark mistook this bounce for real information.

### 3.3 Markout: the realistic adverse-selection estimate

The technique we settled on is standard in real trading-performance
analysis: **markout**. Instead of marking a hypothetical fill against the
single next trade (noisy) or the final outcome (long-run directional bet,
not market-making), we mark it against the **volume-weighted average price
of the trades that follow it over a defined window**. Averaging over a
window of many subsequent prints cancels out the random bid/ask bounce
while still catching genuine, persistent price drift — which is the actual
signature of adverse selection.

We computed markout two ways, because they answer different questions:

- **A fixed number of trades** (the next 20 prints). Problem: in a fast
  market that's a matter of seconds; in a slow one it can be many hours —
  it does not hold real exposure time constant, so it doesn't represent a
  consistent "how fast can a market maker react" assumption. We measured
  this directly rather than assuming it: across the original test
  population, a 20-trade window spanned a **median of 3.3 hours** and a
  **mean of 22 hours**.
- **A fixed time window** (15 seconds). This is the more realistic proxy
  for "how long does it take a market maker to notice an adverse price
  move and pull or reprice a quote" — a fixed reaction latency rather than
  a fixed print count.

Both are reported side by side throughout our research rather than one
replacing the other, and the gap between them is itself informative: it
tells you how much of the naive "best case" number depends on unrealistic
holding time.

### 3.4 Realism constraints

Three additional guardrails keep the model from producing numbers that
don't correspond to anything achievable:

- **Per-trade position cap ($25 notional).** Stops a single unusually large
  trade print from dominating a market's entire result.
- **Relative-spread cap.** Polymarket prices run from near-$0 to near-$1;
  a flat dollar half-spread is nonsensical at the extremes (a 1-cent spread
  on a 1-cent-priced token is a 100%+ relative spread). The effective
  half-spread used is capped at 30% of the distance to the nearer boundary.
- **Liquidity-share cap (20% of a market's own total real trade volume).**
  Stops the model from implicitly assuming it was the dominant source of
  liquidity in a market it never actually quoted in. Across our final
  research population, this cap essentially never had to bind — actual
  captured volume stayed well under the ceiling (mean 9.7%, maximum 15%
  in the deepest analysis) — which is a reassuring sign that the headline
  numbers weren't being propped up by an unrealistic liquidity assumption.

---

## 4. What we looked for: which markets suit this strategy

Not every market is equally suited to market making. We investigated three
characteristics of a market that plausibly affect it:

- **Trading pace** — how frequently the market trades (median seconds
  between trades). Intuition: a market that trades too fast doesn't give a
  human or algorithmic quoter time to react to new information before
  getting run over; a market that trades too slowly may not generate enough
  volume to be worth quoting in at all.
- **Trading volume** — larger markets presumably offer more real opportunity
  to capture spread.
- **Proximity to resolution** — markets very close to resolving may be
  trading on information the true outcome is about to reveal (an
  "informational cascade"), which looks like ordinary market-making
  activity in the data but isn't.

Each of these produced an interesting pattern in our initial exploration.
The most important thing to understand about that exploration, covered in
the next section, is that **it does not, by itself, constitute evidence of
a real trading edge** — and our validation process exists specifically to
distinguish a real pattern from an artifact of how the data was examined.

---

## 5. Research process and why validation matters here

### 5.1 A bias we found in our own data

Our first pass reused a market population that had been built for a
*different* strategy — one that specifically trades markets where an
outcome's probability crosses 99%. That is not a representative sample of
markets a market-making desk would actually quote in; it's a narrow,
extreme-probability-skewed slice. We caught this, and rebuilt the research
population from scratch: a **stratified random sample of 1,517 markets**,
drawn proportionally across time period and market category from the
**complete population of 844,529 resolved Polymarket markets**, with no
probability-based filter of any kind. This is the population all further
results in this report are based on.

### 5.2 The overfitting risk, and how we tested for it

Once we had the right population, a second risk remained: if you try
several different market-selection rules (by pace, then by volume, then by
proximity to resolution) and keep whichever one looks best on the same data
you're evaluating it on, you will almost always find *something* that looks
good — purely by chance, not because it represents a real, repeatable
pattern. This is a well-known trap in quantitative research generally, and
guarding against it is standard practice at any serious trading desk.

Our validation process:

1. **Split the data by time.** The earlier 70% of markets (by resolution
   date) became the "training" set; the later 30% became a "test" set that
   was never looked at while choosing anything.
2. **Choose the market-selection rule using only the training set.** We let
   the model search over a range of pace, volume, and resolution-timing
   thresholds and pick whichever combination looked best — but strictly on
   training data only, exactly as a rule would have to be chosen in real
   time before deployment.
3. **Apply that exact rule, unchanged, to the test set** — data the rule
   selection process never saw — and report what actually happened.
4. **Repeat the whole process at several different strictness settings**,
   to check that the conclusion doesn't depend on one arbitrary choice.
5. **Statistically resample the test result** (2,000 times) to see how much
   of any given number is a real, stable pattern versus what could easily
   be explained by chance given how few markets were involved.

---

## 6. Findings

### 6.1 The headline pattern, holds up

Across every version of this analysis — the original population and the
rebuilt, unbiased one — the same basic story appears: a **naive spread-
capture estimate looks strongly profitable, and this looks materially worse
once a realistic adverse-selection cost is included.** On the unbiased
population's held-out test data, a best-case estimate of **+$8,399** across
345 markets became **-$13,239** once realistic markout was applied. That
part of the finding is consistent and, in our assessment, trustworthy — it
appeared independently on two different, non-overlapping populations.

### 6.2 The market-selection rules did not hold up

Our exploratory work suggested that markets with a *moderate* trading pace
(roughly one trade every 1–5 minutes), higher trading volume, and a longer
history of trading activity before resolution were meaningfully better
suited to this strategy — in some cuts, profitable even after realistic
adverse-selection costs.

When we tested this properly — training the selection rule on one period
and checking it on a later, untouched period — **it did not hold up.** The
rule the model chose from training data converged, every time, on a
*different and narrower* pattern (very fast-trading markets) than what the
exploratory work had suggested, looked attractive on the training data it
was chosen from, and then **lost money on the untouched test data at every
strictness setting we tried** — with an estimated probability of being
profitable between 0.2% and 1.1%. This was not a marginal or ambiguous
result; it was a consistent, repeated failure.

### 6.3 What this means

The market-selection patterns from our exploratory phase were **artifacts
of examining a limited dataset too many ways**, not a real, exploitable
signal. This is exactly the outcome our validation process is designed to
surface, and surfacing it before capital is committed is the intended
function of that process working correctly.

---

## 7. Limitations to keep in mind regardless of the outcome above

- **No real order-book data.** Every number here is reconstructed from
  trade prints, not measured against the order book a real resting quote
  would have interacted with. This is the single biggest source of
  uncertainty in the whole exercise and cannot be resolved with more
  backtesting of this kind — it requires live data.
- **No market-impact or capital-scaling model.** The model does not attempt
  to estimate how spreads and fill rates would change once real capital
  (and the market's other participants' reactions to it) enters the
  picture.
- **No portfolio-level risk model.** Every result is a sum of independent
  per-market outcomes; correlated risk across many simultaneous positions
  (for instance, many same-day sports markets, or a single crypto-market
  move affecting several positions at once) is not represented.
- **Execution-latency assumption is unverified.** The 15-second reaction
  window is a reasonable placeholder borrowed from elsewhere in our
  research, not a measurement of actual achievable latency on Polymarket's
  live infrastructure.
- **Fees and gas costs are not yet included** in this particular model —
  adding them would only make the already-negative validated result worse,
  not better.

---

## 8. Recommendation

**We do not recommend allocating capital to this strategy as currently
specified.** The validated, out-of-sample evidence does not support it, and
the failure was consistent across multiple independent checks rather than
borderline or ambiguous.

If there is interest in continuing this line of research, the productive
next steps, in order, are:

1. **Real order-book / live-quote data**, even a modest recorded sample, to
   replace the trade-print reconstruction this entire analysis has had to
   rely on.
2. **A small, capital-limited live pilot** rather than further backtesting,
   specifically to measure real fill rates and real execution latency — the
   two inputs this model has been estimating rather than observing.
3. **If a future rule is found that survives the same walk-forward testing
   described above**, layer in real trading fees, gas costs, and a
   market-impact model before sizing any capital against it.

We're glad to walk through any part of this analysis in more detail, or to
scope the live-data / pilot work described above.
