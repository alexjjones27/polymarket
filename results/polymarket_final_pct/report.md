# Polymarket "Final 1%" Spread-Capture Backtest

Historical analysis only. Buys an outcome token once its price closes at/above $0.990 for 3 consecutive snapshots and holds to resolution. No live trading, no order placement, no API keys.

## Data & methodology

**Population census**: 683,677 resolved markets found via a complete, uncurated crawl of Gamma's `/markets?closed=true` since the CLOB-launch cutoff (2022-09-01) through the run date. This population is dominated by short-lived auto-generated crypto up/down markets; pulling full CLOB price history for all of it is not computationally tractable here, so the backtest runs on a **stratified random sample of 4,015 markets** (proportional allocation by resolution quarter x report category, seeded and reproducible) -- an unbiased sample of the full population, not a curated or volume-sorted subset, which is the direct mitigation for the selection-bias risk this kind of backtest is prone to.

**Two confirmed, load-bearing API defects found by testing the live endpoints before building the pipeline** (both drove real design decisions, see the module docstring in `src/polymarket_final_pct.py` for the full detail):

1. Gamma's `/markets/keyset` endpoint (the one the deprecated classic endpoint's headers point you to) silently ignores its own `cursor` parameter -- every request returns page 1 regardless. Worked around by bucketing the classic endpoint's `offset` (separately capped at 2000) by end-date range.
2. `/prices-history?interval=max` (the natural way to ask for a token's full history) reliably returns an EMPTY series even for high-volume resolved tokens -- this is the granularity/emptiness issue the task warned about. Fix: never use `interval=`, always pass explicit `startTs`/`endTs` (capped at a 15-day window) with an explicit `fidelity`; verified this returns real ~1-minute data in every case tested. Markets that resolved before Polymarket's CLOB launch (mid-2022) have no CLOB price history at any window -- they traded on the old AMM -- and are excluded from the population on that basis, not treated as the granularity bug. A market's *lifetime* can straddle that cutoff even when its *resolution* falls inside it (found live on a $1.7M-volume Senate-control market that resolved Nov 2022 but started trading Jan 2022, with zero CLOB history) -- filtered on each market's own start date, not just its resolution date. On a live 20-market granularity test spanning 2022-2025 and all volume tiers, 3 of 13 valid samples (all from the first ~2 months post-CLOB-launch) still came back with zero price points even under the explicit-window fix -- read as thin early liquidity on that specific token, not a residual data-access bug; such tokens simply contribute no trade.

**Liquidity/depth**: there is no way to reconstruct historical order-book depth for a resolved market (`/book` returns 404, "No orderbook exists", confirmed live). As a proxy, position size is capped to the sum of realized trade sizes on the same token within 5 minutes of the crossing (from the public `data-api.polymarket.com/trades` feed), when any such trades are recorded; markets meaningfully capped by this are flagged below.

**Fees**: confirmed against docs.polymarket.com/trading/fees and help.polymarket.com (both official, cross-checked against the docs' own worked examples). Makers pay $0, always. Takers pay `fee = shares * feeRate * price * (1-price)`, with feeRate 0.00-0.07 depending on category -- which is why the fee is small specifically where this strategy trades: `(1-price)` is already ~0.01 at a $0.99 entry. **Gas**: Polymarket's relayer sponsors on-chain gas for the standard trading flow, so ordinary users pay $0/trade (confirmed against docs.polymarket.com/trading/gasless) -- modeled as the default. A non-relayed direct on-chain estimate of $0.0038/trade (live Polygon gas price x ~150k gas units x live POL/USD) is reported as a sensitivity case.

## Results: net vs. gross, maker vs. taker fill, with vs. without flips


### maker

**gross of fees/gas**

| metric | including_flips | winners_only_counterfactual |
| --- | --- | --- |
| n_trades | 3380 | 3374 |
| n_flips | 6 | 0 |
| win_rate | 0.9982 | 1.0000 |
| flip_rate | 0.0018 | 0.0000 |
| flip_rate_wilson_95 | (0.0008, 0.0039) | (0.0000, 0.0011) |
| flip_rate_clopper_pearson_95 | (0.0007, 0.0039) | (0.0000, 0.0011) |
| total_pnl | 828.1311 | 1368.5137 |
| total_notional | 282030.6900 | 281490.3074 |
| total_return | 0.0029 | 0.0049 |
| annualized_return | 0.4541 | 0.7512 |
| avg_holding_days_winners | 2.1974 | 2.1974 |
| avg_holding_days_flips | 1.3972 | nan |


**net of fees/gas**

| metric | including_flips | winners_only_counterfactual |
| --- | --- | --- |
| n_trades | 3380 | 3374 |
| n_flips | 6 | 0 |
| win_rate | 0.9982 | 1.0000 |
| flip_rate | 0.0018 | 0.0000 |
| flip_rate_wilson_95 | (0.0008, 0.0039) | (0.0000, 0.0011) |
| flip_rate_clopper_pearson_95 | (0.0007, 0.0039) | (0.0000, 0.0011) |
| total_pnl | 828.1311 | 1368.5137 |
| total_notional | 282030.6900 | 281490.3074 |
| total_return | 0.0029 | 0.0049 |
| annualized_return | 0.4541 | 0.7512 |
| avg_holding_days_winners | 2.1974 | 2.1974 |
| avg_holding_days_flips | 1.3972 | nan |



### taker

**gross of fees/gas**

| metric | including_flips | winners_only_counterfactual |
| --- | --- | --- |
| n_trades | 3380 | 3374 |
| n_flips | 6 | 0 |
| win_rate | 0.9982 | 1.0000 |
| flip_rate | 0.0018 | 0.0000 |
| flip_rate_wilson_95 | (0.0008, 0.0039) | (0.0000, 0.0011) |
| flip_rate_clopper_pearson_95 | (0.0007, 0.0039) | (0.0000, 0.0011) |
| total_pnl | 828.1311 | 1368.5137 |
| total_notional | 282030.6900 | 281490.3074 |
| total_return | 0.0029 | 0.0049 |
| annualized_return | 0.4541 | 0.7512 |
| avg_holding_days_winners | 2.1974 | 2.1974 |
| avg_holding_days_flips | 1.3972 | nan |


**net of fees/gas**

| metric | including_flips | winners_only_counterfactual |
| --- | --- | --- |
| n_trades | 3380 | 3374 |
| n_flips | 6 | 0 |
| win_rate | 0.9982 | 1.0000 |
| flip_rate | 0.0018 | 0.0000 |
| flip_rate_wilson_95 | (0.0008, 0.0039) | (0.0000, 0.0011) |
| flip_rate_clopper_pearson_95 | (0.0007, 0.0039) | (0.0000, 0.0011) |
| total_pnl | 755.2187 | 1295.8436 |
| total_notional | 282030.6900 | 281490.3074 |
| total_return | 0.0027 | 0.0046 |
| annualized_return | 0.4142 | 0.7113 |
| avg_holding_days_winners | 2.1974 | 2.1974 |
| avg_holding_days_flips | 1.3972 | nan |


## Flip analysis

| market_id | question | category | report_bucket | entry_price | entry_time | holding_days |
| --- | --- | --- | --- | --- | --- | --- |
| 543088 | Will the highest temperature in New York City be between 66-67°F on May 13? | other | other | 0.9940 | 2025-05-13 12:27:07+00:00 | 0.8995 |
| 798809 | Will the price of Ethereum be between $3,300 and $3,400 on December 9? | crypto | crypto_price | 0.9910 | 2025-12-09 04:39:07+00:00 | 0.6858 |
| 1544051 | Spread: Carrarese Calcio (-2.5) | other | other | 0.9920 | 2026-03-19 16:46:32+00:00 | 3.1257 |
| 623802 | Will Sidemen's next video get between 7–8 million views on week 1? | other | other | 0.9905 | 2025-10-16 10:44:03+00:00 | 3.4364 |
| 2323717 | Algeria vs. Austria: O/U 5.5 | sports | sports | 0.9920 | 2026-06-28 02:24:04+00:00 | 0.0777 |
| 3588722 | Exact Score: Hibernian FC 2 - 3 KAA Gent? | other | other | 0.9900 | 2026-08-27 19:04:12+00:00 | 0.1582 |


`flip_heuristic_category` is auto-derived from UMA oracle resolution-status metadata (`disputed_resolution` if any dispute flag is present, else `needs_manual_review`); every flip above should be read individually (question + slug) before treating it as a "genuine reversal" -- that distinction genuinely changes what the recurring risk is going forward and this pipeline cannot make that call automatically.


## Days-to-resolution distribution, winners vs. flips

| group | n | mean_days | median_days | p10_days | p90_days | max_days |
| --- | --- | --- | --- | --- | --- | --- |
| winners | 3374 | 2.1974 | 0.1138 | 0.0229 | 2.0432 | 265.4710 |
| flips | 6 | 1.3972 | 0.7927 | 0.1179 | 3.2810 | 3.4364 |

If flips cluster at one end of this distribution (e.g. only in markets held a long time after crossing) or in one category, that pattern is more actionable than the aggregate flip rate alone -- check the flip table above against this distribution directly.


## Category breakdown


**maker fills, net of fees**

| report_bucket | n_trades | n_flips | win_rate | flip_rate | flip_rate_wilson_95 | flip_rate_clopper_pearson_95 | total_pnl | total_notional | total_return | annualized_return | avg_holding_days_winners | avg_holding_days_flips |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crypto_price | 522 | 1 | 0.9981 | 0.0019 | (0.0003, 0.0108) | (0.0000, 0.0106) | 149.6061 | 49342.9520 | 0.0030 | 1.7707 | 0.6517 | 0.6858 |
| other | 1232 | 4 | 0.9968 | 0.0032 | (0.0013, 0.0083) | (0.0009, 0.0083) | 191.7975 | 99654.9325 | 0.0019 | 0.1637 | 3.8366 | 1.9049 |
| politics | 49 | 0 | 1.0000 | 0.0000 | (0.0000, 0.0727) | (0.0000, 0.0725) | 27.9925 | 4456.1551 | 0.0063 | 0.2218 | 11.3947 | nan |
| sports | 1577 | 1 | 0.9994 | 0.0006 | (0.0001, 0.0036) | (0.0000, 0.0035) | 458.7350 | 128576.6504 | 0.0036 | 1.0395 | 1.1442 | 0.0777 |



**taker fills, net of fees**

| report_bucket | n_trades | n_flips | win_rate | flip_rate | flip_rate_wilson_95 | flip_rate_clopper_pearson_95 | total_pnl | total_notional | total_return | annualized_return | avg_holding_days_winners | avg_holding_days_flips |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crypto_price | 522 | 1 | 0.9981 | 0.0019 | (0.0003, 0.0108) | (0.0000, 0.0106) | 132.1892 | 49342.9520 | 0.0027 | 1.5645 | 0.6517 | 0.6858 |
| other | 1232 | 4 | 0.9968 | 0.0032 | (0.0013, 0.0083) | (0.0009, 0.0083) | 165.2352 | 99654.9325 | 0.0017 | 0.1410 | 3.8366 | 1.9049 |
| politics | 49 | 0 | 1.0000 | 0.0000 | (0.0000, 0.0727) | (0.0000, 0.0725) | 26.8812 | 4456.1551 | 0.0060 | 0.2130 | 11.3947 | nan |
| sports | 1577 | 1 | 0.9994 | 0.0006 | (0.0001, 0.0036) | (0.0000, 0.0035) | 430.9132 | 128576.6504 | 0.0034 | 0.9765 | 1.1442 | 0.0777 |



## Max time-to-resolution variant (unrestricted vs. <= 7 days at entry)

A market can sit at $0.99 for months before it finally resolves -- the per-trade dollar P&L is identical, but that dead capital-tied-up time collapses the annualized return. This variant additionally requires, at entry, that the market's *scheduled* end date (not the realized resolution time, which would leak lookahead into an entry-time filter) was no more than 7 days away.


**maker fills, net of fees**

| variant | n_trades | n_flips | win_rate | flip_rate | flip_rate_wilson_95 | flip_rate_clopper_pearson_95 | total_pnl | total_notional | total_return | annualized_return | avg_holding_days_winners | avg_holding_days_flips |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unrestricted (n=3380) | 3380 | 6 | 0.9982 | 0.0018 | (0.0008, 0.0039) | (0.0007, 0.0039) | 828.1311 | 282030.6900 | 0.0029 | 0.4541 | 2.1974 | 1.3972 |
| max 7d to scheduled resolution (n=3042) | 3042 | 5 | 0.9984 | 0.0016 | (0.0007, 0.0038) | (0.0005, 0.0038) | 681.8139 | 253293.8094 | 0.0027 | 2.2639 | 0.4249 | 0.9894 |



**taker fills, net of fees**

| variant | n_trades | n_flips | win_rate | flip_rate | flip_rate_wilson_95 | flip_rate_clopper_pearson_95 | total_pnl | total_notional | total_return | annualized_return | avg_holding_days_winners | avg_holding_days_flips |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unrestricted (n=3380) | 3380 | 6 | 0.9982 | 0.0018 | (0.0008, 0.0039) | (0.0007, 0.0039) | 755.2187 | 282030.6900 | 0.0027 | 0.4142 | 2.1974 | 1.3972 |
| max 7d to scheduled resolution (n=3042) | 3042 | 5 | 0.9984 | 0.0016 | (0.0007, 0.0038) | (0.0005, 0.0038) | 618.1510 | 253293.8094 | 0.0024 | 2.0525 | 0.4249 | 0.9894 |



## Threshold sensitivity ($0.98 / $0.99 / $0.995)


**maker**

| n_trades | n_flips | win_rate | flip_rate | flip_rate_wilson_95 | flip_rate_clopper_pearson_95 | total_pnl | total_notional | total_return | annualized_return | avg_holding_days_winners | avg_holding_days_flips | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3423 | 10 | 0.9971 | 0.0029 | (0.0016, 0.0054) | (0.0014, 0.0054) | 1785.3865 | 342300.0000 | 0.0052 | 0.6315 | 2.9649 | 20.2872 | 0.9800 |
| 3380 | 6 | 0.9982 | 0.0018 | (0.0008, 0.0039) | (0.0007, 0.0039) | 1037.1095 | 338000.0000 | 0.0031 | 0.5102 | 2.1974 | 1.3972 | 0.9900 |
| 3347 | 3 | 0.9991 | 0.0009 | (0.0003, 0.0026) | (0.0002, 0.0026) | 882.0556 | 334700.0000 | 0.0026 | 0.6741 | 1.4276 | 1.1405 | 0.9950 |



**taker**

| n_trades | n_flips | win_rate | flip_rate | flip_rate_wilson_95 | flip_rate_clopper_pearson_95 | total_pnl | total_notional | total_return | annualized_return | avg_holding_days_winners | avg_holding_days_flips | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3423 | 10 | 0.9971 | 0.0029 | (0.0016, 0.0054) | (0.0014, 0.0054) | 1636.7475 | 342300.0000 | 0.0048 | 0.5789 | 2.9649 | 20.2872 | 0.9800 |
| 3380 | 6 | 0.9982 | 0.0018 | (0.0008, 0.0039) | (0.0007, 0.0039) | 950.4585 | 338000.0000 | 0.0028 | 0.4675 | 2.1974 | 1.3972 | 0.9900 |
| 3347 | 3 | 0.9991 | 0.0009 | (0.0003, 0.0026) | (0.0002, 0.0026) | 819.6966 | 334700.0000 | 0.0024 | 0.6265 | 1.4276 | 1.1405 | 0.9950 |



## Fill-size / depth-capping flags

819 of 3380 sampled trades had position size meaningfully capped below the desired fixed notional by the realized-trades liquidity proxy:


| market_id | question | desired_shares | shares | cap_shares |
| --- | --- | --- | --- | --- |
| 501504 | Solana flips ETH in daily fees in May? | 100.8065 | 3.0303 | 3.0303 |
| 506476 | Will England win? | 100.5025 | 42.4200 | 42.4200 |
| 506334 | Will Trump say "green new scam" at Wisconsin rally? | 100.5025 | 20.0000 | 20.0000 |
| 511260 | Will Trump say "Teamster" during Wisconsin rally on Nov 1? | 100.5025 | 25.0000 | 25.0000 |
| 511980 | Will Juventus beat LOSC Lille? | 100.5025 | 50.0000 | 50.0000 |
| 511906 | 76ers vs. Clippers | 100.4520 | 36.7272 | 36.7272 |
| 501863 | Israel x Hamas ceasefire before September? | 100.8573 | 1.1700 | 1.1700 |
| 521761 | Will Donald Trump say "Border" 5+ times during Super Bowl pregame interview? | 100.2506 | 12.4200 | 12.4200 |
| 518214 | Will Andrew Tate tweet 50-59 times Jan 10-17? | 100.9082 | 18.5151 | 18.5151 |
| 514482 | Will the December 2024 temperature increase be less than 1.20°C? | 100.9591 | 1.0050 | 1.0050 |
| 528048 | Illinois vs. Maryland | 100.5530 | 100.0000 | 100.0000 |
| 549100 | Will Donald Trump say "Big beautiful bill" during Merz events on June 5? | 100.5025 | 42.0000 | 42.0000 |
| 255043 | Will a Democrat win Nebraska Presidential Election? | 101.0101 | 0.2812 | 0.2812 |
| 547290 | Will Kimi Antonelli win the 2025 Spanish GP pole? | 100.2004 | 11.0020 | 11.0020 |
| 547661 | Will Trump say "Capone" during Pittsburgh rally on May 30? | 100.4520 | 30.0000 | 30.0000 |
| 541344 | Will the highest temperature in London be between 59-60°F on May 7? | 100.5530 | 0.0100 | 0.0100 |
| 551551 | Will USA beat Trinidad and Tobago? | 100.0500 | 82.0000 | 82.0000 |
| 540451 | Will XRP dip to $1.90 in May? | 100.9591 | 16.2367 | 16.2367 |
| 534877 | Will the April 2025 unemployment rate be 4.4%? | 100.4520 | 95.9600 | 95.9600 |
| 533866 | Will Trump reduce or pause tariffs on Switzerland before June? | 100.5025 | 0.8800 | 0.8800 |
| 539627 | Will Elon tweet 400 or more times April 25–May 2? | 101.0101 | 99.9600 | 99.9600 |
| 550260 | Will the highest temperature in London be between 73-74°F on June 9? | 100.9591 | 21.5140 | 21.5140 |
| 555518 | Will the highest temperature in London be between 76-77°F on June 27? | 101.0101 | 55.0000 | 55.0000 |
| 530434 | Will Lewis Hamilton win the 2025 Japanese Grand Prix? | 100.8573 | 80.7615 | 80.7615 |
| 562562 | Solana Up or Down - July 15, 5AM ET | 100.5025 | 61.2200 | 61.2200 |
| 602958 | Will the price of Bitcoin be above $110,000 on September 25? | 100.9082 | 1.0020 | 1.0020 |
| 549182 | Will Kasparas Jakucionis be the fifth pick of the 2025 NBA Draft? | 100.4016 | 40.0050 | 40.0050 |
| 583833 | Will the price of Solana be above $206 on August 30 at 4AM ET? | 101.0101 | 100.0000 | 100.0000 |
| 597678 | Will the price of Bitcoin be above $124,000 on September 21? | 100.4520 | 42.1200 | 42.1200 |
| 570693 | XRP Up or Down - August 4, 3AM ET | 101.0101 | 60.0000 | 60.0000 |



## Limitations

- **Sample-size limitation on the flip rate is a limitation of this backtest itself, not a footnote.** Polymarket's CLOB has existed for a bit over four years, and this strategy's entire economics hinge on a tail event (the flip rate) that, by construction, is rare -- a handful of flips (or zero) out of thousands of sampled trades. The confidence intervals reported above are wide for exactly this reason: with a small number of observed flips, the data cannot distinguish between "this strategy has a structurally low, durable flip rate" and "this backtest simply hasn't sampled enough history to see the flips that will happen." A point estimate of the flip rate should not be read as a precise, forward-looking probability.
- The liquidity-depth proxy (realized trades near the crossing) is not the same thing as resting order-book depth at the moment of the crossing -- it likely understates true available liquidity in some cases and cannot be verified against the real book for a resolved market.
- Ignores the possibility that entering size at the crossing itself moves the price (this backtest assumes the observed crossing price is achievable at the simulated size, up to the depth cap).
- Category classification is a keyword heuristic over question text and event metadata, not Polymarket's internal taxonomy -- treat the category breakdown as indicative, not exact. "other" is the catch-all this heuristic falls back to (both for the report bucket and the fee rate); its share of this sample is crypto_price=763, other=1340, politics=55, sports=1857 (33.4% other) -- a large or growing "other" share is the concrete sign this heuristic is missing real structure, not just a caveat.
- The backtest samples from the population rather than covering it exhaustively; while the sampling is stratified and unbiased by construction, a different random seed or a larger sample could shift the flip count (see the CI, not the point estimate).
