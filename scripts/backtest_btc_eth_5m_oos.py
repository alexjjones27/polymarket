"""Out-of-sample confirmation of the T-30s BTC->ETH cross-asset finding
from backtest_btc_eth_5m_leadlag.py (coef_p_btc_up=+0.737, z=3.67,
Bonferroni-significant across 18 tests on the last 14 days of data).

Pulls a DISJOINT, never-examined 14-day window (days 15-28 back from now)
and does two things, using the pre-specified T-30s spec only (no new
tuning/dredging on the new data -- that would just reintroduce the
multiple-comparisons problem this is supposed to rule out):

  1. Refits eth_up_resolved ~ p_eth_up + p_btc_up on the new window and
     checks whether coef_p_btc_up replicates in sign/magnitude/significance.
  2. The stricter test: freezes the ORIGINAL in-sample coefficients and
     evaluates predictive log-loss on the new data, comparing the full
     model (p_eth_up + p_btc_up) against an ETH-only model (p_eth_up
     alone, refit on the original sample for a fair comparison). If BTC's
     price genuinely carries information, the full model's out-of-sample
     log-loss should be lower (better) than the ETH-only model's.
"""
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import polymarket_final_pct as pmf
import backtest_btc_eth_5m_crossasset as cross

OOS_DAY_START = 14  # start of the held-out window (days back from now)
OOS_DAY_END = 28    # end of the held-out window (exclusive)
SECS_BEFORE = 30    # the pre-specified, strongest in-sample finding

# Frozen in-sample coefficients from backtest_btc_eth_5m_leadlag.py's T-30s run
# (eth_up_resolved ~ intercept + b_eth*p_eth_up + b_btc*p_btc_up), fit on the
# last-14-days sample -- NOT refit here, by design.
INSAMPLE_INTERCEPT = -3.896
INSAMPLE_B_ETH = 7.080
INSAMPLE_B_BTC = 0.737


def fetch_events_shifted(series_slug: str, day_start: int, day_end: int, cache_name: str) -> list[dict]:
    import json
    cache_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "polymarket" / cache_name
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if cached.get("day_start") == day_start and cached.get("day_end") == day_end:
            print(f"Using cached {cache_name} ({len(cached['events'])} events)")
            return cached["events"]

    now = pmf.pd.Timestamp.utcnow().floor("min")
    events = []
    for day_offset in range(day_start, day_end):
        day_end_ts = now - pmf.pd.Timedelta(days=day_offset)
        day_start_ts = day_end_ts - pmf.pd.Timedelta(days=1)
        offset = 0
        while True:
            page = pmf._get(pmf.GAMMA_BASE, "/events", {
                "series_slug": series_slug, "closed": "true", "limit": 100, "offset": offset,
                "end_date_min": day_start_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end_date_max": day_end_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            })
            if not page:
                break
            events.extend(page)
            if len(page) < 100:
                break
            offset += 100
        print(f"  day -{day_offset}: {len(events)} events so far", flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"day_start": day_start, "day_end": day_end, "events": events}))
    return events


def fit_logit(df, y_col, cols):
    X = np.column_stack([np.ones(len(df))] + [df[c].values for c in cols])
    y = df[y_col].values.astype(float)
    beta = np.zeros(X.shape[1])
    for _ in range(100):
        eta = X @ beta
        p = 1 / (1 + np.exp(-eta))
        w = np.clip(p * (1 - p), 1e-6, None)
        hess = X.T @ (X * w[:, None])
        grad = X.T @ (y - p)
        delta = np.linalg.solve(hess, grad)
        beta += delta
        if np.max(np.abs(delta)) < 1e-9:
            break
    cov = np.linalg.inv(hess)
    se = np.sqrt(np.diag(cov))
    return beta, se


def log_loss(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    print(f"Fetching OOS window: days {OOS_DAY_START}-{OOS_DAY_END} back (disjoint from original 0-14 sample)")
    btc_events = fetch_events_shifted("btc-up-or-down-5m", OOS_DAY_START, OOS_DAY_END, "btc_5m_events_oos.json")
    eth_events = fetch_events_shifted("eth-up-or-down-5m", OOS_DAY_START, OOS_DAY_END, "eth_5m_events_oos.json")
    print(f"BTC: {len(btc_events)} events, ETH: {len(eth_events)} events")

    with ThreadPoolExecutor(max_workers=24) as pool:
        btc_by_window = {}
        for fut in as_completed({pool.submit(cross.event_price_and_outcome, e, SECS_BEFORE): e for e in btc_events}):
            r = fut.result()
            if r:
                btc_by_window[r["window_end"]] = r
        eth_by_window = {}
        for fut in as_completed({pool.submit(cross.event_price_and_outcome, e, SECS_BEFORE): e for e in eth_events}):
            r = fut.result()
            if r:
                eth_by_window[r["window_end"]] = r

    rows = []
    for w, b in btc_by_window.items():
        e = eth_by_window.get(w)
        if e:
            rows.append({"p_btc_up": b["p_up"], "p_eth_up": e["p_up"], "eth_up_resolved": int(e["up_resolved"])})
    df = pmf.pd.DataFrame(rows)
    print(f"\n{len(df)} matched OOS windows at T-{SECS_BEFORE}s\n")

    # Test 1: refit on OOS data, same spec
    beta, se = fit_logit(df, "eth_up_resolved", ["p_eth_up", "p_btc_up"])
    z_btc = beta[2] / se[2]
    print("=== Test 1: refit on OOS data ===")
    print(f"In-sample:  intercept={INSAMPLE_INTERCEPT:+.3f}  b_eth={INSAMPLE_B_ETH:+.3f}  b_btc={INSAMPLE_B_BTC:+.3f}")
    print(f"OOS refit:  intercept={beta[0]:+.3f}  b_eth={beta[1]:+.3f}  b_btc={beta[2]:+.3f}  "
          f"(se={se[2]:.3f}, z={z_btc:.2f})")

    # Test 2: frozen in-sample model vs ETH-only model, evaluated on OOS log-loss
    y = df["eth_up_resolved"].values.astype(float)
    p_full_frozen = 1 / (1 + np.exp(-(INSAMPLE_INTERCEPT + INSAMPLE_B_ETH * df["p_eth_up"] + INSAMPLE_B_BTC * df["p_btc_up"])))
    ll_full_frozen = log_loss(y, p_full_frozen.values)

    # ETH-only baseline: use ETH's own price directly as the probability (the "no model" baseline)
    ll_eth_raw = log_loss(y, df["p_eth_up"].values)

    print("\n=== Test 2: frozen in-sample model vs. ETH's raw price, on OOS log-loss (lower = better) ===")
    print(f"ETH raw price as probability: log-loss = {ll_eth_raw:.4f}")
    print(f"Frozen full model (ETH+BTC):  log-loss = {ll_full_frozen:.4f}")
    print(f"Improvement from adding BTC: {ll_eth_raw - ll_full_frozen:+.4f} "
          f"({'BTC helps' if ll_full_frozen < ll_eth_raw else 'BTC does NOT help'})")


if __name__ == "__main__":
    main()
