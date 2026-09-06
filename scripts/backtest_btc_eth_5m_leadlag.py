"""Properly controlled version of backtest_btc_eth_5m_crossasset.py.

The earlier bucket-by-spread test had a confound: averaging "eth_edge"
within a spread bucket mixes windows where ETH's own price varies, so an
apparent spread effect could just be ETH's own calibration curve bending,
misattributed to BTC. This fits, at each of 9 check-times (30s to 270s
before resolution), a logistic regression:

    eth_up_resolved ~ p_eth_up + p_btc_up
    btc_up_resolved ~ p_btc_up + p_eth_up

The coefficient on the OTHER asset's price, after controlling for the
asset's own price, is the real test of "does the other market carry
incremental information" -- i.e. any type of spread, not one specific
bucketing of it. Reuses cached price series for both assets (no new
network calls) across all 9 checkpoints matching backtest_btc_5m_sweep.py.

18 total regressions (9 times x 2 directions) -- reports z-scores with
both naive (p<0.05) and Bonferroni-corrected significance flags.
"""
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import NormalDist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import polymarket_final_pct as pmf
import backtest_btc_5m_momentum as base
import backtest_btc_eth_5m_crossasset as cross

CHECK_TIMES_S = [30, 60, 90, 120, 150, 180, 210, 240, 270]


def build_matched(secs_before: int) -> "pmf.pd.DataFrame":
    btc_events = base.fetch_all_events(base.LOOKBACK_DAYS)
    eth_events = cross.fetch_eth_events(base.LOOKBACK_DAYS)
    with ThreadPoolExecutor(max_workers=24) as pool:
        btc_by_window = {}
        for fut in as_completed({pool.submit(cross.event_price_and_outcome, e, secs_before): e for e in btc_events}):
            r = fut.result()
            if r:
                btc_by_window[r["window_end"]] = r
        eth_by_window = {}
        for fut in as_completed({pool.submit(cross.event_price_and_outcome, e, secs_before): e for e in eth_events}):
            r = fut.result()
            if r:
                eth_by_window[r["window_end"]] = r

    rows = []
    for w, b in btc_by_window.items():
        e = eth_by_window.get(w)
        if e:
            rows.append({"window_end": w, "p_btc_up": b["p_up"], "p_eth_up": e["p_up"],
                         "btc_up_resolved": int(b["up_resolved"]), "eth_up_resolved": int(e["up_resolved"])})
    return pmf.pd.DataFrame(rows)


def fit_logit(df, y_col, own_col, other_col):
    """Manual Newton-Raphson logistic regression (2 predictors + intercept)
    to avoid adding a statsmodels dependency just for this. Returns
    (coef_other, se_other, z_other)."""
    import numpy as np
    X = np.column_stack([np.ones(len(df)), df[own_col].values, df[other_col].values])
    y = df[y_col].values.astype(float)
    beta = np.zeros(3)
    for _ in range(50):
        eta = X @ beta
        p = 1 / (1 + np.exp(-eta))
        w = p * (1 - p)
        w = np.clip(w, 1e-6, None)
        W = X * w[:, None]
        hessian = X.T @ W
        grad = X.T @ (y - p)
        try:
            delta = np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            return None
        beta += delta
        if np.max(np.abs(delta)) < 1e-8:
            break
    try:
        cov = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        return None
    se = np.sqrt(np.diag(cov))
    return beta[2], se[2], beta[2] / se[2] if se[2] > 0 else 0.0


def main():
    results = []
    for secs_before in CHECK_TIMES_S:
        df = build_matched(secs_before)
        print(f"T-{secs_before}s: {len(df)} matched windows", flush=True)
        if len(df) < 200:
            continue

        r = fit_logit(df, "eth_up_resolved", "p_eth_up", "p_btc_up")
        if r:
            coef, se, z = r
            results.append({"secs_before": secs_before, "target": "eth_up_resolved",
                             "own": "p_eth_up", "other": "p_btc_up", "coef_other": coef, "se": se, "z": z, "n": len(df)})

        r = fit_logit(df, "btc_up_resolved", "p_btc_up", "p_eth_up")
        if r:
            coef, se, z = r
            results.append({"secs_before": secs_before, "target": "btc_up_resolved",
                             "own": "p_btc_up", "other": "p_eth_up", "coef_other": coef, "se": se, "z": z, "n": len(df)})

    n_tests = len(results)
    alpha_naive = 0.05
    z_naive = NormalDist().inv_cdf(1 - alpha_naive / 2)
    z_bonf = NormalDist().inv_cdf(1 - (alpha_naive / max(n_tests, 1)) / 2)
    print(f"\n{n_tests} regressions run. Naive |z|>{z_naive:.2f}, Bonferroni |z|>{z_bonf:.2f}\n")

    results.sort(key=lambda r: abs(r["z"]), reverse=True)
    print(f"{'secs_before':>11} {'target':>16} {'other_var':>10} {'n':>6} {'coef':>8} {'z':>7}")
    for r in results:
        flag = " **BONFERRONI**" if abs(r["z"]) > z_bonf else (" *naive*" if abs(r["z"]) > z_naive else "")
        print(f"{r['secs_before']:>11} {r['target']:>16} {r['other']:>10} {r['n']:>6} "
              f"{r['coef_other']:>+8.3f} {r['z']:>7.2f}{flag}")

    out_path = Path(__file__).resolve().parents[1] / "results" / "btc_5m_momentum" / "leadlag_regressions.csv"
    pmf.pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
