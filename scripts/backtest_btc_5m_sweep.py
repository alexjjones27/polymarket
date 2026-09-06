"""Sweep version of backtest_btc_5m_momentum.py: instead of one fixed
check-time (T-2.5min), reuses the SAME cached CLOB price series (already
on disk from the first run -- no new network calls) to test many
check-times x price-buckets at once, looking for ANY combination with a
statistically real edge.

Multiple-comparison warning: this tests ~9 check-times x ~10 buckets =
~90 combinations. At a naive p<0.05 threshold we'd expect ~4-5 "hits" by
chance alone even if the market is perfectly efficient everywhere. Ranks
by |z-score| and reports a Bonferroni-adjusted threshold alongside the
naive one -- treat anything that doesn't clear Bonferroni as a lead to
re-test out-of-sample, not a finding.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import polymarket_final_pct as pmf
import backtest_btc_5m_momentum as base

CHECK_TIMES_S = [30, 60, 90, 120, 150, 180, 210, 240, 270]
BUCKET_WIDTH = 0.05
MIN_BUCKET_N = 100  # ignore buckets too thin to say anything


def analyze_event_multi(event: dict) -> list[dict]:
    markets = event.get("markets") or []
    if not markets:
        return []
    m = markets[0]
    outcomes = pmf._safe_json_list(m.get("outcomes"))
    outcome_prices = pmf._safe_json_list(m.get("outcomePrices"))
    token_ids = pmf._safe_json_list(m.get("clobTokenIds"))
    if outcomes != ["Up", "Down"] or len(outcome_prices) != 2 or len(token_ids) != 2:
        return []
    try:
        up_resolved = float(outcome_prices[0]) == 1.0
    except (ValueError, TypeError):
        return []

    end_s = int(pmf.pd.Timestamp(event["endDate"]).timestamp())
    df = pmf.fetch_price_series(token_ids[0], end_s - 3600, end_s + 60, fidelity=1)  # cache hit, no network call
    if df.empty:
        return []

    out = []
    for secs_before in CHECK_TIMES_S:
        check_s = end_s - secs_before
        prior = df[df["t"] <= check_s]
        if prior.empty:
            continue
        staleness = check_s - int(prior.iloc[-1]["t"])
        if staleness > max(60, secs_before):  # scale tolerance with how far back we're looking
            continue
        p_up = float(prior.iloc[-1]["p"])
        if p_up >= 0.5:
            leading_side, leading_price, won = "Up", p_up, up_resolved
        else:
            leading_side, leading_price, won = "Down", 1.0 - p_up, not up_resolved
        out.append({"slug": event.get("slug"), "secs_before": secs_before,
                     "leading_side": leading_side, "leading_price": leading_price, "won": won})
    return out


def main():
    events = base.fetch_all_events(base.LOOKBACK_DAYS)
    print(f"{len(events)} events (reusing cached price series where available) ...")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    rows = []
    with ThreadPoolExecutor(max_workers=24) as pool:
        futures = {pool.submit(analyze_event_multi, e): e for e in events}
        for i, fut in enumerate(as_completed(futures)):
            try:
                r = fut.result()
            except Exception:
                r = None
            if r:
                rows.extend(r)
            if (i + 1) % 500 == 0:
                print(f"  processed {i+1}/{len(events)} ...", flush=True)

    df = pmf.pd.DataFrame(rows)
    df["bucket"] = (df["leading_price"] // BUCKET_WIDTH) * BUCKET_WIDTH
    print(f"\n{len(df)} total (event, check-time) observations\n")

    results = []
    for (secs_before, bucket), g in df.groupby(["secs_before", "bucket"]):
        n = len(g)
        if n < MIN_BUCKET_N:
            continue
        wins = int(g["won"].sum())
        win_rate = wins / n
        implied_mid = bucket + BUCKET_WIDTH / 2
        edge = win_rate - implied_mid
        se = (win_rate * (1 - win_rate) / n) ** 0.5
        z = edge / se if se > 0 else 0.0
        results.append({"secs_before": secs_before, "bucket": bucket, "n": n, "win_rate": win_rate,
                         "implied_mid": implied_mid, "edge": edge, "se": se, "z": z})

    results.sort(key=lambda r: abs(r["z"]), reverse=True)
    n_tests = len(results)
    from statistics import NormalDist
    alpha_naive = 0.05
    alpha_bonf = alpha_naive / max(n_tests, 1)
    z_naive = NormalDist().inv_cdf(1 - alpha_naive / 2)
    z_bonf = NormalDist().inv_cdf(1 - alpha_bonf / 2)
    print(f"{n_tests} (check-time, bucket) combinations tested with n>={MIN_BUCKET_N}.")
    print(f"Naive |z| > {z_naive:.2f} ~ p<0.05 uncorrected. Bonferroni |z| > {z_bonf:.2f} for family-wise p<0.05.\n")

    print(f"{'secs_before':>11} {'bucket':>8} {'n':>6} {'win_rate':>9} {'implied':>8} {'edge':>8} {'z':>6}")
    for r in results[:25]:
        flag = " **BONFERRONI**" if abs(r["z"]) > z_bonf else (" *naive*" if abs(r["z"]) > z_naive else "")
        print(f"{r['secs_before']:>11} {r['bucket']:>8.2f} {r['n']:>6} {r['win_rate']:>9.3f} "
              f"{r['implied_mid']:>8.3f} {r['edge']:>+8.3f} {r['z']:>6.2f}{flag}")

    out_dir = Path(__file__).resolve().parents[1] / "results" / "btc_5m_momentum"
    out_dir.mkdir(parents=True, exist_ok=True)
    pmf.pd.DataFrame(results).to_csv(out_dir / "sweep_summary.csv", index=False)
    df.to_csv(out_dir / "sweep_observations.csv", index=False)
    print(f"\nSaved {out_dir / 'sweep_summary.csv'} and {out_dir / 'sweep_observations.csv'}")


if __name__ == "__main__":
    main()
