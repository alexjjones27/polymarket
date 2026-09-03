"""Backtest: short-horizon log-odds momentum (and its mirror, reversal).

Tests the momentum hypothesis directly: after a large, volume-confirmed
price move, does a Polymarket contract keep moving in the same direction
over the next 15-60 minutes (momentum), or tend to give it back
(reversal)? Both directions are run from the same signal and the same
population so the comparison is apples-to-apples, per the request this
was built against.

Population: Bitcoin/Ethereum/Solana/XRP "Up or Down" markets -- this
project's largest short-lived, continuously-traded population (~11% of
the full census), post-CLOB-cutoff, cleanly resolved. Chosen specifically
because a 1-3 day lifetime makes full-lifetime 1-minute price granularity
cheap to fetch (thousands of points in under a second per market, vs.
the coarse 60-minute bars used elsewhere in this project for markets that
live for months) -- and because "does a move keep moving" needs minute
granularity to even ask the question.

Signal: log-odds momentum M_h = logit(p_t) - logit(p_{t-h}) at h=5min and
h=30min, matching the request's own reasoning for using log-odds over raw
cents (a 50c->53c move and a 90c->93c move are not the same economic
event). Confirmed by trailing 5-minute trade volume relative to that
market's own full-lifetime average -- a market-level, not signal-level,
lookahead (see the module docstring caveat below); the momentum signal
itself is strictly walk-forward, computed only from prices already
observed by t.

Honest limitations, same disclosure standard as the rest of this project:
  - No historical bid/ask exists for a resolved market (`/book` 404s).
    Entry AND exit both use the same last-observed-price proxy this whole
    project already relies on elsewhere -- which means this backtest
    cannot fully rule out the "last-price illusion" failure mode the
    request itself names (a move in the last print with no real
    executable size behind it). The volume confirmation filter is a
    partial mitigation, not a fix.
  - No true historical spread, so no spread filter is applied -- disclosed
    as missing, not faked with an invented number.
  - "Order-flow imbalance" is approximated as trailing dollar volume
    across both outcome tokens (this market's own trade tape), not a
    true buy/sell-aggressor classification, which Polymarket's public
    trade feed does not expose historically.
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import polymarket_final_pct as pmf

REPO = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO / "results" / "polymarket_final_pct"

SAMPLE_SIZE = 400
FIDELITY_MIN = 1                  # 1-minute bars -- affordable given this population's short lifetime
LOOKBACK_5M = 5 * 60
LOOKBACK_30M = 30 * 60
MOM5_THRESHOLD = 0.12              # log-odds; ~ a 3-cent move at p=0.5, per the request's own worked example
MIN_TTR_BUFFER_S = 20 * 60         # don't enter with < 20min left to trade the move
PRICE_LO, PRICE_HI = 0.03, 0.97    # skip near-certain zones -- that's the separately-tested Final-1% regime
HOLDING_PERIODS_S = [15 * 60, 30 * 60, 60 * 60]
TRAILING_STOP_DELTA = 0.03
STAKE = 100.0
VOL_CONFIRM_MULTIPLE = 1.0         # trailing 5min volume must be >= this x the market's own mean 5min volume
FETCH_WORKERS = 16


def _slim_updown_meta(m: dict) -> dict | None:
    q = m.get("question", "")
    if not pmf.re.search(r"\b(bitcoin|btc|ethereum|eth|solana|sol|xrp)\b.*up or down", q, pmf.re.IGNORECASE):
        return None
    cid = m.get("conditionId")
    if not cid:
        return None
    idx = pmf.resolved_outcome_index(m)
    if idx is None:
        return None
    res_ts = pmf._resolution_timestamp(m)
    if res_ts is None:
        return None
    start_ts = pmf._to_epoch_s(m.get("startDate") or m.get("createdAt"))
    if start_ts is None or start_ts < pmf._to_epoch_s(pmf.CLOB_LAUNCH_CUTOFF):
        return None
    token_ids = pmf._safe_json_list(m.get("clobTokenIds"))
    if len(token_ids) != 2:
        return None
    return {
        "conditionId": cid, "question": q, "token_id": token_ids[0],
        "start_s": start_ts, "resolution_time": res_ts.isoformat(),
        "resolved_yes": pmf._safe_json_list(m.get("outcomes"))[idx] == "Yes",
    }


def stream_updown_population(cache_dir: Path = pmf.GAMMA_CACHE_DIR) -> list[dict]:
    leaf_files = sorted(cache_dir.glob("leaf_*.json"))
    print(f"[momentum] streaming {len(leaf_files)} cached census leaf files ...")
    metas: dict[str, dict] = {}
    for i, path in enumerate(leaf_files):
        raw = json.loads(path.read_text())
        for m in raw:
            meta = _slim_updown_meta(m)
            if meta is not None:
                metas[meta["conditionId"]] = meta
        del raw
        if (i + 1) % 100 == 0:
            print(f"  [momentum] {i+1}/{len(leaf_files)} leaves, {len(metas)} up/down markets so far", flush=True)
    return list(metas.values())


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def simulate_market(meta: dict, direction: str, holding_s: int):
    """direction: 'momentum' (bet the move continues) or 'reversal' (bet
    it gives back). Returns a list of closed trade dicts for this one
    market at this one holding period."""
    end_s = int(pmf.pd.Timestamp(meta["resolution_time"]).timestamp())
    start_s = meta["start_s"]
    if end_s <= start_s:
        return []

    prices = pmf.fetch_price_series(meta["token_id"], start_s, end_s, fidelity=FIDELITY_MIN)
    if prices.empty or len(prices) < 40:
        return []
    trades_raw = pmf.fetch_market_trades(meta["conditionId"])
    t_arr = prices["t"].to_numpy(dtype=np.int64)
    p_arr = prices["p"].to_numpy(dtype=np.float64)
    ell = logit(p_arr)

    # trailing-volume lookup: cumulative notional over sorted trade timestamps
    vol_t, vol_notional = [], []
    for tr in (trades_raw or []):
        try:
            vol_t.append(float(tr.get("timestamp", 0)))
            vol_notional.append(float(tr["price"]) * float(tr["size"]))
        except (KeyError, ValueError, TypeError):
            continue
    if vol_t:
        order = np.argsort(vol_t)
        vol_t = np.array(vol_t)[order]
        cum_notional = np.concatenate([[0.0], np.cumsum(np.array(vol_notional)[order])])
        total_span_min = max((end_s - start_s) / 60.0, 1.0)
        mean_5min_vol = cum_notional[-1] / total_span_min * 5.0
    else:
        mean_5min_vol = 0.0
        cum_notional = np.array([0.0])
        vol_t = np.array([])

    def trailing_vol(t):
        if len(vol_t) == 0:
            return 0.0
        hi = np.searchsorted(vol_t, t, side="right")
        lo = np.searchsorted(vol_t, t - LOOKBACK_5M, side="left")
        return cum_notional[hi] - cum_notional[lo]

    category = pmf.classify_report_bucket({"question": meta["question"], "slug": "", "events": []})
    fee = pmf.taker_fee_frac_of_notional

    trades = []
    i = 0
    n = len(t_arr)
    in_position_until_idx = -1
    while i < n:
        t = t_arr[i]
        if in_position_until_idx >= i:
            i += 1
            continue
        ttr = end_s - t
        if ttr < MIN_TTR_BUFFER_S:
            break
        p = p_arr[i]
        if not (PRICE_LO <= p <= PRICE_HI):
            i += 1
            continue

        idx_5m = np.searchsorted(t_arr, t - LOOKBACK_5M, side="left")
        idx_30m = np.searchsorted(t_arr, t - LOOKBACK_30M, side="left")
        if idx_5m >= i or idx_30m >= i or t_arr[idx_30m] < start_s:
            i += 1
            continue
        m5 = ell[i] - ell[idx_5m]
        m30 = ell[i] - ell[idx_30m]

        vconfirm = trailing_vol(t) >= VOL_CONFIRM_MULTIPLE * mean_5min_vol and mean_5min_vol > 0

        long_signal = m5 > MOM5_THRESHOLD and m30 > 0 and vconfirm
        short_signal = m5 < -MOM5_THRESHOLD and m30 < 0 and vconfirm
        if not (long_signal or short_signal):
            i += 1
            continue

        # momentum bets the move continues; reversal bets it gives back
        if direction == "momentum":
            side = "yes" if long_signal else "no"
        else:
            side = "no" if long_signal else "yes"

        entry_price = p if side == "yes" else 1 - p
        entry_fee = fee(entry_price, category)
        entry_t = t

        # walk forward for the exit: fixed time stop, trailing stop, or TTR buffer, whichever first
        j = i + 1
        peak = entry_price
        exit_price, exit_reason = None, None
        while j < n:
            tj = t_arr[j]
            pj = p_arr[j] if side == "yes" else 1 - p_arr[j]
            peak = max(peak, pj)
            if tj - entry_t >= holding_s:
                exit_price, exit_reason = pj, "time_stop"
                break
            if peak - pj >= TRAILING_STOP_DELTA:
                exit_price, exit_reason = pj, "trailing_stop"
                break
            if end_s - tj < MIN_TTR_BUFFER_S:
                exit_price, exit_reason = pj, "ttr_buffer"
                break
            j += 1
        if exit_price is None:
            j = n - 1
            exit_price = p_arr[j] if side == "yes" else 1 - p_arr[j]
            exit_reason = "series_end"

        exit_fee = fee(exit_price, category)
        gross = exit_price - entry_price
        net_frac = gross - entry_price * entry_fee - exit_price * exit_fee
        shares = STAKE / entry_price
        pnl = shares * net_frac

        trades.append({
            "conditionId": meta["conditionId"], "question": meta["question"][:80],
            "direction": direction, "side": side, "entry_time": int(entry_t), "exit_time": int(t_arr[j]),
            "entry_price": round(float(entry_price), 4), "exit_price": round(float(exit_price), 4),
            "exit_reason": exit_reason, "net_return_frac": round(float(net_frac / entry_price), 4),
            "pnl": round(float(pnl), 2),
        })
        in_position_until_idx = j
        i = j + 1

    return trades


def run_variant(sample, direction, holding_s):
    all_trades = []
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        futures = {pool.submit(simulate_market, meta, direction, holding_s): meta for meta in sample}
        for fut in as_completed(futures):
            all_trades.extend(fut.result())

    n = len(all_trades)
    if n == 0:
        return {"n_trades": 0}
    pnls = [t["pnl"] for t in all_trades]
    rets = [t["net_return_frac"] for t in all_trades]
    wins = sum(1 for r in rets if r > 0)
    equity = 10_000.0
    curve = []
    for t in sorted(all_trades, key=lambda x: x["entry_time"]):
        equity += t["pnl"]
        curve.append((t["entry_time"], round(equity, 2)))
    return {
        "n_trades": n,
        "win_rate": round(wins / n, 4),
        "total_pnl": round(sum(pnls), 2),
        "mean_return_frac": round(float(np.mean(rets)), 4),
        "median_return_frac": round(float(np.median(rets)), 4),
        "sharpe_like": round(float(np.mean(rets) / np.std(rets)) * np.sqrt(n), 3) if np.std(rets) > 0 and n > 1 else None,
        "final_equity_flat_stake": round(equity, 2),
        "sample_trades": sorted(all_trades, key=lambda x: -abs(x["pnl"]))[:20],
    }


def main():
    pop_cache = REPO / "data" / "raw" / "polymarket" / "momentum_updown_population.json"
    if pop_cache.exists():
        population = json.loads(pop_cache.read_text())
        print(f"[momentum] loaded cached population: {len(population)} markets")
    else:
        population = stream_updown_population()
        pop_cache.parent.mkdir(parents=True, exist_ok=True)
        pop_cache.write_text(json.dumps(population))
    print(f"[momentum] {len(population)} crypto up/down markets in the census")

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(population), size=min(SAMPLE_SIZE, len(population)), replace=False)
    sample = [population[i] for i in sample_idx]
    print(f"[momentum] sample: {len(sample)} markets")

    results = {}
    t0 = time.time()
    for direction in ["momentum", "reversal"]:
        for holding_s in HOLDING_PERIODS_S:
            key = f"{direction}_{holding_s//60}min"
            print(f"[momentum] running {key} ...", flush=True)
            r = run_variant(sample, direction, holding_s)
            results[key] = r
            print(f"  {key}: n={r['n_trades']}  win_rate={r.get('win_rate')}  "
                  f"total_pnl=${r.get('total_pnl')}  mean_ret={r.get('mean_return_frac')}  "
                  f"({time.time()-t0:.0f}s elapsed)", flush=True)

    out_path = RESULTS_DIR / "momentum_backtest_results.json"
    with open(out_path, "w") as f:
        json.dump({"n_population": len(population), "n_sampled": len(sample), "variants": results}, f, indent=2, default=str)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
