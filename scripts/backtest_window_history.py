"""Tests whether a window's EARLY history predicts whether a late entry survives.

Hypothesis (user's, and the third stop-out is a textbook case): the bot only watches
the final 90 seconds, so it cannot tell the difference between

  a decided window  -- one side has been dominant for minutes, and the late high
                       price reflects genuine conviction
  a contested window -- the price has churned around 0.5 all window, and the late
                       spike to 0.96+ is noise, because with BTC sitting on the
                       strike the tiniest move flips it

Window 1788789600 was the second kind: over T-300s..T-90s it crossed 0.50 six
times, spent only 41% of the time above 0.70, and ranged 0.29-0.94. Then it printed
0.99 at T-60s, we bought, and it settled at 0.001.

Nothing in the current rule can see any of that, because prepare_event and the live
bot both start at close-90.

Method: fetch the FULL window of prints, compute path features over T-300..T-90
only (strictly before any entry, so no lookahead), then simulate the entry rule and
test which features separate good outcomes from bad.

Two targets, because the print data badly under-counts real losses (2 in 584):
  stopped  -- price later dips below 0.70, i.e. our stop-loss would have fired.
              This is ~10x more frequent than an outright loss, so it is far better
              powered, and stop-outs are what is actually costing us money now.
  lost     -- the window genuinely resolved against us.
"""
import json
import random
import statistics as st
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scipy import stats as sps

REPO = Path(r"D:\Finance\polymarket")
sys.path.insert(0, str(REPO / "src"))
import polymarket_final_pct as pmf

THRESHOLD = 0.94
SUSTAIN_S = 5
STOP_LEVEL = 0.70
HIST_START = -300      # window open
HIST_END = -90         # where live monitoring begins
SAMPLE_N = 250


def fetch_all_trades(cond_id):
    """Full-window fetches are heavy (1,000+ prints per window), so a single slow
    response must not abort the whole run -- return what we have and move on."""
    out, offset = [], 0
    while True:
        try:
            page = pmf._get(pmf.DATA_API_BASE, "/trades",
                            {"market": cond_id, "limit": 500, "offset": offset})
        except Exception:
            break
        if not page:
            break
        out.extend(page)
        if len(page) < 500:
            break
        offset += 500
        if offset > 3000:
            break
    return out


def build(event):
    try:
        m = event["markets"][0]
        prices = pmf._safe_json_list(m.get("outcomePrices"))
        if not m.get("closed") or len(prices) != 2:
            return None
        up_won = float(prices[0]) == 1.0
        end_s = int(pmf.pd.Timestamp(event["endDate"]).timestamp())
        cond = m.get("conditionId")
        if not cond:
            return None
    except Exception:
        return None
    trades = fetch_all_trades(cond)
    if not trades:
        return None
    by_side = {"Up": [], "Down": []}
    for t in sorted(trades, key=lambda x: x["timestamp"]):
        try:
            p = float(t["price"])
        except (TypeError, ValueError):
            continue
        rel = t["timestamp"] - end_s
        if t.get("outcome") == "Up":
            by_side["Up"].append((rel, p))
            by_side["Down"].append((rel, 1.0 - p))
        elif t.get("outcome") == "Down":
            by_side["Down"].append((rel, p))
            by_side["Up"].append((rel, 1.0 - p))
    return {"by_side": by_side, "up_won": up_won}


def features(series_hist):
    """Path features over T-300..T-90 for one side. No lookahead."""
    ps = [p for _, p in series_hist]
    if len(ps) < 10:
        return None
    crossings = sum(1 for i in range(1, len(ps)) if (ps[i-1] - 0.5) * (ps[i] - 0.5) < 0)
    # how long since it last sat on the fence
    last_cross_rel = None
    for i in range(len(ps) - 1, 0, -1):
        if (ps[i-1] - 0.5) * (ps[i] - 0.5) < 0:
            last_cross_rel = series_hist[i][0]
            break
    return {
        "frac_above_70": sum(1 for p in ps if p > 0.70) / len(ps),
        "frac_above_85": sum(1 for p in ps if p > 0.85) / len(ps),
        "mean": st.mean(ps),
        "min": min(ps),
        "crossings_50": crossings,
        "secs_since_cross": (HIST_END - last_cross_rel) if last_cross_rel is not None else 210.0,
        "vol": st.pstdev(ps) if len(ps) > 1 else 0.0,
    }


def evaluate(prep):
    """Simulate entry, return features + outcomes."""
    for side, series in prep["by_side"].items():
        hist = [(r, p) for r, p in series if HIST_START <= r < HIST_END]
        live = [(r, p) for r, p in series if r >= HIST_END]
        if not hist or not live:
            continue
        # sustained crossing of THRESHOLD in the live portion
        entry = None
        i = 0
        while i < len(live):
            r, p = live[i]
            if p >= THRESHOLD:
                ok, j = True, i + 1
                last = (r, p)
                while j < len(live) and live[j][0] <= r + SUSTAIN_S:
                    if live[j][1] < THRESHOLD:
                        ok = False
                        break
                    last = live[j]
                    j += 1
                if ok:
                    entry = last
                    break
                i = j + 1
                continue
            i += 1
        if entry is None:
            continue
        f = features(hist)
        if f is None:
            continue
        after = [p for r, p in live if r > entry[0]]
        won = prep["up_won"] if side == "Up" else (not prep["up_won"])
        return {**f, "side": side, "entry_price": entry[1], "won": won,
                "stopped": bool(after and min(after) < STOP_LEVEL)}
    return None


def prepare(cache):
    events = json.loads((REPO / "data" / "raw" / "polymarket" / cache).read_text())["events"]
    random.seed(42)
    sample = random.sample(events, min(SAMPLE_N, len(events)))
    print(f"  fetching full-window trades for {len(sample)} events ...", flush=True)
    rows = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(build, e): e for e in sample}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                r = fut.result()
            except Exception:
                continue
            if r:
                ev = evaluate(r)
                if ev:
                    rows.append(ev)
            if done % 50 == 0:
                print(f"    {done}/{len(futs)} fetched, {len(rows)} usable", flush=True)
    return rows


FEATS = ["frac_above_70", "frac_above_85", "mean", "min", "crossings_50",
         "secs_since_cross", "vol"]


def report(rows, label):
    print(f"\n=== {label} (n={len(rows)}) ===")
    bad = [r for r in rows if r["stopped"]]
    good = [r for r in rows if not r["stopped"]]
    print(f"  stopped: {len(bad)} ({len(bad)/len(rows):.1%})   clean: {len(good)}")
    if not bad or not good:
        return
    print(f"\n  {'feature':>18} {'clean':>9} {'stopped':>9} {'diff':>9} {'p':>8}")
    for f in FEATS:
        a = [r[f] for r in good]
        b = [r[f] for r in bad]
        try:
            _, p = sps.mannwhitneyu(a, b, alternative="two-sided")
        except Exception:
            p = float("nan")
        flag = "  <--" if p < 0.05 else ""
        print(f"  {f:>18} {st.mean(a):>9.3f} {st.mean(b):>9.3f} "
              f"{st.mean(a)-st.mean(b):>+9.3f} {p:>8.4f}{flag}")


def filter_test(rows, label):
    print(f"\n=== FILTER TEST -- {label} ===")
    print(f"  {'rule':>34} {'kept':>6} {'kept%':>7} {'stop_rate':>10} {'win_rate':>9}")
    base_stop = sum(1 for r in rows if r["stopped"]) / len(rows)
    base_win = sum(1 for r in rows if r["won"]) / len(rows)
    print(f"  {'(no filter)':>34} {len(rows):>6} {100:>6.0f}% {base_stop:>9.1%} {base_win:>8.2%}")
    rules = [
        ("frac_above_70 >= 0.60", lambda r: r["frac_above_70"] >= 0.60),
        ("frac_above_70 >= 0.75", lambda r: r["frac_above_70"] >= 0.75),
        ("frac_above_70 >= 0.90", lambda r: r["frac_above_70"] >= 0.90),
        ("crossings_50 == 0", lambda r: r["crossings_50"] == 0),
        ("crossings_50 <= 1", lambda r: r["crossings_50"] <= 1),
        ("min >= 0.40", lambda r: r["min"] >= 0.40),
        ("min >= 0.50", lambda r: r["min"] >= 0.50),
        ("secs_since_cross >= 120", lambda r: r["secs_since_cross"] >= 120),
        ("frac_above_70>=.75 & cross<=1", lambda r: r["frac_above_70"] >= 0.75 and r["crossings_50"] <= 1),
    ]
    for name, fn in rules:
        kept = [r for r in rows if fn(r)]
        if not kept:
            continue
        s = sum(1 for r in kept if r["stopped"]) / len(kept)
        w = sum(1 for r in kept if r["won"]) / len(kept)
        print(f"  {name:>34} {len(kept):>6} {len(kept)/len(rows):>6.0%} {s:>9.1%} {w:>8.2%}")


def main():
    print("IN-SAMPLE")
    is_rows = prepare("btc_5m_events.json")
    print("OOS")
    oos_rows = prepare("btc_5m_events_oos.json")
    for rows, lab in [(is_rows, "IN-SAMPLE"), (oos_rows, "OOS")]:
        report(rows, lab)
        filter_test(rows, lab)


if __name__ == "__main__":
    main()
