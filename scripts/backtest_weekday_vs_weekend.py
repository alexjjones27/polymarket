"""Tests whether this edge is more dangerous at weekends, when traditional markets
are shut and BTC liquidity is thinner.

This matters because the entire live sample is a single day: every one of the 129
live trades has weekday == 6 (Sunday). There is no live weekday data at all, so
the question can only be answered historically.

Methodological problem: the historical backtest wins ~99.7% of the time (2 losses
in 584), so comparing WIN RATES across subgroups has essentially no statistical
power -- any split would put ~1 loss on each side. So the primary metrics here are
continuous or high-frequency measures of post-entry instability, which is the thing
that actually causes losses and occurs often enough to measure:

  dip_rate_070   -- fraction of positions whose price dips below 0.70 after entry
                    (the stop-loss level; happens ~3-4% of the time, ~10x more
                    often than an outright loss, so ~10x the power)
  mean_min_after -- average minimum price reached after entry (fully continuous)
  p10_min_after  -- 10th percentile of that minimum, i.e. the bad tail

If weekends are genuinely riskier, positions entered then should wander lower after
entry, and these metrics will show it long before win rate could.
"""
import json
import random
import statistics as st
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from scipy import stats as sps

REPO = Path(r"D:\Finance\polymarket")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
import backtest_btc_5m_sustained as sust

THRESHOLD = 0.94
SUSTAIN_S = 5
DIP_LEVEL = 0.70
PER_GROUP = 500


def load_events():
    out = []
    for cache in ("btc_5m_events.json", "btc_5m_events_oos.json"):
        path = REPO / "data" / "raw" / "polymarket" / cache
        out.extend(json.loads(path.read_text())["events"])
    return out


def weekday_of(event):
    try:
        return datetime.fromisoformat(
            event["endDate"].replace("Z", "+00:00")).astimezone(timezone.utc).weekday()
    except Exception:
        return None


def analyse(prepared):
    """Per-entry post-entry instability metrics."""
    rows = []
    for p in prepared:
        r = sust.find_sustained_crossing(p, THRESHOLD, SUSTAIN_S)
        if not r:
            continue
        entry_ts = p["end_s"] + r["secs_after_close"]
        series = p["by_side"][r["side"]]
        after = [pr for ts, pr, _ in series if ts > entry_ts]
        if not after:
            continue
        rows.append({"won": r["won"], "min_after": min(after),
                     "dipped": min(after) < DIP_LEVEL, "price": r["price"]})
    return rows


def prepare_events(events, label):
    print(f"{label}: fetching trades for {len(events)} events ...", flush=True)
    prepared = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futs = {pool.submit(sust.prepare_event, e): e for e in events}
        for fut in as_completed(futs):
            r = fut.result()
            if r:
                prepared.append(r)
    return prepared


def summarize(name, rows):
    n = len(rows)
    if not n:
        print(f"  {name}: no data")
        return None
    wins = sum(1 for r in rows if r["won"])
    dips = sum(1 for r in rows if r["dipped"])
    mins = sorted(r["min_after"] for r in rows)
    p10 = mins[max(0, int(0.10 * len(mins)) - 1)]
    print(f"  {name:10s} n={n:4d}  win_rate={wins/n:7.3%}  dip_rate_070={dips/n:6.2%}  "
          f"mean_min_after={st.mean(mins):.4f}  p10_min_after={p10:.4f}")
    return {"n": n, "wins": wins, "dips": dips, "mins": mins}


def main():
    events = load_events()
    tagged = [(e, weekday_of(e)) for e in events]
    weekend = [e for e, wd in tagged if wd in (5, 6)]
    weekday = [e for e, wd in tagged if wd is not None and wd < 5]
    print(f"cached universe: {len(weekend)} weekend windows, {len(weekday)} weekday windows")

    random.seed(42)
    we_sample = random.sample(weekend, min(PER_GROUP, len(weekend)))
    wd_sample = random.sample(weekday, min(PER_GROUP, len(weekday)))

    we_rows = analyse(prepare_events(we_sample, "weekend"))
    wd_rows = analyse(prepare_events(wd_sample, "weekday"))

    print("\n=== WEEKEND vs WEEKDAY (post-entry instability) ===")
    we = summarize("weekend", we_rows)
    wd = summarize("weekday", wd_rows)

    if we and wd:
        # dip rate: 2x2 Fisher
        table = [[we["dips"], we["n"] - we["dips"]], [wd["dips"], wd["n"] - wd["dips"]]]
        _, p_dip = sps.fisher_exact(table)
        # min-after distribution: Mann-Whitney (continuous, best powered)
        _, p_min = sps.mannwhitneyu(we["mins"], wd["mins"], alternative="two-sided")
        print(f"\n  dip_rate_070   weekend {we['dips']}/{we['n']} vs weekday "
              f"{wd['dips']}/{wd['n']}   Fisher p={p_dip:.4f}")
        print(f"  min_after dist                                        "
              f"   Mann-Whitney p={p_min:.4f}")

    print("\n=== BY DAY OF WEEK ===")
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    by_day = {}
    for wd_i in range(7):
        day_events = [e for e, w in tagged if w == wd_i]
        if not day_events:
            continue
        samp = random.sample(day_events, min(160, len(day_events)))
        by_day[wd_i] = analyse(prepare_events(samp, names[wd_i]))
    print()
    for wd_i, rows in sorted(by_day.items()):
        summarize(names[wd_i], rows)


if __name__ == "__main__":
    main()
