"""Serial reversion and its pricing, across every timeframe.

The 4h series shows real reversion -- after an up window the next is up 44.13%,
after a down window 55.12%, Fisher p=0.0036, replicating in both halves. And the
market appears not to price the first half of it: after an UP window it quotes "Up"
near 0.494 while the realised rate is 0.4435, a ~5pp gap consistent across five
different sampling offsets.

Testing every timeframe serves two purposes at once. It adds power, and more
importantly it is a NEGATIVE CONTROL: the 5m and 15m series showed no serial
dependence at all (p=0.095/0.244 and p=0.712/0.874). If a pricing gap of the same
shape shows up there anyway, the gap is an artifact of my method rather than a real
mispricing, and the 4h result should be discarded with it.

For each timeframe this reports:
  serial dependence  -- does the prior window's outcome predict this one
  pricing gap        -- realised rate minus the market's early quote, by prior
  net edge           -- that gap after the half-spread and the real taker fee

A gap only matters where serial dependence exists. A gap where it does not is noise
in the measurement.
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

HALF_SPREAD = 0.005
SAMPLE = 420

# timeframe: (caches, window length, early-sample offsets relative to close)
FRAMES = {
    "5m": (["btc_5m_events", "btc_5m_events_oos"], 300, [-280, -240, -180]),
    "15m": (["btc_updown_15m_events", "btc_updown_15m_events_oos"], 900, [-840, -720, -540]),
    "4h": (["btc_updown_4h_events", "btc_updown_4h_events_oos"], 14400,
           [-14000, -13500, -12600]),
}


def fee(p):
    return 0.07 * p * (1 - p)


def fetch_trades(cond):
    out, off = [], 0
    while True:
        try:
            pg = pmf._get(pmf.DATA_API_BASE, "/trades",
                          {"market": cond, "limit": 500, "offset": off})
        except Exception:
            break
        if not pg:
            break
        out.extend(pg)
        if len(pg) < 500:
            break
        off += 500
        if off > 2000:
            break
    return out


def build(e):
    try:
        m = e["markets"][0]
        pr = m.get("outcomePrices")
        pr = json.loads(pr) if isinstance(pr, str) else pr
        if not m.get("closed") or len(pr) != 2:
            return None
        end_s = int(pmf.pd.Timestamp(e["endDate"]).timestamp())
        cond = m.get("conditionId")
        if not cond:
            return None
    except Exception:
        return None
    ser = []
    for t in sorted(fetch_trades(cond) or [], key=lambda x: x["timestamp"]):
        try:
            p = float(t["price"])
        except (TypeError, ValueError):
            continue
        rel = t["timestamp"] - end_s
        if t.get("outcome") == "Up":
            ser.append((rel, p))
        elif t.get("outcome") == "Down":
            ser.append((rel, 1.0 - p))
    return {"end_s": end_s, "up": float(pr[0]) == 1.0, "ser": ser}


def price_at(ser, off):
    last = None
    for rel, p in ser:
        if rel <= off:
            last = p
        else:
            break
    return last


def load_events(caches):
    seen = {}
    for c in caches:
        p = REPO / "data" / "raw" / "polymarket" / f"{c}.json"
        if not p.exists():
            continue
        for e in json.loads(p.read_text())["events"]:
            try:
                seen[e["slug"]] = e
            except Exception:
                continue
    return sorted(seen.values(), key=lambda e: e.get("endDate", ""))


def main():
    for tf, (caches, wlen, offsets) in FRAMES.items():
        evs = load_events(caches)
        if len(evs) < 200:
            print(f"\n{tf}: only {len(evs)} windows, skipping")
            continue
        print(f"\n{'='*94}\n{tf.upper()}   {len(evs)} windows\n{'='*94}")

        # --- serial dependence, on the FULL set (free) ---
        outs = []
        for e in evs:
            try:
                m = e["markets"][0]
                pr = m.get("outcomePrices")
                pr = json.loads(pr) if isinstance(pr, str) else pr
                if m.get("closed") and len(pr) == 2:
                    outs.append(float(pr[0]) == 1.0)
            except Exception:
                continue
        a = b = c = d = 0
        for i in range(1, len(outs)):
            if outs[i-1]:
                a, b = (a + 1, b) if outs[i] else (a, b + 1)
            else:
                c, d = (c + 1, d) if outs[i] else (c, d + 1)
        _, sp = sps.fisher_exact([[a, b], [c, d]])
        up_after_up = a / (a + b) if a + b else float("nan")
        up_after_dn = c / (c + d) if c + d else float("nan")
        verdict = "REVERSION" if sp < 0.05 else "none"
        print(f"  serial dependence: after-up {up_after_up:.2%}, after-down "
              f"{up_after_dn:.2%}, spread {up_after_dn-up_after_up:+.2%}, "
              f"p={sp:.4f}  -> {verdict}")

        # --- pricing gap (costs API calls) ---
        random.seed(42)
        idx = sorted(random.sample(range(1, len(evs)), min(SAMPLE, len(evs) - 1)))
        todo = sorted(set(idx) | {i - 1 for i in idx})
        print(f"  fetching {len(todo)} windows for pricing test ...", flush=True)
        built = {}
        with ThreadPoolExecutor(max_workers=12) as pool:
            futs = {pool.submit(build, evs[i]): i for i in todo}
            for f in as_completed(futs):
                try:
                    r = f.result()
                except Exception:
                    continue
                if r:
                    built[futs[f]] = r

        off = offsets[0]
        rows = []
        for i in idx:
            cur, prev = built.get(i), built.get(i - 1)
            if not cur or not prev:
                continue
            px = price_at(cur["ser"], off)
            if px is None or not (0.02 < px < 0.98):
                continue
            rows.append({"prev_up": prev["up"], "px": px, "up": cur["up"]})
        if len(rows) < 60:
            print("  too few priced windows")
            continue
        print(f"\n  pricing at close{off}s   n={len(rows)}")
        print(f"  {'prior':>12} {'n':>5} {'mkt price(Up)':>15} {'actual':>9} "
              f"{'gap':>9} {'p':>8} {'net edge/share':>16}")
        for lab, sel, buy_side in (("after UP", True, "Down"), ("after DOWN", False, "Up")):
            g = [r for r in rows if r["prev_up"] is sel]
            if len(g) < 25:
                continue
            mp = st.mean(r["px"] for r in g)
            k = sum(1 for r in g if r["up"])
            act = k / len(g)
            pv = sps.binomtest(k, len(g), min(0.999, max(0.001, mp))).pvalue
            # the tradeable side: after UP we would buy Down, after DOWN we buy Up
            if buy_side == "Down":
                entry = (1.0 - mp) + HALF_SPREAD
                win = 1.0 - act
            else:
                entry = mp + HALF_SPREAD
                win = act
            net = win - entry - fee(entry)
            flag = ""
            if pv < 0.05 and net > 0:
                flag = f"  <== buy {buy_side}"
            elif net > 0:
                flag = f"  (buy {buy_side}, ns)"
            print(f"  {lab:>12} {len(g):>5} {mp:>15.4f} {act:>9.4f} {act-mp:>+9.4f} "
                  f"{pv:>8.4f} {net:>+16.4f}{flag}")


if __name__ == "__main__":
    main()
