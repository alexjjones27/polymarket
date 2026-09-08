"""Cross-market consistency: does the 15m contract price what its own 5m sub-windows
have already revealed?

Every edge tested so far has been statistical -- a pattern that might be noise. This
is structural. A 15m window [T, T+15] contains exactly the 5m windows [T,T+5],
[T+5,T+10] and [T+10,T+15], and all four contracts settle on the SAME price path. So
by T+10 two of the three sub-moves are public and settled, and they constrain the 15m
outcome in a way that is not a model or a hypothesis but arithmetic:

  both sub-windows UP    -> price is above its start after 10 minutes, so 15m
                            resolves UP unless the final 5 minutes give back the
                            entire gain
  both sub-windows DOWN  -> the mirror
  mixed                  -> genuinely uncertain, and this is the control group

If the 15m market at T+10 prices Up near 0.50 regardless of what the sub-windows did,
it is ignoring settled public information and the mispricing is exploitable with no
forecasting skill at all. If instead its price already tracks the conditional rate,
the market is efficient across contracts and this closes.

Measures P(15m up | sub-window outcomes) against the 15m market price at that moment,
and reports the gap net of the half-spread and the real taker fee.
"""
import json
import statistics as st
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from scipy import stats as sps

REPO = Path(r"D:\Finance\polymarket")
sys.path.insert(0, str(REPO / "src"))
import polymarket_final_pct as pmf

HALF_SPREAD = 0.005
SAMPLE = 500


def fee(p):
    return 0.07 * p * (1 - p)


def load_outcomes(caches):
    """slug-epoch -> (end_epoch, up)."""
    out = {}
    for c in caches:
        p = REPO / "data" / "raw" / "polymarket" / f"{c}.json"
        if not p.exists():
            continue
        for e in json.loads(p.read_text())["events"]:
            try:
                m = e["markets"][0]
                pr = m.get("outcomePrices")
                pr = json.loads(pr) if isinstance(pr, str) else pr
                if not m.get("closed") or len(pr) != 2:
                    continue
                end = int(pmf.pd.Timestamp(e["endDate"]).timestamp())
                out[end] = {"up": float(pr[0]) == 1.0, "event": e}
            except Exception:
                continue
    return out


def fetch_trades(cond):
    res, off = [], 0
    while True:
        try:
            pg = pmf._get(pmf.DATA_API_BASE, "/trades",
                          {"market": cond, "limit": 500, "offset": off})
        except Exception:
            break
        if not pg:
            break
        res.extend(pg)
        if len(pg) < 500:
            break
        off += 500
        if off > 2000:
            break
    return res


def price_series(event):
    try:
        m = event["markets"][0]
        end_s = int(pmf.pd.Timestamp(event["endDate"]).timestamp())
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
    return ser or None


def price_at(ser, off):
    last = None
    for rel, p in ser:
        if rel <= off:
            last = p
        else:
            break
    return last


def main():
    five = load_outcomes(["btc_5m_events", "btc_5m_events_oos"])
    fifteen = load_outcomes(["btc_updown_15m_events", "btc_updown_15m_events_oos"])
    print(f"{len(five)} 5m windows, {len(fifteen)} 15m windows")

    # a 15m window ending at E contains 5m windows ending at E-600, E-300, E
    aligned = []
    for end15, rec in fifteen.items():
        s1, s2, s3 = five.get(end15 - 600), five.get(end15 - 300), five.get(end15)
        if not (s1 and s2 and s3):
            continue
        aligned.append({"end15": end15, "up15": rec["up"], "event": rec["event"],
                        "s1": s1["up"], "s2": s2["up"], "s3": s3["up"]})
    print(f"{len(aligned)} 15m windows fully aligned with their three 5m sub-windows\n")
    if len(aligned) < 100:
        print("insufficient overlap between the caches")
        return

    print("=" * 92)
    print("1. THE ARITHMETIC: does the 15m outcome follow its sub-windows?")
    print("=" * 92)
    print(f"  {'first two 5m':>16} {'n':>6} {'P(15m up)':>11} {'p vs 50%':>10}")
    groups = {}
    for lab, sel in (("both UP", lambda r: r["s1"] and r["s2"]),
                     ("both DOWN", lambda r: not r["s1"] and not r["s2"]),
                     ("mixed", lambda r: r["s1"] != r["s2"])):
        g = [r for r in aligned if sel(r)]
        if len(g) < 30:
            continue
        k = sum(1 for r in g if r["up15"])
        pv = sps.binomtest(k, len(g), 0.5).pvalue
        groups[lab] = (g, k / len(g))
        print(f"  {lab:>16} {len(g):>6} {k/len(g):>10.2%} {pv:>10.2e}")

    print("\n" + "=" * 92)
    print("2. DOES THE 15m MARKET PRICE IT?  (price at T+10, i.e. close-300s)")
    print("=" * 92)
    random_sample = aligned
    if len(aligned) > SAMPLE:
        import random
        random.seed(42)
        random_sample = random.sample(aligned, SAMPLE)
    print(f"  fetching prices for {len(random_sample)} 15m windows ...", flush=True)
    built = {}
    with ThreadPoolExecutor(max_workers=12) as pool:
        futs = {pool.submit(price_series, r["event"]): r["end15"] for r in random_sample}
        done = 0
        for f in as_completed(futs):
            done += 1
            try:
                s = f.result()
            except Exception:
                continue
            if s:
                built[futs[f]] = s
            if done % 150 == 0:
                print(f"    {done}/{len(random_sample)}", flush=True)

    rows = []
    for r in random_sample:
        ser = built.get(r["end15"])
        if not ser:
            continue
        px = price_at(ser, -300)          # T+10 into a 15m window
        if px is None or not (0.02 < px < 0.98):
            continue
        rows.append({**r, "px": px})
    print(f"  {len(rows)} with a usable price at T+10\n")
    if len(rows) < 60:
        print("  too few priced windows")
        return

    print(f"  {'first two 5m':>16} {'n':>5} {'mkt price':>10} {'actual':>9} {'gap':>8} "
          f"{'p':>9} {'net edge/share':>16}")
    for lab, sel in (("both UP", lambda r: r["s1"] and r["s2"]),
                     ("both DOWN", lambda r: not r["s1"] and not r["s2"]),
                     ("mixed", lambda r: r["s1"] != r["s2"])):
        g = [r for r in rows if sel(r)]
        if len(g) < 25:
            continue
        mp = st.mean(r["px"] for r in g)
        k = sum(1 for r in g if r["up15"])
        act = k / len(g)
        pv = sps.binomtest(k, len(g), min(0.999, max(0.001, mp))).pvalue
        # tradeable side: buy Up if underpriced, else buy Down
        if act > mp:
            entry, win, side = mp + HALF_SPREAD, act, "Up"
        else:
            entry, win, side = (1 - mp) + HALF_SPREAD, 1 - act, "Down"
        net = win - entry - fee(entry)
        flag = f"  <== buy {side}" if (pv < 0.05 and net > 0) else (
            f"  (buy {side}, ns)" if net > 0 else "")
        print(f"  {lab:>16} {len(g):>5} {mp:>10.4f} {act:>9.4f} {act-mp:>+8.4f} "
              f"{pv:>9.4f} {net:>+16.4f}{flag}")


if __name__ == "__main__":
    main()
