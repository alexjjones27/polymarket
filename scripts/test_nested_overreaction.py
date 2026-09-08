"""Does the 15m market OVERPRICE the continuation after two aligned sub-windows?

The nested test showed the 15m contract broadly prices its own 5m sub-windows
correctly -- which is the efficient outcome and closed most of that idea. One cell
did not close. After both of the first two 5m sub-windows resolved UP, the 15m market
quoted Up at 0.8445 while the realised rate was 0.8025: a 4.2pp overpricing of the
continuation, worth +0.0276/share to fade by buying Down.

At n=81 that is about one standard error, so it proves nothing. It is also
economically plausible in a way most of the noise found in this project was not:
overreaction to a salient recent run is one of the most repeatedly documented biases
in prediction markets, and the direction here (fade the continuation) is the one that
bias predicts.

This prices ONLY the both-UP and both-DOWN cells, so the API budget goes where the
power is needed rather than being spent on the uninformative mixed cell. Target is
~350 per cell, which brings the standard error to ~2.1pp and makes a 4pp effect
detectable.

Split chronologically, because a favourable second half has flattered three
candidates in this project already.
"""
import json
import statistics as st
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scipy import stats as sps

REPO = Path(r"D:\Finance\polymarket")
sys.path.insert(0, str(REPO / "src"))
import polymarket_final_pct as pmf

HALF_SPREAD = 0.005
OFFSETS = [-300, -240, -180]     # T+10, T+11, T+12 into a 15m window


def fee(p):
    return 0.07 * p * (1 - p)


def load_outcomes(caches):
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


def series_for(event):
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

    aligned = []
    for end15, rec in fifteen.items():
        s1, s2 = five.get(end15 - 600), five.get(end15 - 300)
        if not (s1 and s2):
            continue
        if s1["up"] != s2["up"]:
            continue                      # only the informative cells
        aligned.append({"end15": end15, "up15": rec["up"], "event": rec["event"],
                        "both_up": s1["up"]})
    aligned.sort(key=lambda r: r["end15"])
    n_up = sum(1 for r in aligned if r["both_up"])
    print(f"{len(aligned)} aligned windows in the informative cells "
          f"({n_up} both-UP, {len(aligned)-n_up} both-DOWN)")
    print(f"fetching prices for all of them ...\n", flush=True)

    built = {}
    with ThreadPoolExecutor(max_workers=12) as pool:
        futs = {pool.submit(series_for, r["event"]): r["end15"] for r in aligned}
        done = 0
        for f in as_completed(futs):
            done += 1
            try:
                s = f.result()
            except Exception:
                continue
            if s:
                built[futs[f]] = s
            if done % 250 == 0:
                print(f"  {done}/{len(aligned)}", flush=True)

    for off in OFFSETS:
        rows = []
        for r in aligned:
            ser = built.get(r["end15"])
            if not ser:
                continue
            px = price_at(ser, off)
            if px is None or not (0.02 < px < 0.98):
                continue
            rows.append({**r, "px": px})
        if len(rows) < 100:
            continue
        print("=" * 100)
        print(f"PRICE AT close{off}s   ({len(rows)} windows)")
        print("=" * 100)
        print(f"  {'cell':>11} {'sample':>13} {'n':>5} {'mkt price':>10} {'actual':>9} "
              f"{'gap':>8} {'p':>8} {'fade edge/sh':>14}")
        for cell, sel in (("both UP", True), ("both DOWN", False)):
            g = [r for r in rows if r["both_up"] is sel]
            if len(g) < 60:
                continue
            half = len(g) // 2
            for lab, seg in (("first half", g[:half]), ("second half", g[half:]),
                             ("FULL", g)):
                mp = st.mean(r["px"] for r in seg)
                k = sum(1 for r in seg if r["up15"])
                act = k / len(seg)
                pv = sps.binomtest(k, len(seg), min(0.999, max(0.001, mp))).pvalue
                # fade the continuation: after both-UP buy Down, after both-DOWN buy Up
                if sel:
                    entry, win = (1 - mp) + HALF_SPREAD, 1 - act
                else:
                    entry, win = mp + HALF_SPREAD, act
                net = win - entry - fee(entry)
                flag = "  <==" if (pv < 0.05 and net > 0) else ""
                print(f"  {cell:>11} {lab:>13} {len(seg):>5} {mp:>10.4f} {act:>9.4f} "
                      f"{act-mp:>+8.4f} {pv:>8.4f} {net:>+14.4f}{flag}")
            print()


if __name__ == "__main__":
    main()
