"""Estimates P&L at larger position sizes using real collected order-book depth,
rather than assuming it scales linearly.

Naive scaling says $50 stakes earn 10x what $5 stakes earn. That is wrong in two
places, and both are measurable from the book snapshots now being collected:

  ENTRY  we take liquidity off the ask ladder. 5 shares usually clears at the best
         level; 52 shares may walk several levels up, so the average fill price
         rises with size and the edge per share shrinks.

  EXIT   the stop-loss sells into a book that has just lost most of its depth --
         measured at 726-1,280 shares remaining during the two real stop-outs. A
         larger position eats further down the bid ladder at exactly the worst
         moment, so the realised loss per share also worsens with size.

Both effects push the same way: bigger size earns less per dollar and loses more
per dollar. This walks the actual ladders recorded at each entry and each stop to
quantify how much.
"""
import csv
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

REPO = Path(r"D:\Finance\polymarket")
SIZES = [5, 8, 10, 20, 50]
MIN_SHARES = 5.0
FEE_RATE = 0.07
FIX = datetime(2026, 9, 7, 9, 55, 55, tzinfo=timezone(timedelta(hours=1)))


def tt(r):
    return datetime.strptime(r["trade_time"], "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=timezone(timedelta(hours=1))).timestamp()


def walk(levels, target, ascending=True):
    """Weighted-average price to fill `target` shares. Returns (filled, avg, levels_used)."""
    rem, cost, filled, used = target, 0.0, 0.0, 0
    for lv in levels:
        try:
            px, sz = float(lv[0]), float(lv[1])
        except (TypeError, ValueError, IndexError):
            continue
        take = min(rem, sz)
        if take <= 0:
            continue
        cost += take * px
        filled += take
        rem -= take
        used += 1
        if rem <= 1e-9:
            break
    if rem > 1e-9:
        return filled, (cost / filled if filled else None), used
    return filled, cost / filled, used


def load_books():
    books = {}
    path = REPO / "data" / "raw" / "polymarket" / "book_snapshots"
    for f in sorted(path.glob("*.jsonl")):
        for line in f.open():
            try:
                r = json.loads(line)
            except Exception:
                continue
            books.setdefault((r["window_end"], r["side"]), []).append(r)
    for k in books:
        books[k].sort(key=lambda r: r["ts"])
    return books


def snapshot_at(books, window, side, ts):
    rows = books.get((window, side))
    if not rows:
        return None
    best = None
    for r in rows:
        if r["ts"] <= ts + 1.0:
            best = r
        else:
            break
    return best


def main():
    books = load_books()
    print(f"loaded book snapshots for {len(books)} (window, side) pairs")

    trades = [r for r in csv.DictReader(open(REPO / "results" / "btc_5m_live" / "trade_log.csv"))
              if r.get("resolved_won") in ("True", "False") and tt(r) >= FIX.timestamp()]

    matched = []
    for r in trades:
        snap = snapshot_at(books, int(r["window_end"]), r["side"], tt(r))
        if snap and snap.get("asks"):
            matched.append((r, snap))
    print(f"matched {len(matched)} of {len(trades)} post-fix trades to a book snapshot\n")
    if not matched:
        print("no matches yet -- collector needs more runtime")
        return

    print("=== ENTRY FILL vs SIZE (walking the real ask ladder) ===")
    print(f"{'stake':>6} {'n_fillable':>11} {'avg_fill_px':>12} {'vs $5':>8} {'avg_levels':>11}")
    base_px = None
    entry_stats = {}
    for stake in SIZES:
        fills, lv_used, unfillable = [], [], 0
        for r, snap in matched:
            best_ask = float(snap["asks"][0][0])
            target = max(MIN_SHARES, stake / best_ask)
            filled, avg, used = walk(snap["asks"], target)
            if avg is None or filled < target - 1e-6:
                unfillable += 1
                continue
            fills.append(avg)
            lv_used.append(used)
        if not fills:
            continue
        avg_px = sum(fills) / len(fills)
        if base_px is None:
            base_px = avg_px
        entry_stats[stake] = (avg_px, len(fills), unfillable)
        print(f"${stake:>5} {len(fills):>11} {avg_px:>12.4f} {avg_px-base_px:>+8.4f} "
              f"{sum(lv_used)/len(lv_used):>11.2f}"
              + (f"   ({unfillable} unfillable)" if unfillable else ""))

    # exit side: only the real stop-outs
    stops = [(r, s) for r, s in matched if r.get("exited") == "True"]
    print(f"\n=== STOP-LOSS EXIT vs SIZE (walking the real bid ladder at the stop) ===")
    if not stops:
        print("  no stop-outs matched to snapshots")
    exit_stats = {}
    for stake in SIZES:
        outs = []
        for r, _ in stops:
            window, side = int(r["window_end"]), r["side"]
            rows = books.get((window, side), [])
            # book at the moment the stop fired
            et = datetime.strptime(r["exit_time"], "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone(timedelta(hours=1))).timestamp()
            snap = None
            for x in rows:
                if x["ts"] <= et + 1.0:
                    snap = x
            if not snap or not snap.get("bids"):
                continue
            entry_ask = float(r["ask_price"])
            target = max(MIN_SHARES, stake / entry_ask)
            filled, avg, _ = walk(snap["bids"], target)
            if avg is None:
                # not enough depth to exit the whole position
                outs.append(("partial", filled))
            else:
                outs.append(("full", avg))
        full = [v for k, v in outs if k == "full"]
        part = [v for k, v in outs if k == "partial"]
        if full:
            exit_stats[stake] = sum(full) / len(full)
            print(f"  ${stake:>3}: avg exit price {sum(full)/len(full):.4f} "
                  f"({len(full)} full exits" + (f", {len(part)} COULD NOT FULLY EXIT)" if part else ")"))
        elif part:
            print(f"  ${stake:>3}: NO full exit possible in {len(part)} stop(s) -- "
                  f"insufficient bid depth")

    print("\n=== ESTIMATED P&L BY SIZE ===")
    print("  naive = linear scaling; adjusted = using measured fill degradation")
    obs_hr, obs_stake = 0.635, 5.0
    print(f"{'stake':>6} {'naive $/hr':>12} {'adjusted $/hr':>15} {'naive $/day':>13} "
          f"{'adjusted $/day':>16}")
    for stake in SIZES:
        naive = obs_hr * stake / obs_stake
        adj = naive
        if stake in entry_stats and base_px:
            px_now = entry_stats[stake][0]
            # edge per share shrinks by the extra price paid
            edge_base = 0.0134  # measured calibration edge/share in the 0.95+ band
            degrade = (px_now - base_px)
            adj = naive * max(0.0, (edge_base - degrade)) / edge_base
        print(f"${stake:>5} {naive:>+12.2f} {adj:>+15.2f} {naive*24:>+13.2f} {adj*24:>+16.2f}")

    print("\n  NOTE: adjusted figures assume the stop-out rate stays at the observed")
    print("  4.3%. Break-even is 7.61%, and the 95% CI still spans it, so every")
    print("  number here inherits that uncertainty -- scaled up proportionally.")


if __name__ == "__main__":
    main()
