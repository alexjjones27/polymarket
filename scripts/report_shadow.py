"""Reports shadow-trading results per config, alongside the real live results.

The comparison that matters is mid0.70 vs mid0.94 -- both paper, same book, same
period, so the only difference is the threshold. mid0.94 also acts as a sanity
check: it mirrors the live rule, so if its shadow P&L diverges badly from real
live P&L over the same window, the shadow model itself is suspect.
"""
import csv
import json
from pathlib import Path

STATE_DIR = Path(__file__).resolve().parents[1] / "results" / "btc_5m_live"
SHADOW_LOG = STATE_DIR / "shadow_trades.csv"
TRADE_LOG = STATE_DIR / "trade_log.csv"
SHADOW_STATE = STATE_DIR / "shadow_state.json"


def main():
    if not SHADOW_LOG.exists():
        print("no shadow trades settled yet")
        if SHADOW_STATE.exists():
            st = json.loads(SHADOW_STATE.read_text())
            print(f"  open shadow positions: {len(st.get('pending', []))}")
            for p in st.get("pending", []):
                print(f"    [{p['config']}] window {p['window_end']} {p['side']} "
                      f"{p['shares']}@{p['avg_price']} cost ${p['cost_usd']}")
        return

    rows = list(csv.DictReader(open(SHADOW_LOG)))
    by_cfg = {}
    for r in rows:
        by_cfg.setdefault(r["config"], []).append(r)

    print(f"=== SHADOW RESULTS ({len(rows)} settled paper trades) ===")
    print(f"{'config':>14} {'n':>4} {'W':>4} {'L':>3} {'win_rate':>9} {'avg_px':>7} "
          f"{'avg_slip':>9} {'invested':>9} {'pnl':>9} {'return':>8}")
    for cfg in sorted(by_cfg):
        g = by_cfg[cfg]
        n = len(g)
        w = sum(1 for r in g if r["won"] == "True")
        pnl = sum(float(r["pnl_usd"]) for r in g)
        inv = sum(float(r["cost_usd"]) for r in g)
        avg_px = sum(float(r["avg_price"]) for r in g) / n
        slip = sum(float(r["slippage"] or 0) for r in g) / n
        print(f"{cfg:>14} {n:>4} {w:>4} {n-w:>3} {w/n:>8.1%} {avg_px:>7.4f} "
              f"{slip:>+9.4f} {inv:>9.2f} {pnl:>+9.3f} {pnl/inv:>+7.2%}")

    if SHADOW_STATE.exists():
        st = json.loads(SHADOW_STATE.read_text())
        if st.get("pending"):
            print(f"\n  ({len(st['pending'])} shadow position(s) still open)")

    # real live results over the same period, for reference
    if TRADE_LOG.exists():
        live = [r for r in csv.DictReader(open(TRADE_LOG))
                if r.get("resolved_won") in ("True", "False")]
        if live and rows:
            first = min(r["trade_time"] for r in rows)
            recent = [r for r in live if r["trade_time"] >= first]
            if recent:
                w = sum(1 for r in recent if r["resolved_won"] == "True")
                pnl = 0.0
                for r in recent:
                    c, s = float(r["cost_usd"]), float(r["size"])
                    pnl += (s - c) if r["resolved_won"] == "True" else -c
                print(f"\n=== REAL LIVE over the same period (from {first}) ===")
                print(f"  n={len(recent)} W={w} L={len(recent)-w} "
                      f"win_rate={w/len(recent):.1%} pnl=${pnl:+.3f}")


if __name__ == "__main__":
    main()
