"""Exports every live trade to a human-readable CSV.

trade_log.csv is the machine record: epoch window ids, order hashes, token ids, and
a stop-out whose realised P&L lives in a different column from an ordinary loss. This
produces something a person can actually read and audit -- real clock times, the
window's actual time range, one unambiguous outcome column, and a running balance.

Also writes a one-page summary of the same numbers grouped by era, because the single
most important fact about this dataset is that it contains two different strategies:
everything before 2026-09-07 09:55:55 ran an ask-triggered rule that the backtest had
never validated, and everything after runs a mid-triggered one with a stop-loss.
"""
import csv
from datetime import datetime, timezone, timedelta
from pathlib import Path

REPO = Path(r"D:\Finance\polymarket")
SRC = REPO / "results" / "btc_5m_live" / "trade_log.csv"
OUT = REPO / "results" / "btc_5m_live" / "trades_readable.csv"
SUMMARY = REPO / "results" / "btc_5m_live" / "trades_summary.txt"

LOCAL = timezone(timedelta(hours=1))
MID_FIX = datetime(2026, 9, 7, 9, 55, 55, tzinfo=LOCAL)
STOP_ADDED = datetime(2026, 9, 7, 9, 43, 38, tzinfo=LOCAL)


def f(v, nd=4, dash="-"):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return dash


def main():
    rows = [r for r in csv.DictReader(open(SRC))
            if r.get("resolved_won") in ("True", "False")]

    def tt(r):
        return datetime.strptime(r["trade_time"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=LOCAL)

    rows.sort(key=tt)

    out = []
    running = 0.0
    for i, r in enumerate(rows, 1):
        w = int(r["window_end"])
        open_dt = datetime.fromtimestamp(w, timezone.utc).astimezone(LOCAL)
        close_dt = datetime.fromtimestamp(w + 300, timezone.utc).astimezone(LOCAL)
        entered = tt(r)
        won = r["resolved_won"] == "True"
        stopped = r.get("exited") == "True"

        cost = float(r["cost_usd"])
        shares = float(r["size"])
        if stopped and r.get("realized_pnl_usd"):
            pnl = float(r["realized_pnl_usd"])
            outcome = "STOPPED OUT"
        elif won:
            pnl = shares - cost
            outcome = "WON"
        else:
            pnl = -cost
            outcome = "LOST"
        running += pnl

        out.append({
            "#": i,
            "date": entered.strftime("%Y-%m-%d"),
            "time_entered": entered.strftime("%H:%M:%S"),
            "window": f"{open_dt:%H:%M}-{close_dt:%H:%M}",
            "secs_before_close": int((close_dt - entered).total_seconds()),
            "strategy_era": "ask-trigger (v1)" if entered < MID_FIX else "mid-trigger (v2)",
            "side": r["side"],
            "entry_price": f(r["ask_price"]),
            "shares": f(shares, 2),
            "cost_usd": f(cost, 2),
            "outcome": outcome,
            "exit_price": f(r.get("exit_price"), 4) if stopped else "-",
            "pnl_usd": f(pnl, 4),
            "running_total_usd": f(running, 4),
            "entry_bid": f(r.get("entry_bid")),
            "entry_ask": f(r.get("entry_ask")),
            "entry_spread": f(r.get("entry_spread")),
            "false_stop": r.get("false_stop") or "-",
        })

    cols = list(out[0].keys())
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(out)
    print(f"wrote {len(out)} trades -> {OUT}")

    # summary
    lines = []
    def block(label, sel):
        g = [o for o in out if sel(o)]
        if not g:
            return
        n = len(g)
        won = sum(1 for o in g if o["outcome"] == "WON")
        lost = sum(1 for o in g if o["outcome"] == "LOST")
        stop = sum(1 for o in g if o["outcome"] == "STOPPED OUT")
        pnl = sum(float(o["pnl_usd"]) for o in g)
        inv = sum(float(o["cost_usd"]) for o in g)
        avg = sum(float(o["entry_price"]) for o in g) / n
        lines.append(f"{label}")
        lines.append(f"  trades          {n}")
        lines.append(f"  won             {won}   ({won/n:.1%})")
        lines.append(f"  lost outright   {lost}")
        lines.append(f"  stopped out     {stop}")
        lines.append(f"  avg entry price {avg:.4f}")
        lines.append(f"  capital cycled  ${inv:,.2f}")
        lines.append(f"  P&L             ${pnl:+,.2f}   ({pnl/inv:+.2%} on turnover)")
        lines.append("")

    lines.append("POLYMARKET BTC 5-MINUTE STRATEGY -- TRADE SUMMARY")
    lines.append(f"generated {datetime.now(LOCAL):%Y-%m-%d %H:%M:%S}")
    lines.append("=" * 62)
    lines.append("")
    block("ALL TRADES", lambda o: True)
    block("v1  ask-triggered (before 2026-09-07 09:55:55)",
          lambda o: o["strategy_era"].startswith("ask"))
    block("v2  mid-triggered + stop-loss (after)",
          lambda o: o["strategy_era"].startswith("mid"))
    lines.append("=" * 62)
    lines.append("NOTE ON THE TWO ERAS")
    lines.append("  v1 triggered on the best ASK. The backtest that justified the")
    lines.append("  strategy read TRADE PRINTS. Those are not the same signal: only")
    lines.append("  13.5% of v1 trades would have been taken by the validated rule,")
    lines.append("  and 11 of its 12 losses came from the other 86.5%.")
    lines.append("")
    lines.append("  v2 triggers on the mid with a spread guard, and adds a stop-loss")
    lines.append("  at 0.70 which converts a total loss into a partial one.")
    lines.append("")
    lines.append("  Live trading was stopped on 2026-09-07; the bot now runs in")
    lines.append("  collect-only mode gathering order-book data at no risk.")
    SUMMARY.write_text("\n".join(lines))
    print(f"wrote summary -> {SUMMARY}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
