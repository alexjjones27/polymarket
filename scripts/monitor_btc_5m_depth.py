"""Live feasibility check for the 95%+/<=90s-left edge found in
backtest_btc_5m_sweep.py: historical CLOB price-history data only has
trade prints, not order-book depth, so the only way to know whether
there's ever enough live size to actually buy into that scenario is to
watch it happen in real time.

Follows the currently-active 5-min BTC window and the ones after it,
polling the real order book every few seconds, and specifically flags +
records full book snapshots whenever either side's best ask reaches the
target zone (>=95%) with <=TARGET_SECONDS_LEFT remaining -- the exact
scenario the backtest found an edge in. Runs for RUN_MINUTES, appending
structured rows to a CSV for later review.
"""
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import polymarket_final_pct as pmf
from py_clob_client_v2 import ClobClient

TARGET_PRICE = 0.95
TARGET_SECONDS_LEFT = 90
POLL_INTERVAL_S = 3
RUN_MINUTES = 240  # ~4 hours, to span more than one volatility regime

OUT_PATH = Path(__file__).resolve().parents[1] / "results" / "btc_5m_momentum" / "live_depth_log.csv"
FIELDS = ["poll_time", "window_end_epoch", "secs_left", "side", "best_ask_price", "best_ask_size",
          "best_bid_price", "best_bid_size", "n_asks", "n_bids", "is_target_scenario"]


def get_current_window():
    epoch_now = int(time.time())
    window_end = ((epoch_now // 300) + 1) * 300
    slug = f"btc-updown-5m-{window_end}"
    res = pmf._get(pmf.GAMMA_BASE, "/events", {"slug": slug})
    if not res:
        return None
    m = res[0]["markets"][0]
    tokens = pmf._safe_json_list(m.get("clobTokenIds"))
    return {"window_end": window_end, "tokens": tokens, "question": m["question"]}


def poll_book(client, tokens, window_end, writer, f):
    now = time.time()
    secs_left = window_end - now
    for side, token in zip(["Up", "Down"], tokens):
        try:
            book = client.get_order_book(token)
            asks = sorted(book.get("asks") or [], key=lambda a: float(a["price"]))
            bids = sorted(book.get("bids") or [], key=lambda b: -float(b["price"]))
            best_ask = asks[0] if asks else None
            best_bid = bids[0] if bids else None
            is_target = bool(best_ask and float(best_ask["price"]) >= TARGET_PRICE
                              and secs_left <= TARGET_SECONDS_LEFT)
            row = {
                "poll_time": now, "window_end_epoch": window_end, "secs_left": round(secs_left, 1),
                "side": side,
                "best_ask_price": best_ask["price"] if best_ask else None,
                "best_ask_size": best_ask["size"] if best_ask else None,
                "best_bid_price": best_bid["price"] if best_bid else None,
                "best_bid_size": best_bid["size"] if best_bid else None,
                "n_asks": len(asks), "n_bids": len(bids), "is_target_scenario": is_target,
            }
            writer.writerow(row)
            f.flush()
            if is_target:
                print(f"  TARGET HIT: {side} ask={best_ask['price']}x{best_ask['size']} "
                      f"secs_left={secs_left:.1f}")
        except Exception as e:
            print(f"  poll error ({side}): {e}")
    return secs_left


def main():
    client = ClobClient(host="https://clob.polymarket.com", chain_id=137)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not OUT_PATH.exists()
    deadline = time.time() + RUN_MINUTES * 60

    with open(OUT_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()

        current = None
        while time.time() < deadline:
            if current is None:
                current = get_current_window()
                if current is None:
                    time.sleep(POLL_INTERVAL_S)
                    continue
                print(f"Watching {current['question']} (ends epoch {current['window_end']})")

            secs_left = poll_book(client, current["tokens"], current["window_end"], writer, f)
            if secs_left <= 2:
                print("  window resolved, moving to next\n")
                current = None
                time.sleep(2)
                continue
            time.sleep(min(POLL_INTERVAL_S, max(1, secs_left / 4)))

    print(f"Done. Log saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
