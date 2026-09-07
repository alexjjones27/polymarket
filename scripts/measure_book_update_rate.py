"""Before raising the live poll rate from 1s to 0.1s, measure whether that actually
buys anything.

Two things decide it, and neither can be assumed:

  1. Does the book endpoint actually change faster than once a second? If Polymarket
     caches it server-side, polling 10x more often returns the same bytes ten times
     and we gain nothing while spending ten times the rate-limit budget.

  2. Does the endpoint tolerate the rate? At 1s with two sides and ~2 live windows we
     make ~2-4 req/s. At 0.1s that becomes 20-40 req/s, which is where public APIs
     usually start throttling or banning.

Reports observed update interval, duplicate-response rate, latency, and any errors.
"""
import os
import statistics as st
import sys
import time
from pathlib import Path

REPO = Path(r"D:\Finance\polymarket")
sys.path.insert(0, str(REPO / "src"))
from dotenv import load_dotenv
import polymarket_final_pct as pmf

DURATION_S = 30
INTERVAL_S = 0.1


def main():
    load_dotenv(REPO / ".env")
    from py_clob_client_v2 import ClobClient

    client = ClobClient(host="https://clob.polymarket.com", chain_id=137,
                        key=os.environ["POLYMARKET_PRIVATE_KEY"], signature_type=3,
                        funder=os.environ["POLYMARKET_PROXY_ADDRESS"])
    client.set_api_creds(client.create_or_derive_api_key())

    # find a window that is currently live and actually trading
    now = int(time.time())
    token = None
    for cand in (((now - 300) // 300) * 300, ((now) // 300) * 300, ((now - 600) // 300) * 300):
        res = pmf._get(pmf.GAMMA_BASE, "/events", {"slug": f"btc-updown-5m-{cand}"})
        if not res:
            continue
        toks = pmf._safe_json_list(res[0]["markets"][0].get("clobTokenIds"))
        if len(toks) != 2:
            continue
        try:
            client.get_order_book(toks[0])
            token = toks[0]
            print(f"polling window {cand} (close {cand+300}, now {now})")
            break
        except Exception:
            continue
    if not token:
        print("no live orderbook available right now -- rerun during an active window")
        return

    samples, latencies, errors = [], [], 0
    t_end = time.time() + DURATION_S
    while time.time() < t_end:
        t0 = time.time()
        try:
            book = client.get_order_book(token)
            bids = sorted(book.get("bids") or [], key=lambda b: float(b["price"]), reverse=True)
            asks = sorted(book.get("asks") or [], key=lambda a: float(a["price"]))
            state = (bids[0]["price"] if bids else None, bids[0]["size"] if bids else None,
                     asks[0]["price"] if asks else None, asks[0]["size"] if asks else None)
            samples.append((t0, state))
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  error: {e}")
        latencies.append(time.time() - t0)
        sleep = INTERVAL_S - (time.time() - t0)
        if sleep > 0:
            time.sleep(sleep)

    if not samples:
        print("no successful samples")
        return

    changes = [samples[i][0] for i in range(1, len(samples))
               if samples[i][1] != samples[i-1][1]]
    dup = len(samples) - 1 - len(changes)
    gaps = [changes[i] - changes[i-1] for i in range(1, len(changes))]

    print(f"\n=== {len(samples)} samples over {DURATION_S}s at {INTERVAL_S}s target ===")
    print(f"  achieved rate        : {len(samples)/DURATION_S:.1f} req/s")
    print(f"  errors               : {errors}")
    print(f"  median latency       : {st.median(latencies)*1000:.0f} ms")
    print(f"  p90 latency          : {sorted(latencies)[int(0.9*len(latencies))]*1000:.0f} ms")
    print(f"  distinct book changes: {len(changes)}")
    print(f"  duplicate responses  : {dup} = {dup/max(1,len(samples)-1):.1%}")
    if gaps:
        print(f"  median gap between real changes: {st.median(gaps)*1000:.0f} ms")
        print(f"  min gap                        : {min(gaps)*1000:.0f} ms")
        faster = sum(1 for g in gaps if g < 0.9)
        print(f"  changes arriving faster than 1s: {faster}/{len(gaps)} = {faster/len(gaps):.1%}")
    print()
    if gaps and st.median(gaps) < 0.5:
        print("  -> book DOES move faster than 1s; sub-second polling sees real new information")
    else:
        print("  -> book does NOT reliably move faster than 1s; faster polling mostly re-reads")


if __name__ == "__main__":
    main()
