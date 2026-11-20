"""Tour of the matching engine API: order types, execution reports, modify
semantics and self-trade prevention.

    python examples/quickstart.py
"""

from __future__ import annotations

from microbook import ExecutionReport, LimitOrderBook, Order, Side, STPMode, TimeInForce


def show(label: str, r: ExecutionReport) -> None:
    parts = [r.status.value]
    if r.fills:
        parts.append("fills " + ", ".join(f"{f.qty}@{f.price}" for f in r.fills))
    if r.resting_qty:
        parts.append(f"resting {r.resting_qty}")
    if r.canceled_qty:
        parts.append(f"canceled {r.canceled_qty}")
    if r.stp_canceled:
        parts.append(f"STP removed {list(r.stp_canceled)}")
    if r.reason:
        parts.append(f"({r.reason})")
    print(f"{label:<38} {' | '.join(parts)}")


def main() -> None:
    lob = LimitOrderBook(tick_size=0.01, stp=STPMode.CANCEL_RESTING)

    # Resting liquidity
    for i, (px, qty) in enumerate([(100.02, 5), (100.03, 10), (100.05, 20)]):
        lob.submit(Order.limit(f"ask{i}", Side.SELL, px, qty, owner="mm"))
    for i, (px, qty) in enumerate([(99.99, 5), (99.98, 10)]):
        lob.submit(Order.limit(f"bid{i}", Side.BUY, px, qty, owner="mm"))
    print(lob)
    print("depth:", lob.depth(3), "\n")

    # Marketable limit: sweeps two levels, remainder rests (GTC)
    show("buy 20 @ 100.03 (GTC)", lob.submit(Order.limit("t1", Side.BUY, 100.03, 20, owner="fund")))
    # IOC: fill what is available, cancel the rest
    show("sell 30 @ 99.98 (IOC)", lob.submit(Order.limit("t2", Side.SELL, 99.98, 30, tif=TimeInForce.IOC)))
    # FOK: all or nothing, no side effects when killed
    show("buy 100 @ 100.05 (FOK)", lob.submit(Order.limit("t3", Side.BUY, 100.05, 100, tif=TimeInForce.FOK)))
    # Post-only: rejected if it would take liquidity
    show("buy 1 @ 100.05 post-only", lob.submit(Order.limit("t4", Side.BUY, 100.05, 1, post_only=True)))
    # Off-grid price
    show("buy 1 @ 100.005 (off tick)", lob.submit(Order.limit("t5", Side.BUY, 100.005, 1)))
    # Self-trade prevention: "mm" lifting its own offer cancels the resting order instead
    show("mm buys 5 @ 100.05 (STP)", lob.submit(Order.limit("t6", Side.BUY, 100.05, 5, owner="mm")))

    print()
    lob.submit(Order.limit("q1", Side.BUY, 99.95, 10))
    lob.submit(Order.limit("q2", Side.BUY, 99.95, 10))
    show("reduce q1 to 4 (keeps priority)", lob.modify("q1", new_qty=4))
    print(f"{'  queue ahead of q2':<38} {lob.queue_ahead('q2')} shares")
    lob.submit(Order.limit("ask9", Side.SELL, 100.10, 3))
    show("reprice q2 through the ask (trades)", lob.modify("q2", new_price=100.10))
    print("\nfinal depth:", lob.depth(3))
    lob.check_invariants()


if __name__ == "__main__":
    main()
