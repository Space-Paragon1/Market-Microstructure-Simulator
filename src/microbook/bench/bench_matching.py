from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Tuple

from microbook import LimitOrderBook, Order, Side


@dataclass
class BenchResult:
    n_orders: int
    seconds: float
    orders_per_sec: float


def run_bench(n: int = 200_000) -> BenchResult:
    lob = LimitOrderBook()

    # Seed book with depth
    ts = 0
    for i in range(1000):
        ts += 1
        lob.place_limit(Order(f"s{i}", Side.SELL, 101.0 + (i % 10), 10, ts=ts))
        ts += 1
        lob.place_limit(Order(f"b{i}", Side.BUY,  99.0 - (i % 10), 10, ts=ts))

    start = time.perf_counter()
    for i in range(n):
        ts += 1
        # alternate aggressive buys/sells to trigger matching
        if i % 2 == 0:
            lob.place_limit(Order(f"x{i}", Side.BUY, 200.0, 5, ts=ts))  # crosses
        else:
            lob.place_limit(Order(f"x{i}", Side.SELL, 1.0, 5, ts=ts))   # crosses
    end = time.perf_counter()

    secs = end - start
    ops = n / secs if secs > 0 else float("inf")
    return BenchResult(n_orders=n, seconds=secs, orders_per_sec=ops)


if __name__ == "__main__":
    r = run_bench()
    print(f"Orders: {r.n_orders:,}")
    print(f"Time:   {r.seconds:.3f}s")
    print(f"Rate:   {r.orders_per_sec:,.0f} orders/s")
