"""Matching-engine benchmark on a realistic message mix.

Real order-entry traffic is dominated by passive adds and cancels; only a small
fraction of messages trade. This workload reproduces that mix against a book
with persistent depth around the touch:

    ~55%  passive limit adds, 1-20 ticks from the opposite touch
    ~35%  cancels of a uniformly random live order (some will already be filled)
    ~10%  aggressive orders (half marketable limits, half market orders)

It only touches the long-standing public API (``place_limit``, ``place_market``,
``cancel``, ``best_bid``, ``best_ask``) so the same file can benchmark older
revisions of the engine for before/after comparisons::

    python benchmarks/bench_book.py                      # current checkout
    PYTHONPATH=/path/to/old/src python benchmarks/bench_book.py

Reported figures: sustained throughput plus per-message latency percentiles,
broken down by message type.
"""

from __future__ import annotations

import argparse
import gc
import random
import statistics
import time
from dataclasses import dataclass

from microbook import LimitOrderBook, Order, Side

TICK = 0.01
BASE_TICKS = 10_000  # 100.00


@dataclass
class OpStats:
    count: int
    p50_ns: float
    p99_ns: float
    mean_ns: float


def _px(ticks: int) -> float:
    return round(ticks * TICK, 2)


def run(n: int = 200_000, seed: int = 42) -> tuple[float, dict[str, OpStats]]:
    rng = random.Random(seed)
    lob = LimitOrderBook()
    live: list[str] = []
    next_id = 0

    def new_id() -> str:
        nonlocal next_id
        next_id += 1
        return f"o{next_id}"

    # Seed 40 levels per side, 5 orders per level.
    for lvl in range(1, 41):
        for _ in range(5):
            oid = new_id()
            lob.place_limit(Order(oid, Side.BUY, _px(BASE_TICKS - lvl), rng.randint(1, 10), ts=0))
            live.append(oid)
            oid = new_id()
            lob.place_limit(Order(oid, Side.SELL, _px(BASE_TICKS + lvl), rng.randint(1, 10), ts=0))
            live.append(oid)

    # Pre-draw the random stream so RNG cost is excluded from timings.
    draws = [(rng.random(), rng.random(), rng.randint(1, 20), rng.randint(1, 10), rng.random()) for _ in range(n)]

    lat: dict[str, list[int]] = {"add": [], "cancel": [], "aggress": []}
    gc.collect()
    gc.disable()
    t_start = time.perf_counter_ns()
    for i, (r_kind, r_side, offset, qty, r_pick) in enumerate(draws, start=1):
        side = Side.BUY if r_side < 0.5 else Side.SELL
        bb, ba = lob.best_bid(), lob.best_ask()
        bb_t = round(bb / TICK) if bb is not None else BASE_TICKS - 1
        ba_t = round(ba / TICK) if ba is not None else BASE_TICKS + 1

        if r_kind < 0.55:
            price = _px(ba_t - offset) if side is Side.BUY else _px(bb_t + offset)
            # never let a passive add cross; clamp to one tick behind the touch
            if side is Side.BUY and price >= _px(ba_t):
                price = _px(ba_t - 1)
            if side is Side.SELL and price <= _px(bb_t):
                price = _px(bb_t + 1)
            oid = new_id()
            o = Order(oid, side, price, qty, ts=i)
            t0 = time.perf_counter_ns()
            lob.place_limit(o)
            lat["add"].append(time.perf_counter_ns() - t0)
            live.append(oid)
        elif r_kind < 0.90 and live:
            j = int(r_pick * len(live))
            live[j], live[-1] = live[-1], live[j]
            oid = live.pop()
            t0 = time.perf_counter_ns()
            lob.cancel(oid)
            lat["cancel"].append(time.perf_counter_ns() - t0)
        else:
            oid = new_id()
            if r_pick < 0.5:
                price = _px(ba_t + 2) if side is Side.BUY else _px(max(1, bb_t - 2))
                o = Order(oid, side, price, qty, ts=i)
                t0 = time.perf_counter_ns()
                lob.place_limit(o)
                lat["aggress"].append(time.perf_counter_ns() - t0)
                live.append(oid)  # remainder may rest
            else:
                o = Order(oid, side, 1.0, qty, ts=i)
                t0 = time.perf_counter_ns()
                lob.place_market(o)
                lat["aggress"].append(time.perf_counter_ns() - t0)
    elapsed_s = (time.perf_counter_ns() - t_start) / 1e9
    gc.enable()

    stats = {}
    for k, xs in lat.items():
        xs.sort()
        stats[k] = OpStats(
            count=len(xs),
            p50_ns=xs[len(xs) // 2],
            p99_ns=xs[int(len(xs) * 0.99)],
            mean_ns=statistics.fmean(xs),
        )
    total_ops = sum(s.count for s in stats.values())
    return total_ops / elapsed_s, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", type=int, default=200_000, help="number of messages")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeat", type=int, default=3, help="report the best of N runs")
    args = ap.parse_args()

    best_rate, best_stats = 0.0, {}
    for _ in range(args.repeat):
        rate, stats = run(args.n, args.seed)
        if rate > best_rate:
            best_rate, best_stats = rate, stats

    print(f"messages: {args.n:,}   throughput (wall, incl. harness): {best_rate:,.0f} msg/s")
    print(f"{'type':<9}{'count':>9}{'p50 (ns)':>11}{'p99 (ns)':>11}{'mean (ns)':>11}")
    for k, s in best_stats.items():
        print(f"{k:<9}{s.count:>9,}{s.p50_ns:>11,.0f}{s.p99_ns:>11,.0f}{s.mean_ns:>11,.0f}")


if __name__ == "__main__":
    main()
