"""End-to-end simulator throughput: the closed-loop market plus one market maker.

python benchmarks/bench_sim.py [--horizon 50000]
"""

from __future__ import annotations

import argparse
import time

from microbook.sim import AdaptiveMarketMaker, MarketConfig, build_market


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mkt = build_market(MarketConfig(seed=args.seed), [AdaptiveMarketMaker()], horizon=args.horizon)
    t0 = time.perf_counter()
    sim = mkt.run(args.horizon)
    elapsed = time.perf_counter() - t0
    print(
        f"{args.horizon:,} steps: {sim.events_processed:,} events, {len(sim.result.fills):,} fills "
        f"in {elapsed:.2f}s -> {sim.events_processed / elapsed:,.0f} events/s"
    )


if __name__ == "__main__":
    main()
