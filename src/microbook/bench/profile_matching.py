from __future__ import annotations
import cProfile
import pstats
from pstats import SortKey

from microbook.bench.bench_matching import run_bench


def main():
    pr = cProfile.Profile()
    pr.enable()

    # smaller first so it finishes fast
    r = run_bench(n=50_000)

    pr.disable()

    print(f"Orders: {r.n_orders:,}  Time: {r.seconds:.3f}s  Rate: {r.orders_per_sec:,.0f} orders/s")

    stats = pstats.Stats(pr).strip_dirs().sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(30)  # top 30 hottest functions


if __name__ == "__main__":
    main()
