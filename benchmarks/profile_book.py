"""cProfile the matching engine on the realistic message mix from ``bench_book``.

python benchmarks/profile_book.py [-n 100000] [--top 25]
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bench_book import run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=100_000)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    prof = cProfile.Profile()
    prof.enable()
    rate, _ = run(args.n)
    prof.disable()
    print(f"{args.n:,} messages at {rate:,.0f} msg/s (slowed by the profiler)\n")
    pstats.Stats(prof).strip_dirs().sort_stats(pstats.SortKey.TIME).print_stats(args.top)


if __name__ == "__main__":
    main()
