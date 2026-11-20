"""Market-maker comparison in a market with informed flow.

Every strategy trades in the same closed-loop market: a zero-intelligence noise
population and informed traders around a latent random-walk fundamental (see
``microbook.sim.agents``). For a given seed all strategies face the same
fundamental path and the same exogenous order arrivals (common random numbers),
and results are reported as mean +/- standard error across seeds.

Experiments
  1. strategy comparison at zero latency
  2. latency sensitivity of the adaptive market maker
  3. adverse selection: informed-flow intensity vs maker PnL

Usage
    python examples/compare_market_makers.py                       # tables only
    python examples/compare_market_makers.py --plot docs/figures   # + PNG figures
    python examples/compare_market_makers.py --seeds 4 --horizon 5000  # quick run
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

from microbook.sim import (
    AdaptiveMarketMaker,
    AdaptiveMMConfig,
    AvellanedaStoikovConfig,
    AvellanedaStoikovMarketMaker,
    InformedConfig,
    MarketConfig,
    MarketMaker,
    Strategy,
    build_market,
    format_table,
    strategy_report,
)
from microbook.sim.analytics import markouts

TICK = 0.01
MARKOUT_HORIZONS = (1, 5, 10, 25, 50, 100, 250)


def make_strategy(kind: str, latency: int, horizon: int) -> Strategy:
    if kind == "Symmetric MM":
        return MarketMaker("mm", half_spread_ticks=1, size=5, tick_interval=10, latency=latency)
    if kind in ("Adaptive MM", "Adaptive MM (no imbalance)"):
        cfg = AdaptiveMMConfig(
            base_half_spread_ticks=0.5,
            size=5,
            tick_interval=10,
            inv_k=0.3,
            inv_limit=50,
            vol_k=1.0,
            imb_k=0.0 if "no imbalance" in kind else 1.0,
            latency=latency,
        )
        return AdaptiveMarketMaker(cfg, name="mm")
    if kind == "Avellaneda-Stoikov":
        cfg_as = AvellanedaStoikovConfig(
            gamma=2.0, k=200, rolling_tau=2000, horizon=horizon, size=5, tick_interval=10, inv_limit=50, latency=latency
        )
        return AvellanedaStoikovMarketMaker(cfg_as, name="mm")
    raise ValueError(kind)


STRATEGIES = ["Symmetric MM", "Adaptive MM", "Adaptive MM (no imbalance)", "Avellaneda-Stoikov"]


def run_one(job: tuple[str, int, int, float, int]) -> dict[str, Any]:
    kind, seed, latency, informed_rate, horizon = job
    cfg = MarketConfig(seed=seed, informed=replace(InformedConfig(), rate=informed_rate))
    sim = build_market(cfg, [make_strategy(kind, latency, horizon)], horizon=horizon).run(horizon)
    row = strategy_report(sim, ["mm"], horizons=(1, 100))["mm"]
    m = sim.exec_metrics["mm"]
    row["markout_curve"] = markouts(m.fills, sim.result.mid_path, MARKOUT_HORIZONS, end_time=horizon)
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(key: str, scale: float = 1.0) -> float:
        xs = [r[key] for r in rows if r[key] is not None and not math.isnan(r[key])]
        return statistics.fmean(xs) * scale if xs else math.nan

    pnl = [r["final"] for r in rows]
    se = statistics.stdev(pnl) / math.sqrt(len(pnl)) if len(pnl) > 1 else math.nan
    return {
        "PnL": mean("final"),
        "PnL s.e.": se,
        "Sharpe": mean("sharpe"),
        "max DD": mean("max_drawdown"),
        "volume": mean("volume"),
        "mean |inv|": mean("mean_abs_inventory"),
        "capture (t)": mean("spread_capture", 1 / TICK),
        "markout@1 (t)": mean("markout_1", 1 / TICK),
        "markout@100 (t)": mean("markout_100", 1 / TICK),
    }


def run_grid(
    jobs: dict[str, list[tuple[str, int, int, float, int]]], pool: ProcessPoolExecutor
) -> dict[str, list[dict[str, Any]]]:
    flat = [(label, job) for label, js in jobs.items() for job in js]
    results = list(pool.map(run_one, [job for _, job in flat], chunksize=2))
    out: dict[str, list[dict[str, Any]]] = {label: [] for label in jobs}
    for (label, _), row in zip(flat, results, strict=True):
        out[label].append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=20_000)
    ap.add_argument("--plot", type=Path, default=None, help="directory to write PNG figures into")
    args = ap.parse_args()
    seeds, horizon = range(args.seeds), args.horizon
    t0 = time.perf_counter()

    with ProcessPoolExecutor() as pool:
        comp = run_grid({k: [(k, s, 0, 0.4, horizon) for s in seeds] for k in STRATEGIES}, pool)
        latencies = [0, 5, 20, 50]
        lat = run_grid({f"{L}": [("Adaptive MM", s, L, 0.4, horizon) for s in seeds] for L in latencies}, pool)
        rates = [0.0, 0.2, 0.4, 0.8]
        inf = run_grid({f"{r}": [("Adaptive MM", s, 0, r, horizon) for s in seeds] for r in rates}, pool)

    n = len(seeds)
    print(f"\n{n} seeds x {horizon:,} steps per run, tick = {TICK}  (t = ticks per share)\n")
    print("1) Strategy comparison (latency 0, informed rate 0.4)")
    print(format_table({k: summarize(v) for k, v in comp.items()}, precision=2))
    print("\n2) Latency sensitivity: Adaptive MM")
    print(format_table({f"latency {k}": summarize(v) for k, v in lat.items()}, precision=2))
    print("\n3) Adverse selection: Adaptive MM vs informed-flow intensity")
    print(format_table({f"informed {k}/step": summarize(v) for k, v in inf.items()}, precision=2))
    print(f"\n{(len(STRATEGIES) + len(latencies) + len(rates)) * n} simulations in {time.perf_counter() - t0:.1f}s")

    if args.plot is not None:
        from _plotting import plot_experiments, plot_market_run

        args.plot.mkdir(parents=True, exist_ok=True)
        plot_experiments(comp, lat, inf, MARKOUT_HORIZONS, TICK, args.plot / "mm_experiments.png")
        plot_market_run(_single_run(seed=0, horizon=horizon), TICK, args.plot / "market_run.png")
        print(f"figures written to {args.plot}/")


def _single_run(seed: int, horizon: int) -> dict[str, Any]:
    """One seed, all strategies: price path, inventories and PnL paths for the figure."""
    out: dict[str, Any] = {"pnl": {}, "inventory": {}}
    for kind in STRATEGIES:
        mkt = build_market(MarketConfig(seed=seed), [make_strategy(kind, 0, horizon)], horizon=horizon)
        sim = mkt.run(horizon)
        res = sim.result
        out["t"] = res.pnl_t
        out["pnl"][kind] = res.pnl_series["mm"]
        out["inventory"][kind] = res.inventory_series["mm"]
        out["mid"] = sim.ts.mid
        out["fundamental"] = [mkt.fundamental.at(t) for t in res.pnl_t]
    return out


Factory = Callable[[str, int, int], Strategy]

if __name__ == "__main__":
    main()
