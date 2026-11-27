"""Does the simulated market behave like a real one?

Measures the stylized facts of real markets (fat tails, aggregational
Gaussianity, no return autocorrelation, volatility clustering, long memory of
order signs, square-root impact) in two closed-loop markets:

* ``simple``     the default ``MarketConfig()`` (Poisson flow, fixed-size informed trades)
* ``realistic``  ``MarketConfig.realistic()`` (stochastic volatility, Hawkes market
                 orders, metaorder splitting, arbitrage that scales with the mispricing)

Time steps are read as seconds: the fundamental's volatility then corresponds to
about 0.8% a day for a $100 stock with a 1-cent tick. Statistics are averaged
across seeds (mean +/- standard error).

Usage
    python examples/stylized_facts.py                       # table only
    python examples/stylized_facts.py --plot docs/figures   # + PNG figure
    python examples/stylized_facts.py --seeds 2 --horizon 20000  # quick run
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from microbook.sim import MarketConfig, build_market
from microbook.stylized import (
    acf,
    aggressor_signs,
    excess_kurtosis,
    hill_tail_index,
    log_lags,
    metaorder_impacts,
    power_law_fit,
    sample_returns,
)

TICK = 0.01
DAY = 23_400  # steps per "day" (6.5 hours of seconds)
KURT_DTS = (10, 30, 100, 300, 1_000, 3_000)
ACF_DT = 10
ACF_LAGS = (1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300)  # in units of ACF_DT
SIGN_LAGS = tuple(log_lags(1_000, 14))
MARKETS = ("simple", "realistic")


def config(market: str, seed: int) -> MarketConfig:
    if market == "simple":
        return MarketConfig(seed=seed, snapshot_every=1_000)
    return MarketConfig.realistic(seed=seed, snapshot_every=1_000)


def run_one(job: tuple[str, int, int]) -> dict[str, Any]:
    market, seed, horizon = job
    mkt = build_market(config(market, seed), horizon=horizon)
    sim = mkt.run(horizon)
    res = sim.result
    path = res.mid_path
    out: dict[str, Any] = {}

    out["kurtosis"] = {dt: excess_kurtosis(sample_returns(path, dt, end=horizon)) for dt in KURT_DTS}
    r10 = sample_returns(path, ACF_DT, end=horizon)
    r100 = sample_returns(path, 100, end=horizon)
    lags = [k for k in ACF_LAGS if k < len(r10) - 1]
    out["acf_r"] = dict(zip(lags, acf(r10, lags), strict=True))
    out["acf_abs"] = dict(zip(lags, acf([abs(x) for x in r10], lags), strict=True))
    out["acf_r100_1"] = acf(r100, [1])[0] if len(r100) > 2 else math.nan
    out["hill"] = hill_tail_index(r10)
    sd = statistics.pstdev(r100) if len(r100) > 1 else math.nan
    out["std_abs_r100"] = [abs(x) / sd for x in r100] if sd and not math.isnan(sd) else []

    signs = aggressor_signs(res.fills)
    slags = [k for k in SIGN_LAGS if k < len(signs) - 1]
    out["acf_sign"] = dict(zip(slags, acf(signs, slags), strict=True)) if slags else {}

    steps = [(t, m) for t, m in zip(sim.ts.t, sim.ts.mid, strict=True) if not math.isnan(m)]
    out["tracking"] = statistics.fmean(abs(m - mkt.fundamental.at(t)) for t, m in steps) / TICK

    # impact: daily volume and daily volatility of the same run
    days = horizon / DAY
    volume = sum(f.qty for f in res.fills) / days
    sigma_day = sd * math.sqrt(DAY / 100) if sd else math.nan
    out["impacts"] = (
        metaorder_impacts(mkt.metaorders, path, daily_volume=volume, sigma=sigma_day) if mkt.metaorders else []
    )
    return out


def mean_se(xs: list[float]) -> tuple[float, float]:
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if not xs:
        return math.nan, math.nan
    return statistics.fmean(xs), statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else math.nan


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def curve(key: str) -> dict[int, tuple[float, float]]:
        keys = sorted({k for r in rows for k in r[key]})
        return {k: mean_se([r[key][k] for r in rows if k in r[key]]) for k in keys}

    s: dict[str, Any] = {
        "kurtosis": {dt: mean_se([r["kurtosis"][dt] for r in rows]) for dt in KURT_DTS},
        "acf_r": curve("acf_r"),
        "acf_abs": curve("acf_abs"),
        "acf_sign": curve("acf_sign"),
        "hill": mean_se([r["hill"] for r in rows]),
        "acf_r100_1": mean_se([r["acf_r100_1"] for r in rows]),
        "tracking": mean_se([r["tracking"] for r in rows]),
        "std_abs_r100": [x for r in rows for x in r["std_abs_r100"]],
    }
    sign = s["acf_sign"]
    s["sign_gamma"] = power_law_fit(list(sign), [m for m, _ in sign.values()])[0] if sign else math.nan
    pts = [p for r in rows for p in r["impacts"]]
    s["impact_bins"] = impact_bins(pts)
    # square-root-law prefactor Y in  I = Y * sigma * sqrt(Q/V), over metaorders >= 0.1% of daily volume
    big = [(q, i) for q, i in pts if q >= 1e-3]
    s["impact_Y"] = mean_se([i / math.sqrt(q) for q, i in big])
    s["n_metaorders"] = len(pts)
    return s


def impact_bins(
    pts: list[tuple[float, float]], n_bins: int = 6, min_count: int = 10
) -> list[tuple[float, float, float]]:
    """Log-spaced size bins: (mean Q/V, mean impact/sigma, standard error); sparse bins dropped."""
    if len(pts) < n_bins * min_count:
        return []
    lo, hi = min(q for q, _ in pts), max(q for q, _ in pts)
    width = (math.log(hi) - math.log(lo)) / n_bins
    groups: list[list[tuple[float, float]]] = [[] for _ in range(n_bins)]
    for q, i in pts:
        groups[min(int((math.log(q) - math.log(lo)) / width), n_bins - 1)].append((q, i))
    return [(statistics.fmean(q for q, _ in g), *mean_se([i for _, i in g])) for g in groups if len(g) >= min_count]


def fmt(ms: tuple[float, float], prec: int = 2) -> str:
    m, se = ms
    if math.isnan(m):
        return "-"
    return f"{m:.{prec}f}" if math.isnan(se) else f"{m:.{prec}f} ± {se:.{prec}f}"


def table(summ: dict[str, dict[str, Any]]) -> str:
    def sign_at(s: dict[str, Any], lag: int) -> tuple[float, float]:
        near = min(s["acf_sign"], key=lambda k: abs(k - lag)) if s["acf_sign"] else None
        return s["acf_sign"][near] if near is not None else (math.nan, math.nan)

    rows: list[tuple[str, str, str, str]] = [
        ("Tail index (Hill, 10-step returns)", *(fmt(summ[m]["hill"]) for m in MARKETS), "≈ 3 (2-5)"),
        *(
            (f"Excess kurtosis, {dt}-step returns", *(fmt(summ[m]["kurtosis"][dt]) for m in MARKETS), ref)
            for dt, ref in ((10, "large"), (100, "> 0"), (1_000, "→ 0 at long horizons"))
        ),
        ("Return ACF lag 1, 10-step", *(fmt(summ[m]["acf_r"].get(1, (math.nan,) * 2)) for m in MARKETS), "≈ 0 or < 0"),
        ("Return ACF lag 1, 100-step", *(fmt(summ[m]["acf_r100_1"]) for m in MARKETS), "≈ 0"),
        *(
            (
                f"|Return| ACF lag {k * ACF_DT} steps",
                *(fmt(summ[m]["acf_abs"].get(k, (math.nan,) * 2)) for m in MARKETS),
                "> 0, slow decay",
            )
            for k in (1, 10, 100)
        ),
        *(
            (f"Order-sign ACF lag {lag}", *(fmt(sign_at(summ[m], lag)) for m in MARKETS), "≈ c·lag^-0.5")
            for lag in (1, 10, 100, 1_000)
        ),
        (
            "Impact prefactor Y, I = Yσ√(Q/V)",
            *(fmt(summ[m]["impact_Y"]) if summ[m]["n_metaorders"] else "-" for m in MARKETS),
            "≈ 0.5-1",
        ),
        ("Tracking error |mid - V| (ticks)", *(fmt(summ[m]["tracking"]) for m in MARKETS), "-"),
    ]
    header = ("Statistic", *MARKETS, "Real markets")
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(4)]
    lines = [
        "  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(r, widths, strict=True)))
        for r in [header, *rows]
    ]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=200_000)
    ap.add_argument("--plot", type=Path, default=None, help="directory to write the PNG figure into")
    args = ap.parse_args()
    t0 = time.perf_counter()
    jobs = [(m, s, args.horizon) for m in MARKETS for s in range(1, args.seeds + 1)]
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(run_one, jobs))
    by_market = {m: [r for (mm, _, _), r in zip(jobs, results, strict=True) if mm == m] for m in MARKETS}
    summ = {m: summarize(rows) for m, rows in by_market.items()}

    print(f"\n{args.seeds} seeds x {args.horizon:,} steps per market (1 step = 1 s, tick = {TICK})\n")
    print(table(summ))
    print(f"\n{len(jobs)} simulations in {time.perf_counter() - t0:.1f}s")

    if args.plot is not None:
        from _plotting import plot_stylized_facts

        args.plot.mkdir(parents=True, exist_ok=True)
        plot_stylized_facts(summ, KURT_DTS, ACF_DT, args.plot / "stylized_facts.png")
        print(f"figure written to {args.plot}/stylized_facts.png")


if __name__ == "__main__":
    main()
