"""Stylized-fact statistics, the agents that generate them, and the calibrated market."""

from __future__ import annotations

import math
import random
import statistics

import pytest

from microbook import Order, Side
from microbook.sim import (
    FundamentalValue,
    HawkesConfig,
    HawkesProcess,
    InformedConfig,
    InformedTrader,
    MarketConfig,
    MetaorderConfig,
    MetaorderTrader,
    build_market,
)
from microbook.stylized import (
    MetaorderRecord,
    acf,
    aggressor_signs,
    excess_kurtosis,
    hill_tail_index,
    impact_curve,
    log_lags,
    power_law_fit,
    sample_returns,
    stylized_facts,
)
from microbook.types import Fill

# --------------------------------------------------------------------------- #
# Statistics on synthetic data with known answers                             #
# --------------------------------------------------------------------------- #


def test_acf_of_iid_noise_is_near_zero_and_ar1_is_phi():
    rng = random.Random(0)
    iid = [rng.gauss(0, 1) for _ in range(20_000)]
    assert all(abs(c) < 0.03 for c in acf(iid, [1, 2, 10]))
    ar, x = [], 0.0
    for _ in range(20_000):
        x = 0.7 * x + rng.gauss(0, 1)
        ar.append(x)
    c1, c2 = acf(ar, [1, 2])
    assert c1 == pytest.approx(0.7, abs=0.03) and c2 == pytest.approx(0.49, abs=0.04)


def test_acf_validates_lags():
    with pytest.raises(ValueError):
        acf([1.0, 2.0, 3.0], [3])
    assert math.isnan(acf([1.0, 1.0, 1.0], [1])[0])


def test_kurtosis_gaussian_vs_laplace():
    rng = random.Random(1)
    gauss = [rng.gauss(0, 1) for _ in range(50_000)]
    laplace = [rng.expovariate(1) * (1 if rng.random() < 0.5 else -1) for _ in range(50_000)]
    assert excess_kurtosis(gauss) == pytest.approx(0.0, abs=0.1)
    assert excess_kurtosis(laplace) == pytest.approx(3.0, abs=0.4)
    assert math.isnan(excess_kurtosis([1.0, 2.0]))


@pytest.mark.parametrize("alpha", [2.0, 3.0])
def test_hill_recovers_pareto_tail_index(alpha):
    rng = random.Random(2)
    xs = [rng.paretovariate(alpha) * (1 if rng.random() < 0.5 else -1) for _ in range(50_000)]
    assert hill_tail_index(xs, tail=0.05) == pytest.approx(alpha, rel=0.08)
    with pytest.raises(ValueError):
        hill_tail_index(xs, tail=0)


def test_power_law_fit_is_exact_on_a_power_law():
    xs = [1, 2, 5, 10, 50, 100]
    gamma, r2 = power_law_fit(xs, [3 * x**-0.5 for x in xs])
    assert gamma == pytest.approx(0.5) and r2 == pytest.approx(1.0)
    assert math.isnan(power_law_fit([1, 2], [1, 1])[0])  # too few points


def test_log_lags_are_distinct_and_span_the_range():
    lags = log_lags(1000, 20)
    assert lags[0] == 1 and lags[-1] == 1000 and lags == sorted(set(lags))


def test_sample_returns_uses_the_step_function():
    path = [(0, 100.0), (5, 101.0), (12, math.nan), (15, 99.0)]
    # samples at t = 0, 10, 20: 100 -> 101 -> 99 (t=12 is one-sided but t=10 and 20 are fine)
    assert sample_returns(path, 10, end=20) == pytest.approx([math.log(1.01), math.log(99 / 101)])
    # an undefined mid at a sample point drops both adjacent returns
    assert sample_returns(path, 12, end=24) == pytest.approx([])
    with pytest.raises(ValueError):
        sample_returns(path, 0, end=10)


def test_aggressor_signs_merge_sweeps_and_skip_unsigned_fills():
    fills = [
        Fill("t1", "m1", 100.0, 1, Side.BUY),
        Fill("t1", "m2", 100.1, 1, Side.BUY),  # same sweep
        Fill("t2", "m3", 99.9, 1, Side.SELL),
        Fill("t3", "m4", 99.9, 1, None),  # unsigned (e.g. hand-built)
        Fill("t1", "m5", 100.0, 1, Side.BUY),  # t1 again, but not consecutive
    ]
    assert aggressor_signs(fills) == [1, -1, 1]


def test_impact_curve_recovers_a_square_root_law():
    """Synthetic metaorders whose impact is exactly 0.5 * sqrt(Q / V) sigma."""
    sigma, volume = 0.01, 10_000.0
    rng = random.Random(3)
    path: list[tuple[int, float]] = []
    metas = []
    t, price = 0, 100.0
    for _ in range(400):
        q = int(10 * rng.paretovariate(1.0))
        side = 1 if rng.random() < 0.5 else -1
        path.append((t, price))
        price *= math.exp(side * 0.5 * math.sqrt(q / volume) * sigma)
        path.append((t + 10, price))
        metas.append(MetaorderRecord(side, q, t, t + 10))
        t += 20
    bins, delta, r2 = impact_curve(metas, path, daily_volume=volume, sigma=sigma)
    assert delta == pytest.approx(0.5, abs=0.02) and r2 > 0.99
    assert sum(n for _, _, n in bins) == len(metas)  # every order lands in a bin
    few, d_few, _ = impact_curve(metas[:3], path, daily_volume=volume, sigma=sigma)
    assert few == [] and math.isnan(d_few)


def test_impact_curve_bins_the_smallest_order_first():
    """Regression: exp(log(min)) can exceed min, which used to drop it in the last bin."""
    path = [(0, 100.0), (1, 100.0)]
    metas = [MetaorderRecord(1, q, 0, 1) for q in (3, 7, 11, 13, 17, 19, 23, 29, 31, 37)]
    bins, _, _ = impact_curve(metas, path, daily_volume=100.0, sigma=0.01, n_bins=4)
    assert bins[0][0] == pytest.approx(statistics.fmean([0.03, 0.07]), rel=0.5)
    assert [b[0] for b in bins] == sorted(b[0] for b in bins)


# --------------------------------------------------------------------------- #
# Generating mechanisms                                                       #
# --------------------------------------------------------------------------- #


def test_hawkes_keeps_the_mean_rate_but_clusters_arrivals():
    rate = 0.3
    h = HawkesProcess(rate, HawkesConfig(branching=0.8, timescales=(10.0, 100.0)), random.Random(4))
    counts = [h.step() for _ in range(200_000)]
    assert statistics.fmean(counts) == pytest.approx(rate, rel=0.1)
    windows = [sum(counts[i : i + 100]) for i in range(0, len(counts), 100)]
    assert statistics.pvariance(windows) / statistics.fmean(windows) > 3  # Fano factor; Poisson would be 1
    assert acf(windows, [1])[0] > 0.3  # busy periods persist


@pytest.mark.parametrize(
    "bad", [HawkesConfig(branching=1.0), HawkesConfig(timescales=()), HawkesConfig(timescales=(0,))]
)
def test_hawkes_rejects_explosive_or_empty_kernels(bad):
    with pytest.raises(ValueError):
        HawkesProcess(0.1, bad, random.Random(0))


def test_metaorder_sizes_have_the_configured_pareto_tail():
    trader = MetaorderTrader(MetaorderConfig(size_alpha=1.5, min_size=10, max_size=10**9), seed=5)
    sizes = [trader.draw_size() for _ in range(50_000)]
    assert min(sizes) >= 10
    assert hill_tail_index(sizes, tail=0.05) == pytest.approx(1.5, rel=0.1)


def test_metaorders_execute_fully_and_are_logged():
    cfg = MarketConfig(seed=6, metaorders=MetaorderConfig(rate=0.05, max_size=200))
    mkt = build_market(cfg, horizon=5_000)
    sim = mkt.run(5_000)
    done = mkt.metaorders
    assert len(done) > 50
    assert all(m.end > m.start and m.qty >= 10 and m.side in (1, -1) for m in done)
    # every completed metaorder of size Q was sent as ceil(Q / child_qty) child orders
    assert sim.exec_metrics["meta"].orders_submitted >= sum(math.ceil(m.qty / 5) for m in done)


def test_metaorder_validation():
    with pytest.raises(ValueError):
        MetaorderTrader(MetaorderConfig(min_size=0))
    with pytest.raises(ValueError):
        MetaorderTrader(MetaorderConfig(child_rate=0))


def test_constant_volatility_path_is_unchanged():
    """vol_of_vol=0 must reproduce the v0.2 fundamental exactly (same RNG draws)."""
    fv = FundamentalValue(100.0, 0.01, seed=9)
    rng = random.Random("9/fundamental")
    v, expected = 100.0, [100.0]
    for _ in range(100):
        v += rng.gauss(0.0, 0.01)
        expected.append(v)
    assert [fv.at(t) for t in range(101)] == expected


def test_stochastic_volatility_keeps_average_variance_and_clusters():
    fv = FundamentalValue(1_000.0, 0.01, vol_of_vol=0.05, vol_persistence=0.995, seed=10)
    dv = [fv.at(t + 1) - fv.at(t) for t in range(200_000)]
    assert statistics.fmean(x * x for x in dv) == pytest.approx(0.01**2, rel=0.15)
    assert excess_kurtosis(dv) > 1.0
    assert acf([abs(x) for x in dv], [10])[0] > 0.1
    with pytest.raises(ValueError):
        FundamentalValue(100.0, 0.01, vol_persistence=1.0)


def test_informed_size_grows_with_mispricing():
    from microbook import LimitOrderBook

    fv = FundamentalValue(100.0, 0.0, seed=0)  # constant V = 100
    lob = LimitOrderBook(tick_size=0.01)
    lob.place_limit(Order("a", Side.SELL, 99.90, 1_000))  # 10 ticks below V
    lob.place_limit(Order("b", Side.BUY, 99.80, 1_000))
    trader = InformedTrader(InformedConfig(rate=50.0, qty=2, qty_per_tick=3, threshold_ticks=1), fv)
    actions = trader.on_timer(1, lob)
    order = next(a.kwargs["order"] for a in actions if "order" in a.kwargs)
    assert order.side is Side.BUY and order.price == pytest.approx(99.99)
    assert order.qty == 2 + 3 * 9  # gap beyond the threshold: 9 ticks


# --------------------------------------------------------------------------- #
# The calibrated market                                                       #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def runs():
    horizon = 40_000
    out = {}
    for name, cfg in {
        "simple": MarketConfig(seed=1, snapshot_every=1_000),
        "realistic": MarketConfig.realistic(seed=1, snapshot_every=1_000),
    }.items():
        mkt = build_market(cfg, horizon=horizon)
        sim = mkt.run(horizon)
        out[name] = (mkt, sim, horizon)
    return out


def test_realistic_market_has_fat_tails_and_efficient_prices(runs):
    _, sim, h = runs["realistic"]
    res = sim.result
    sf = stylized_facts(res.mid_path, res.fills, end=h)
    assert sf.hill_alpha < 4.5  # fat tails (Gaussian: grows without bound)
    assert sf.kurtosis_short > 5
    assert abs(sf.return_acf_1) < 0.1
    r100 = sample_returns(res.mid_path, 100, end=h)
    assert abs(acf(r100, [1])[0]) < 0.2
    assert sf.abs_return_acf[1] > 0.1  # volatility clustering
    sim.lob.check_invariants()


def test_simple_market_shows_the_momentum_the_preset_removes(runs):
    _, sim, h = runs["simple"]
    r100 = sample_returns(sim.result.mid_path, 100, end=h)
    assert acf(r100, [1])[0] > 0.3


def test_realistic_market_still_tracks_the_fundamental(runs):
    mkt, sim, _ = runs["realistic"]
    pts = [(t, m) for t, m in zip(sim.ts.t, sim.ts.mid, strict=True) if not math.isnan(m)]
    err = statistics.fmean(abs(m - mkt.fundamental.at(t)) for t, m in pts)
    assert err < 0.05  # < 5 ticks on average


def test_realistic_preset_accepts_overrides():
    cfg = MarketConfig.realistic(seed=7, snapshot_every=50)
    assert cfg.seed == 7 and cfg.snapshot_every == 50 and cfg.metaorders is not None
