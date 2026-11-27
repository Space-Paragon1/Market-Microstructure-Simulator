"""Ready-made market environments for experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from microbook import LimitOrderBook, STPMode
from microbook.stylized import MetaorderRecord

from .agents import (
    FundamentalValue,
    HawkesConfig,
    InformedConfig,
    InformedTrader,
    MetaorderConfig,
    MetaorderTrader,
    NoiseConfig,
    NoiseTrader,
)
from .simulator import MarketSimulator
from .strategy import Strategy


@dataclass
class MarketConfig:
    """A noise + informed-flow market around a latent random-walk fundamental.

    The defaults are the simple v0.2 market (Poisson flow, fixed-size informed
    trades). :meth:`realistic` returns the calibrated preset that reproduces
    the stylized facts of real markets (see ``examples/stylized_facts.py``).
    """

    seed: int = 0
    tick_size: float = 0.01
    start_price: float = 100.0
    fundamental_vol: float = 0.005  # per time step, in price units
    jump_rate: float = 0.0
    jump_size: float = 0.0
    vol_of_vol: float = 0.0  # > 0: stochastic volatility (see FundamentalValue)
    vol_persistence: float = 0.999
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    informed: InformedConfig | None = field(default_factory=InformedConfig)
    metaorders: MetaorderConfig | None = None
    snapshot_every: int = 10
    record_snapshots: bool = False

    @classmethod
    def realistic(cls, seed: int = 0, **overrides: Any) -> MarketConfig:
        """Calibrated market with fat tails, volatility clustering and efficient prices.

        Compared with the defaults it adds stochastic volatility in the
        fundamental, self-exciting (Hawkes) noise market orders, metaorder
        splitting, and arbitrage capital that scales with the mispricing. That
        last change removes the strong short-horizon momentum of the simple
        market (lag-1 autocorrelation of 100-step returns ~0.5 -> ~0.05).
        Calibration on seeds 1-4, 200k steps each: Hill tail index ~3,
        excess kurtosis ~28 / 2.8 / 0.8 at 10 / 100 / 1000-step horizons, and
        |return| autocorrelation decaying over ~1000 steps.
        """
        base = cls(
            seed=seed,
            vol_of_vol=0.02,
            vol_persistence=0.999,
            noise=replace(NoiseConfig(), market_rate=0.03, market_hawkes=HawkesConfig(branching=0.8)),
            informed=InformedConfig(rate=0.5, qty=2, qty_per_tick=20),
            metaorders=MetaorderConfig(rate=0.05),
        )
        return replace(base, **overrides)


@dataclass
class Market:
    sim: MarketSimulator
    fundamental: FundamentalValue
    background: list[str]  # names of the background agents

    def run(self, until: int) -> MarketSimulator:
        self.sim.run(until)
        return self.sim

    @property
    def metaorders(self) -> list[MetaorderRecord]:
        """Completed background metaorders (empty unless ``MarketConfig.metaorders`` is set)."""
        out: list[MetaorderRecord] = []
        for s in self.sim.strategies:
            if isinstance(s, MetaorderTrader) and s.name in self.background:
                out.extend(s.completed)
        return out


def build_market(cfg: MarketConfig, strategies: Sequence[Strategy] = (), *, horizon: int) -> Market:
    """Assemble a simulator with background agents plus ``strategies``.

    The fundamental path and the agents' arrival streams depend only on
    ``cfg.seed``, so runs with different ``strategies`` see the same exogenous
    flow (common random numbers) and can be compared fairly.
    """
    fv = FundamentalValue(
        cfg.start_price,
        cfg.fundamental_vol,
        jump_rate=cfg.jump_rate,
        jump_size=cfg.jump_size,
        vol_of_vol=cfg.vol_of_vol,
        vol_persistence=cfg.vol_persistence,
        seed=cfg.seed,
    )
    background: list[Strategy] = [NoiseTrader(cfg.noise, fv, seed=cfg.seed)]
    if cfg.informed is not None:
        background.append(InformedTrader(cfg.informed, fv, seed=cfg.seed))
    if cfg.metaorders is not None:
        background.append(MetaorderTrader(cfg.metaorders, seed=cfg.seed))
    lob = LimitOrderBook(cfg.tick_size, stp=STPMode.CANCEL_RESTING)
    sim = MarketSimulator(lob, [*background, *strategies], record_snapshots=cfg.record_snapshots)
    sim.schedule_snapshots(0, horizon, cfg.snapshot_every)
    return Market(sim, fv, [s.name for s in background])
