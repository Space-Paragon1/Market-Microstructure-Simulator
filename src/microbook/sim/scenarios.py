"""Ready-made market environments for experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from microbook import LimitOrderBook, STPMode

from .agents import FundamentalValue, InformedConfig, InformedTrader, NoiseConfig, NoiseTrader
from .simulator import MarketSimulator
from .strategy import Strategy


@dataclass
class MarketConfig:
    """A noise + informed-flow market around a latent random-walk fundamental."""

    seed: int = 0
    tick_size: float = 0.01
    start_price: float = 100.0
    fundamental_vol: float = 0.005  # per time step, in price units
    jump_rate: float = 0.0
    jump_size: float = 0.0
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    informed: InformedConfig | None = field(default_factory=InformedConfig)
    snapshot_every: int = 10
    record_snapshots: bool = False


@dataclass
class Market:
    sim: MarketSimulator
    fundamental: FundamentalValue
    background: list[str]  # names of the background agents

    def run(self, until: int) -> MarketSimulator:
        self.sim.run(until)
        return self.sim


def build_market(cfg: MarketConfig, strategies: Sequence[Strategy] = (), *, horizon: int) -> Market:
    """Assemble a simulator with background agents plus ``strategies``.

    The fundamental path and the agents' arrival streams depend only on
    ``cfg.seed``, so runs with different ``strategies`` see the same exogenous
    flow (common random numbers) and can be compared fairly.
    """
    fv = FundamentalValue(
        cfg.start_price, cfg.fundamental_vol, jump_rate=cfg.jump_rate, jump_size=cfg.jump_size, seed=cfg.seed
    )
    background: list[Strategy] = [NoiseTrader(cfg.noise, fv, seed=cfg.seed)]
    if cfg.informed is not None:
        background.append(InformedTrader(cfg.informed, fv, seed=cfg.seed))
    lob = LimitOrderBook(cfg.tick_size, stp=STPMode.CANCEL_RESTING)
    sim = MarketSimulator(lob, [*background, *strategies], record_snapshots=cfg.record_snapshots)
    sim.schedule_snapshots(0, horizon, cfg.snapshot_every)
    return Market(sim, fv, [s.name for s in background])
