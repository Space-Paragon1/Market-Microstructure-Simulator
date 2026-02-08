from __future__ import annotations
from typing import List

from microbook.sim.simulator import MarketSimulator
from microbook.sim.events import EventType


def schedule_strategy_ticks(sim: MarketSimulator, start: int, end: int, every: int = 1) -> None:
    for t in range(start, end + 1, every):
        sim.schedule(t, EventType.SNAPSHOT)  # snapshots help debugging
        # We'll call strategy ticks inside the sim loop by using SNAPSHOT as a "tick hook"
