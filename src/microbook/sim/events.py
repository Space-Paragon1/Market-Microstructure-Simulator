from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from microbook.order import Order


class EventType(str, Enum):
    SUBMIT = "SUBMIT"  # submit an order (limit or market)
    CANCEL = "CANCEL"  # cancel a resting order by id
    MODIFY = "MODIFY"  # amend a resting order (new_price / new_qty)
    SNAPSHOT = "SNAPSHOT"  # record book state and call every strategy's on_tick
    TIMER = "TIMER"  # wake a single strategy (its on_timer hook)


@dataclass(order=True)
class Event:
    """A scheduled simulator event. Ordered by ``(time, seq)``; ``seq`` is the
    global scheduling sequence, which makes ties deterministic (FIFO)."""

    time: int
    seq: int
    etype: EventType = field(compare=False)

    order: Order | None = field(default=None, compare=False)
    order_id: str | None = field(default=None, compare=False)
    modify: dict[str, Any] | None = field(default=None, compare=False)  # new_price / new_qty
    owner: int | None = field(default=None, compare=False)  # index of the originating strategy
