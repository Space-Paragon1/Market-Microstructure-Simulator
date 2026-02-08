from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class Fill:
    """A single execution event."""
    taker_order_id: str
    maker_order_id: str
    price: float
    qty: int
