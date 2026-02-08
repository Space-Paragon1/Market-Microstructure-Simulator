from __future__ import annotations
from dataclasses import dataclass
from .types import Side


@dataclass
class Order:
    order_id: str
    side: Side
    price: float
    qty: int
    ts: int  # strictly increasing sequence number for time priority

    def __post_init__(self) -> None:
        if self.qty <= 0:
            raise ValueError("qty must be positive")
        if self.price <= 0:
            raise ValueError("price must be positive")
