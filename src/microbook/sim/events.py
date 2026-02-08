from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from microbook.order import Order


class EventType(str, Enum):
    SUBMIT = "SUBMIT"     # submit limit/market order
    CANCEL = "CANCEL"     # cancel existing order_id
    MODIFY = "MODIFY"     # modify existing order_id
    SNAPSHOT = "SNAPSHOT" # record book state


@dataclass(order=True)
class Event:
    # Priority order for heapq: (time, seq)
    time: int
    seq: int
    etype: EventType = field(compare=False)

    order: Optional[Order] = field(default=None, compare=False)
    order_id: Optional[str] = field(default=None, compare=False)
    modify: Optional[Dict[str, Any]] = field(default=None, compare=False)  # new_price/new_qty
