from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from .order_types import OrderType, OrderPriority, OrderState

@dataclass
class Order:
    id: int
    location: int
    order_type: OrderType
    priority: OrderPriority
    state: OrderState = OrderState.NEW
    assigned_robot: Optional[int] = None
    created_step: int = 0
    updated_step: int = 0
    payload: dict = field(default_factory=dict)

    def is_active(self) -> bool:
        return self.state in (OrderState.NEW, OrderState.ASSIGNED, OrderState.PICKED)