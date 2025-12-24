from __future__ import annotations
from typing import List, Dict
from .order import Order
from .order_types import OrderType, OrderPriority, OrderState

class OrderQueue:
    def __init__(self):
        self._orders: Dict[int, Order] = {}
        self._next_id = 0

    def add_orders(self, orders: List[Order]) -> None:
        for o in orders:
            self._orders[o.id] = o
        if orders:
            self._next_id = max(self._next_id, max(o.id for o in orders) + 1)

    def next_id(self) -> int:
        return self._next_id

    def list_active(self) -> List[Order]:
        return [o for o in self._orders.values() if o.is_active()]

    def list_by_type(self, order_type: OrderType) -> List[Order]:
        return [o for o in self.list_active() if o.order_type == order_type]

    def assign_to_robot(self, robot_id: int, orders: List[Order], step: int) -> None:
        for o in orders:
            o.assigned_robot = robot_id
            o.state = OrderState.ASSIGNED
            o.updated_step = step

    def take_by_priority(self, order_type: OrderType, capacity: int) -> List[Order]:
        cand = self.list_by_type(order_type)
        cand.sort(key=lambda o: int(o.priority), reverse=True)
        return cand[:capacity]
