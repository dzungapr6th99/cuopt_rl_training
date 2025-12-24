from __future__ import annotations
import random
from typing import List
from .order import Order
from .order_types import OrderType, OrderPriority

class OrderGenerator:
    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def spawn(
        self,
        start_id: int,
        n_orders: int,
        n_locations: int,
        step: int,
    ) -> List[Order]:
        orders = []
        for k in range(n_orders):
            oid = start_id + k
            otype = self._rng.choice([OrderType.TYPE1, OrderType.TYPE2, OrderType.TYPE3])
            prio = self._rng.choice([OrderPriority.LOW, OrderPriority.MID, OrderPriority.HIGH])
            loc = self._rng.randrange(0, n_locations)
            orders.append(Order(
                id=oid,
                location=loc,
                order_type=otype,
                priority=prio,
                created_step=step,
                updated_step=step,
            ))
        return orders
    def spawn_fixed_type(self, start_id, n_orders, n_locations, step, order_type):
        orders = []
        for k in range(n_orders):
            oid = start_id + k
            prio = self._rng.choice([OrderPriority.LOW, OrderPriority.MID, OrderPriority.HIGH])
            loc = self._rng.randrange(0, n_locations)
            orders.append(Order(
                id=oid,
                location=loc,
                order_type=order_type,
                priority=prio,
                created_step=step,
                updated_step=step,
            ))
        return orders