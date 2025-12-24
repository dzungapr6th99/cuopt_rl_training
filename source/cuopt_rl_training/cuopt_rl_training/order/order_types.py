from __future__ import annotations
from enum import IntEnum

class OrderType(IntEnum):
    TYPE1 = 1
    TYPE2 = 2
    TYPE3 = 3

class OrderPriority(IntEnum):
    LOW = 0
    MID = 1
    HIGH = 2

class OrderState(IntEnum):
    NEW = 0
    ASSIGNED = 1
    PICKED = 2
    DELIVERED = 3
    DROPPED = 4