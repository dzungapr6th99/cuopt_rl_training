from typing import Any, Dict, List, Optional, Sequence, Tuple
import math


class TaskManager:
    """Lightweight wrapper to manage orders, call CuOpt planner, and track waypoint progress."""
    def __init__(
            self,
            planner: Any,
            location_coords: Optional[Dict[int, Tuple[float, float]]] = None,
            waypoint_tolerance: float = 0.3,
    ):
        """
        Args:
            planner: instance of CuOptPlanner (must expose .plan)
            location_coords: optional mapping location_idx -> (x, y) for waypoint distance checks
            waypoint_tolerance: distance (meters) to consider a waypoint reached
        """
        self.planner = planner
        self.location_coords = location_coords or {}
        self.waypoint_tolerance = waypoint_tolerance
        self.cost_matrix: Optional[List[List[float]]] = None
        self.orders: List[Dict[str, Any]] = []
        self.vehicle_starts:Optional[Sequence[int]]=None
        self.vehicle_returns:Optional[Sequence[int]]=None
        self.current_route: List[Dict[str, Any]] = []
        self.current_idx: int = 0

    def set_cost_matrix(self, cost_matrix: List[List[float]]):
        self.cost_matrix = cost_matrix

    def set_orders(self, orders: List[Dict[str, Any]]):
        self.orders = orders
    
    def set_vehicle_locations(
            self,
            vehicle_starts: Sequence[int],
            vehicle_returns: Optional[Sequence[int]] = None,
    ):
        self.vehicle_starts = vehicle_starts
        self.vehicle_returns = vehicle_returns if vehicle_returns is not None else vehicle_starts

    #Planning and waypoint access
    def plan_if_need(self, force:bool = Fasle)-> None:
        """Call CuOpt planner when no activate route of force = True"""
        if not force and self.current_route:
            return
        if self.cost_matrix is None:
            raise ValueError("Cost matrix not set")
        if not self.orders:
            raise ValueError("Orders not set")
        route = self.planner.plan(
            cost_matrix=self.cost_matrix,
            orders=self.orders,
            vehicle_starts=list(self.vehicle_starts),
            vehicle_returns=list(self.vehicle_returns)
        )
        self.current_route = route
        self.current_idx = 0

    def get_current_waypoint(self) -> Optional[Dict[str, Any]]:
        """Get the current waypoint for the active route, or None if complete."""
        if 0<= self.current_idx < len(self.current_route):
            wp = self.current_route[self.current_idx]
            loc_idc = wp.get("location")
            coords = self.location_coords.get(loc_idc)
            if coords:
                wp= {**wp, "pos": coords}
            return wp
        return None
    
    def update_prgress(self, robot_pose_xy: Tuple[float, float],
                       obstabcles_detected: bool = False,
                       timeout:bool = False) -> None:
        """Increase index when reach waypoint; delete route if have obstacle or timeout to replan"""
        wp = self.get_current_waypoint()
        if wp is None:
            return
        if obstabcles_detected or timeout:
            self.current_route = []
            self.current_idx = 0
            return
        target_pos = wp.get("pos")
        if target_pos is None:
            return
        

    @staticmethod
    def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])