from typing import List, Dict, Any, Optional
from cuopt import routing
import cudf


class CuOptPlanner:
    def __init__(self, time_limits: float = 5.0, seed: int = 0):
        self.settings = routing.SolverSettings()
        # notebook dùng set_time_limit()
        self.settings.set_time_limit(time_limits)
        # nếu có seed thì set, tùy version
        if hasattr(self.settings, "random_seed"):
            self.settings.random_seed = seed

    def plan(
        self,
        cost_matrix: List[List[float]],
        orders: List[Dict[str, Any]],
        vehicle_starts: List[int],
        vehicle_returns: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        cost_matrix: NxN
        orders: [{id, location, pickup_idx(optional), delivery_idx(optional)}]
        vehicle_starts/returns: indices into locations
        """
        if vehicle_returns is None:
            vehicle_returns = list(vehicle_starts)

        n_locations = len(cost_matrix)
        n_orders = len(orders)
        n_vehicles = len(vehicle_starts)

        # NOTE: notebook dùng (n_locations, n_vehicles, n_orders)
        dm = routing.DataModel(n_locations, n_vehicles, n_orders)
        dm.add_cost_matrix(cudf.DataFrame(cost_matrix))

        # order locations
        order_locations = cudf.Series([int(o["location"]) for o in orders], dtype="int32")
        dm.set_order_locations(order_locations)

        # vehicle locations
        dm.set_vehicle_locations(
            cudf.Series(vehicle_starts, dtype="int32"),
            cudf.Series(vehicle_returns, dtype="int32"),
        )

        # pickup/delivery pairs nếu có
        pickup_idx = [o.get("pickup_idx") for o in orders if "pickup_idx" in o]
        drop_idx = [o.get("delivery_idx") for o in orders if "delivery_idx" in o]
        if pickup_idx and drop_idx and len(pickup_idx) == len(drop_idx):
            dm.set_pickup_delivery_pairs(
                cudf.Series(pickup_idx, dtype="int32"),
                cudf.Series(drop_idx, dtype="int32"),
            )

        routing_solution = routing.Solve(dm, self.settings)

        status = routing_solution.get_status()
        if int(status) != 0:
            raise RuntimeError(f"cuOpt solve failed: {status} - {routing_solution.get_message()}")

        # notebook dùng routing_solution.route (cudf DataFrame)
        if hasattr(routing_solution, "route"):
            route_df = routing_solution.route
        elif hasattr(routing_solution, "get_route"):
            route_df = routing_solution.get_route()
        else:
            raise RuntimeError("cuOpt solution has no route attribute/method")

        route_pd = route_df.to_pandas()

        rows: List[Dict[str, Any]] = []
        for _, row in route_pd.iterrows():
            rows.append(
                {
                    "seq": int(row["route"]) if "route" in row else int(row["seq"]),
                    "vehicle": int(row["truck_id"]),
                    "location": int(row["location"]),
                    "eta": float(row.get("arrival_stamp", 0.0)),
                }
            )
        return rows

    def evaluate_plan(self, plan: List[Any], metrics: Optional[List[str]] = None) -> Dict[str, float]:
        return {}
