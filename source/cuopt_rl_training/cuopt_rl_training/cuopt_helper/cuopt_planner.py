from typing import List, Dict, Any, Optional
from cuopt.routing import DataModel, Solve, SolverSettings, SolutionStatus
import cudf


class CuOptPlanner:
    def __init__(self, time_limits: float = 5.0, seed: int = 0):
        self.settings = SolverSettings()
        self.settings.time_limit = time_limits
        self.settings.random_seed = seed

    def plan(
        self,
        cost_matrix: List[List[float]],
        orders: List[Dict[str, Any]],
        vehicle_starts: List[int],
        vehicle_returns: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        cost_matrix: NxN (float)
        orders: [{id, location, pickup_idx(optional), delivery_idx(optional)}]
        vehicle_starts/returns: indices into locations (length = num_vehicles)
        """
        if vehicle_returns is None:
            vehicle_returns = list(vehicle_starts)

        n_locations = len(cost_matrix)
        n_orders = len(orders)
        n_vehicles = len(vehicle_starts)

        dm = DataModel(n_locations=n_locations, n_fleet=n_vehicles, n_orders=n_orders)
        dm.add_cost_matrix(cudf.DataFrame(cost_matrix))

        order_locations = [int(order["location"]) for order in orders]
        dm.set_order_locations(cudf.Series(order_locations, dtype="int32"))

        dm.set_vehicle_locations(
            cudf.Series(vehicle_starts, dtype="int32"),
            cudf.Series(vehicle_returns, dtype="int32"),
        )

        # Optional pickup-delivery pairs (only if you actually provide pickup_idx & delivery_idx)
        pickup_idx = [o.get("pickup_idx") for o in orders if "pickup_idx" in o]
        drop_idx = [o.get("delivery_idx") for o in orders if "delivery_idx" in o]
        if pickup_idx and drop_idx and len(pickup_idx) == len(drop_idx):
            dm.set_pickup_delivery_pairs(
                cudf.Series(pickup_idx, dtype="int32"),
                cudf.Series(drop_idx, dtype="int32"),
            )

        solver = Solve(dm, self.settings)
        if solver.get_status() != SolutionStatus.SUCCESS:
            raise RuntimeError(f"cuOpt solve failed: {solver.get_status()} - {solver.get_message()}")

        route_df = solver.get_routes().to_pandas()

        # NOTE: iterrows() yields (index, row)
        rows: List[Dict[str, Any]] = []
        for _, row in route_df.iterrows():
            rows.append(
                {
                    "seq": int(row.route),
                    "vehicle": int(row.truck_id),
                    "location": int(row.location),
                    "eta": float(row.arrival_stamp),
                }
            )
        return rows

    def evaluate_plan(self, plan: List[Any], metrics: Optional[List[str]] = None) -> Dict[str, float]:
        return {}
