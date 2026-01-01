from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any, Dict, List, Tuple

import omni.usd
import torch
import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
import isaaclab.sim.utils.prims as prim_utils
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane, spawn_from_usd
import time
from cuopt_rl_training.cuopt_helper.cuopt_extract_from_scene import (
    extract_edges_from_stage,
    extract_nodes_from_stage,
)
from cuopt_rl_training.cuopt_helper.cuopt_planner_windows import CuOptPlanner
from cuopt_rl_training.order.order import Order
from cuopt_rl_training.order.order_generator import OrderGenerator
from cuopt_rl_training.order.order_queue import OrderQueue
from cuopt_rl_training.order.order_types import OrderState, OrderType
from .cuopt_rl_training_env_cfg import CuoptRlTrainingEnvCfg

import traceback


class CuoptRlTrainingEnv(DirectRLEnv):
    """Server-training env (centralized agent) + cuOpt routing solve.

    - Action is NOT robot control.
    - Action modifies scenario costs (e.g., congestion penalties).
    - cuOpt is called once per episode. Robots then follow the route.
    - Robots are moved via articulation root pose (no wheel control).
    """

    cfg: CuoptRlTrainingEnvCfg

    def __init__(
        self, cfg: CuoptRlTrainingEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)
        self.viewport_camera_controller.update_view_location(
            eye=(14.0, 10.0, 15.0),
            lookat=(0.0, 0.0, 0.0),
        )

        # cuOpt solver
        self.planner = CuOptPlanner(
            time_limits=float(getattr(self.cfg, "cuopt_time_limit", 5.0)),
            seed=int(getattr(self.cfg, "cuopt_seed", 0)),
        )

        # 3 robots
        self.K = 3
        self._robot_speeds = [1.0, 1.0, 1.0]
        self._rng = random.Random(int(getattr(self.cfg, "order_seed", 0)))
        self._quota_per_robot = [1, 2, 3]
        # Buffers
        self.actions: torch.Tensor = torch.zeros(
            (self.num_envs, int(self.cfg.action_space)), device=self.device
        )
        self.last_cost: torch.Tensor = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.float32
        )
        self.last_status_ok: torch.Tensor = torch.ones(
            (self.num_envs,), device=self.device, dtype=torch.bool
        )

        # Visual robot state per env
        self._robot_xy = torch.zeros(
            (self.num_envs, self.K, 2), device=self.device, dtype=torch.float32
        )
        self._robot_yaw = torch.zeros(
            (self.num_envs, self.K), device=self.device, dtype=torch.float32
        )

        # Route + cargo state
        self._route_locs: List[List[List[int]]] = [
            [[] for _ in range(self.K)] for _ in range(self.num_envs)
        ]
        self._route_orders: List[List[List[Order | None]]] = [
            [[] for _ in range(self.K)] for _ in range(self.num_envs)
        ]
        self._robot_cargo: List[List[List[Order]]] = [
            [[] for _ in range(self.K)] for _ in range(self.num_envs)
        ]
        self._last_node: List[List[int | None]] = [
            [None for _ in range(self.K)] for _ in range(self.num_envs)
        ]
        self._planned: torch.Tensor = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.bool
        )

        # Scenario state per env (python lists ok for now)
        self._cost_matrices: List[List[List[float]]] = [None] * self.num_envs  # type: ignore
        self._orders: List[List[Order]] = [None] * self.num_envs  # type: ignore
        self._vehicle_starts: List[List[int]] = [None] * self.num_envs  # type: ignore
        self._vehicle_returns: List[List[int]] = [None] * self.num_envs  # type: ignore
        self._blocked_edges: List[List[Tuple[int, int]]] = [
            [] for _ in range(self.num_envs)
        ]

        # Location -> (x,y) lookup per env
        self._loc_xy: List[List[Tuple[float, float]]] = [None] * self.num_envs  # type: ignore

        # Orders
        self._order_gen = OrderGenerator(seed=int(getattr(self.cfg, "order_seed", 0)))
        self._order_queue_type1 = OrderQueue()
        self._order_queue_type2 = OrderQueue()
        self._order_queue_type3 = OrderQueue()
        # ===== Sequential RL buffers =====
        self._replan_interval = int(getattr(self.cfg, "replan_interval", 12))
        self._steps_since_plan = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)

        self._last_plan_cost = torch.zeros((self.num_envs,), device=self.device)
        self._delivered_count = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self._prev_delivered_count = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)

        # edge congestion per env
        self._edge_congestion: List[Dict[Tuple[int, int], float]] = [
            {} for _ in range(self.num_envs)
        ]

    # ----------------------- Scene -----------------------

    def _setup_scene(self):
        #spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        spawn_from_usd(
            prim_path="/World/envs/env_0/Warehouse",
            cfg=self.cfg.warehouse_cfg,
        )

        # Parent prim for robot assets (required by regex prim paths).
        prim_utils.create_prim("/World/envs/env_0/Robots", prim_type="Xform")

        # Spawn visual robots (no articulation / physics).
        spawn_from_usd(
            prim_path="/World/envs/env_0/Robots/Robot1",
            cfg=sim_utils.UsdFileCfg(usd_path=self.cfg.robot1_cfg.spawn.usd_path),
        )
        spawn_from_usd(
            prim_path="/World/envs/env_0/Robots/Robot2",
            cfg=sim_utils.UsdFileCfg(usd_path=self.cfg.robot2_cfg.spawn.usd_path),
        )
        spawn_from_usd(
            prim_path="/World/envs/env_0/Robots/Robot3",
            cfg=sim_utils.UsdFileCfg(usd_path=self.cfg.robot3_cfg.spawn.usd_path),
        )

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # Light (optional)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _spawn_fixed_orders_for_episode(self, n_locations: int, step: int):
        q1 = self._order_queue_type1
        q2 = self._order_queue_type2
        q3 = self._order_queue_type3

        orders1 = self._order_gen.spawn_fixed_type(
            start_id=q1.next_id(),
            n_orders=10,
            n_locations=n_locations,
            step=step,
            order_type=OrderType.TYPE1,
        )
        q1.add_orders(orders1)

        orders2 = self._order_gen.spawn_fixed_type(
            start_id=q2.next_id(),
            n_orders=20,
            n_locations=n_locations,
            step=step,
            order_type=OrderType.TYPE2,
        )
        q2.add_orders(orders2)

        orders3 = self._order_gen.spawn_fixed_type(
            start_id=q3.next_id(),
            n_orders=30,
            n_locations=n_locations,
            step=step,
            order_type=OrderType.TYPE3,
        )
        q3.add_orders(orders3)

        return orders1, orders2, orders3

    # ----------------------- RL loop hooks -----------------------

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions is None:
            return

        act = actions.clone()
        clip = float(getattr(self.cfg, "action_clip", 1.0))
        act = torch.clamp(act, -clip, clip)

        if act.ndim == 1:
            act = act.unsqueeze(-1)
        if act.shape[-1] != int(self.cfg.action_space):
            raise RuntimeError(
                f"Expected actions dim {int(self.cfg.action_space)}, got {act.shape[-1]}"
            )

        self.actions = act

    def _apply_action(self) -> None:
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        # initial solve if not planned
        init_ids = (~self._planned).nonzero(as_tuple=False).squeeze(-1)
        if init_ids.numel() > 0:
            self._plan_routes_for_env_ids(init_ids.tolist())
            self._planned[init_ids] = True

        # replan when idle + remaining
        replan_ids = []
        for env_id in env_ids.tolist():
            if self._orders[env_id] is None:
                continue
            remaining = any(o.state != OrderState.DELIVERED for o in self._orders[env_id])
            any_idle = any(len(self._route_locs[env_id][rid]) == 0 for rid in range(self.K))
            if remaining and any_idle:
                replan_ids.append(env_id)

        if replan_ids:
            self._plan_routes_for_env_ids(replan_ids)

        self._advance_robots(env_ids)


    def _get_observations(self) -> dict:
        n_vehicles = torch.tensor(
            [len(vs) if vs is not None else 0 for vs in self._vehicle_starts],
            device=self.device,
            dtype=torch.float32,
        )
        n_orders = torch.tensor(
            [len(o) if o is not None else 0 for o in self._orders],
            device=self.device,
            dtype=torch.float32,
        )
        blocked_cnt = torch.tensor(
            [len(be) for be in self._blocked_edges],
            device=self.device,
            dtype=torch.float32,
        )
        status_ok = self.last_status_ok.float()

        base = torch.stack(
            [n_vehicles, n_orders, blocked_cnt, self.last_cost, status_ok], dim=-1
        )
        obs = torch.cat([base, self.actions], dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # 1) improvement reward (after replan)
        improvement = self._last_plan_cost - self.last_cost
        r = improvement

        # 2) delivered orders reward
        delta_delivered = self._delivered_count - self._prev_delivered_count
        r += delta_delivered.float() * 10.0
        self._prev_delivered_count[:] = self._delivered_count

        # 3) time penalty (avoid stalling)
        r -= 0.01

        # 4) solver fail penalty
        fail_penalty = float(getattr(self.cfg, "solve_fail_penalty", 1000.0))
        r = torch.where(self.last_status_ok, r, r - fail_penalty)

        return r

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminate_on_fail = bool(getattr(self.cfg, "terminate_on_solve_fail", False))
        terminated = (
            (~self.last_status_ok)
            if terminate_on_fail
            else torch.zeros_like(time_out, dtype=torch.bool)
        )
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        super()._reset_idx(env_ids)

        # Build scenarios + reset visual robot poses for these envs
        for eid in env_ids:
            e = int(eid)
            self._build_scenario(e)
            self._reset_visual_robots(e)

            self._route_locs[e] = [[] for _ in range(self.K)]
            self._route_orders[e] = [[] for _ in range(self.K)]
            self._robot_cargo[e] = [[] for _ in range(self.K)]
            self._last_node[e] = [None for _ in range(self.K)]
            self._planned[e] = False

        # Reset per-env solve buffers
        self.last_cost[env_ids] = 0.0
        self.last_status_ok[env_ids] = True

        # Optional initial solve
        if bool(getattr(self.cfg, "solve_on_reset", True)):
            self._plan_routes_for_env_ids(env_ids)
            self._planned[env_ids] = True

    def _sample_orders_for_cuopt(self):
        orders = []
        orders.extend(self._order_queue_type1.list_active())
        orders.extend(self._order_queue_type2.list_active())
        orders.extend(self._order_queue_type3.list_active())
        return self._orders_to_cuopt(orders)

    def _extract_graph_from_scene(self, env_id: int):
        stage = omni.usd.get_context().get_stage()
        root_path = f"/World/envs/env_{env_id}/Warehouse/WaypointGraph"
        nodes_xyz = extract_nodes_from_stage(
            stage=stage,
            root_prim_path=f"{root_path}/Nodes",
            node_prefix="Node_",
        )
        edges = extract_edges_from_stage(
            stage=stage,
            n_nodes=len(nodes_xyz),
            root_prim_path=f"{root_path}/Edges",
            edge_prefix="Edge_",
            bidirectional=True,
        )
        return nodes_xyz, edges

    def _build_cost_matrix_from_edges(
        self,
        nodes_xyz: list[tuple[float, float, float]],
        edges: list[tuple[int, int]],
    ) -> list[list[float]]:
        n = len(nodes_xyz)
        blocked = float(getattr(self.cfg, "blocked_edge_cost", 1e6))
        cm = [[blocked for _ in range(n)] for _ in range(n)]
        for i in range(n):
            cm[i][i] = 0.0
        for u, v in edges:
            x0, y0, _ = nodes_xyz[u]
            x1, y1, _ = nodes_xyz[v]
            dist = math.hypot(x1 - x0, y1 - y0)
            cm[u][v] = dist
        return cm

    def _update_edge_congestion(self, env_ids: Sequence[int]) -> None:
        for eid in env_ids:
            env = int(eid)
            self._edge_congestion[env].clear()

            for rid in range(self.K):
                prev = self._last_node[env][rid]
                route = self._route_locs[env][rid]
                if prev is None or not route:
                    continue

                nxt = route[0]
                self._edge_congestion[env][(prev, nxt)] = (
                    self._edge_congestion[env].get((prev, nxt), 0.0) + 1.0
                )

            # normalize congestion
            for k in self._edge_congestion[env]:
                self._edge_congestion[env][k] = min(1.0, self._edge_congestion[env][k] / self.K)
    # ----------------------- Scenario -----------------------

    def _build_scenario(self, env_id: int) -> None:
        nodes_xyz, edges = self._extract_graph_from_scene(env_id)
        cm = self._build_cost_matrix_from_edges(nodes_xyz, edges)
        n_locations = len(nodes_xyz)
        step = int(self.common_step_counter) if hasattr(self, "common_step_counter") else 0

        orders1, orders2, orders3 = self._spawn_fixed_orders_for_episode(n_locations, step)

        # dùng XY từ scene
        self._loc_xy[env_id] = [(x, y) for (x, y, _z) in nodes_xyz]

        starts = [self._rng.randrange(0, n_locations) for _ in range(self.K)]
        self._blocked_edges[env_id] = []
        self._cost_matrices[env_id] = cm
        self._orders[env_id] = orders1 + orders2 + orders3
        self._vehicle_starts[env_id] = starts
        self._vehicle_returns[env_id] = list(starts)

    # ----------------------- Visual robots -----------------------

    def _reset_visual_robots(self, env_id: int) -> None:
        nloc = len(self._loc_xy[env_id])
        starts = [self._rng.randrange(0, nloc) for _ in range(self.K)]

        loc_xy = self._loc_xy[env_id]
        for i, loc in enumerate(starts):
            x, y = loc_xy[loc]
            self._robot_xy[env_id, i, 0] = x
            self._robot_xy[env_id, i, 1] = y
            self._robot_yaw[env_id, i] = 0.0
            prim_path = f"/World/envs/env_{env_id}/Robots/Robot{i+1}"
            self._set_robot_pose_xy(prim_path, x, y, z=0.3, yaw_rad=0.0)

    def _set_robot_pose_xy(
        self,
        prim_path: str,
        x: float,
        y: float,
        z: float = 0.0,
        yaw_rad: float = 0.0,
    ) -> None:
        from pxr import Gf, UsdGeom

        prim = prim_utils.get_prim_at_path(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim not found or invalid: {prim_path}")

        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            raise ValueError(f"Prim is not Xformable: {prim_path}")

        translate_op = None
        rotate_op = None
        orient_op = None
        scale_op = None
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
            elif op.GetOpType() == UsdGeom.XformOp.TypeRotateZ:
                rotate_op = op
            elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                orient_op = op
            elif op.GetOpType() == UsdGeom.XformOp.TypeScale:
                scale_op = op

        if translate_op is None:
            translate_op = xformable.AddTranslateOp()
        if rotate_op is None and orient_op is None:
            rotate_op = xformable.AddRotateZOp()

        translate_op.Set((float(x), float(y), float(z)))
        yaw_deg = float(yaw_rad * 180.0 / math.pi)
        if rotate_op is not None:
            rotate_op.Set(yaw_deg)
        elif orient_op is not None:
            rot = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), yaw_deg)
            orient_op.Set(rot.GetQuat())
        if scale_op is not None:
            scale_op.Set((1.0, 1.0, 1.0))

    # ----------------------- Solve -----------------------

    def _apply_action_to_cost_matrix(
        self, env_id: int, base_cost_matrix: List[List[float]]
    ) -> List[List[float]]:
        cm = [row[:] for row in base_cost_matrix]

        # action[0] = congestion penalty scale
        a0 = float(self.actions[env_id, 0].item()) if self.actions.numel() > 0 else 0.0
        scale = max(0.0, a0)

        for (u, v), congestion in self._edge_congestion[env_id].items():
            if 0 <= u < len(cm) and 0 <= v < len(cm):
                cm[u][v] *= (1.0 + scale * congestion)

        return cm


    def _build_routes_from_plan(self, plan_rows: List[Dict[str, Any]], n_vehicles: int) -> List[List[int]]:
        routes = [[] for _ in range(n_vehicles)]
        by_v = [[] for _ in range(n_vehicles)]
        for r in plan_rows:
            v = int(r["vehicle"])
            if 0 <= v < n_vehicles:
                by_v[v].append(r)

        for v in range(n_vehicles):
            rows = sorted(by_v[v], key=lambda x: x["seq"])
            locs = [int(r["location"]) for r in rows]
            # Remove consecutive duplicates to avoid stuck steps.
            compacted: List[int] = []
            for loc in locs:
                if not compacted or loc != compacted[-1]:
                    compacted.append(loc)
            routes[v] = compacted
        return routes

    def _assign_orders_to_routes(self, env_id: int, routes: List[List[int]]) -> None:
        loc_map: Dict[int, List[Order]] = {}
        for o in self._orders[env_id]:
            loc_map.setdefault(int(o.location), []).append(o)

        for rid in range(self.K):
            self._robot_cargo[env_id][rid] = []
            self._route_orders[env_id][rid] = []
            for loc in routes[rid]:
                assigned = None
                if loc in loc_map and loc_map[loc]:
                    assigned = loc_map[loc].pop(0)
                    assigned.assigned_robot = rid
                    assigned.state = OrderState.ASSIGNED
                    self._robot_cargo[env_id][rid].append(assigned)
                self._route_orders[env_id][rid].append(assigned)

    def _robot_priority(self, env_id: int, robot_id: int) -> int:
        cargo = self._robot_cargo[env_id][robot_id]
        if not cargo:
            return 0
        return max(int(o.priority) for o in cargo)

    def _resolve_next_node_conflicts(self, env_id: int, next_nodes: Dict[int, int]) -> set[int]:
        claims: Dict[int, List[int]] = {}
        for rid, loc in next_nodes.items():
            claims.setdefault(loc, []).append(rid)

        allowed = set(next_nodes.keys())
        for loc, rids in claims.items():
            if len(rids) > 1:
                winner = max(rids, key=lambda r: self._robot_priority(env_id, r))
                for r in rids:
                    if r != winner:
                        allowed.discard(r)
                        print(
                            f"[yield] env {env_id} robot {r} waits for robot {winner} at node {loc}"
                        )
        return allowed

    def _plan_routes_for_env_ids(self, env_ids: Sequence[int]) -> None:

        def _current_node(env_id: int, rid: int) -> int:
            node = self._last_node[env_id][rid]
            if node is not None:
                return int(node)
            # fallback nearest node by XY
            rx = float(self._robot_xy[env_id, rid, 0].item())
            ry = float(self._robot_xy[env_id, rid, 1].item())
            nodes = self._loc_xy[env_id]
            return min(
                range(len(nodes)),
                key=lambda i: (nodes[i][0] - rx) ** 2 + (nodes[i][1] - ry) ** 2,
            )

        for eid in env_ids:
            env_id = int(eid)

            cm0 = self._cost_matrices[env_id]
            if cm0 is None:
                continue

            # action -> dynamic costs
            cm = self._apply_action_to_cost_matrix(env_id, cm0)

            starts = [_current_node(env_id, rid) for rid in range(self.K)]
            rets = list(starts)

            ok = True
            eta_all = [0.0] * self.K

            for rid in range(self.K):
                # ✅ chỉ plan khi robot đang rảnh (route rỗng)
                if self._route_locs[env_id][rid]:
                    continue

                quota = self._quota_per_robot[rid]
                batch_orders = self._pick_batch_orders_for_robot(env_id, rid, quota)

                if not batch_orders:
                    # không còn order cho robot này
                    continue

                cu_orders = self._orders_to_cuopt(batch_orders)

                try:
                    plan_rows = self.planner.plan(
                        cost_matrix=cm,
                        orders=cu_orders,
                        vehicle_starts=[starts[rid]],
                        vehicle_returns=[rets[rid]],
                    )

                    # ETA
                    if plan_rows:
                        etas = []
                        for r in plan_rows:
                            try:
                                etas.append(float(r.get("eta", 0.0)))
                            except Exception:
                                etas.append(0.0)
                        eta_all[rid] = max(etas) if etas else 0.0

                    # route (1 vehicle)
                    rts = self._build_routes_from_plan(plan_rows, n_vehicles=1)
                    route = rts[0] if rts else []

                    # bỏ start node nếu trùng (advance_robots cũng đã skip, nhưng làm luôn cho sạch)
                    cur = starts[rid]
                    while route and int(route[0]) == int(cur):
                        route.pop(0)

                    self._route_locs[env_id][rid] = route

                    # gắn order theo node visit (cùng location thì pop)
                    loc_map = {}
                    for o in batch_orders:
                        loc_map.setdefault(int(o.location), []).append(o)

                    self._route_orders[env_id][rid] = []
                    for loc in route:
                        loc = int(loc)
                        assigned = None
                        if loc in loc_map and loc_map[loc]:
                            assigned = loc_map[loc].pop(0)
                        self._route_orders[env_id][rid].append(assigned)

                    print(f"[batch-plan] env {env_id} rid {rid} quota={quota} "
                        f"picked={len(batch_orders)} start={starts[rid]} "
                        f"route_len={len(route)} eta={eta_all[rid]:.2f}")

                    # ✅ sleep để tránh server cuOpt race /solution 500
                    time.sleep(0.3)

                except Exception:
                    ok = False
                    self.last_status_ok[env_id] = False
                    self.last_cost[env_id] = float(getattr(self.cfg, "solve_fail_cost", 1e6))
                    traceback.print_exc()
                    break  # fail thì dừng luôn cho khỏi bắn request tiếp

            # global cost = makespan (tuỳ bạn có dùng reward theo cost không)
            self.last_cost[env_id] = float(max(eta_all)) if any(eta_all) else self.last_cost[env_id]
            self.last_status_ok[env_id] = bool(ok)

    def _get_robot_current_node(self, env_id: int, rid: int) -> int | None:
        # ưu tiên last_node (đã tới gần nhất)
        if self._last_node[env_id][rid] is not None:
            return int(self._last_node[env_id][rid])

        # fallback: tìm node gần nhất theo XY hiện tại (dùng cho step đầu)
        rx = float(self._robot_xy[env_id, rid, 0].item())
        ry = float(self._robot_xy[env_id, rid, 1].item())
        nodes = self._loc_xy[env_id]
        if not nodes:
            return None
        return min(range(len(nodes)), key=lambda i: (nodes[i][0] - rx) ** 2 + (nodes[i][1] - ry) ** 2)
    def _advance_robots(self, env_ids: Sequence[int]) -> None:
        """
        Edge-per-step motion (debug-friendly):
        - Each RL step moves a robot across exactly ONE edge: current_node -> next_node.
        - If route starts with the current node (e.g., [cur, ...]), we SKIP it (pop) so first move is cur->route[0].
        - Conflict resolution is node-based: if multiple robots claim the same next node, only one moves.
        """
        for eid in env_ids:
            env_id = int(eid)
            loc_xy = self._loc_xy[env_id]

            # ---- 1) Pre-clean routes: skip leading nodes that equal current node ----
            for rid in range(self.K):
                route = self._route_locs[env_id][rid]
                if not route:
                    continue

                cur = self._get_robot_current_node(env_id, rid)  # you added this helper earlier
                if cur is None:
                    continue

                # If route begins with current node (common when plan includes start), skip it.
                while route and int(route[0]) == int(cur):
                    route.pop(0)
                    if self._route_orders[env_id][rid]:
                        self._route_orders[env_id][rid].pop(0)

            # ---- 2) Build next-node claims after cleanup ----
            next_nodes: Dict[int, int] = {}
            for rid in range(self.K):
                route = self._route_locs[env_id][rid]
                if route:
                    next_nodes[rid] = int(route[0])

            allowed = self._resolve_next_node_conflicts(env_id, next_nodes)

            # ---- 3) Debug prints (idle / wait / move) ----
            for rid in range(self.K):
                route = self._route_locs[env_id][rid]
                cur = self._get_robot_current_node(env_id, rid)

                if not route:
                    print(f"[edge] env {env_id} robot {rid}: idle (no route)")
                    continue

                nxt = int(route[0])
                state = "move" if rid in allowed else "wait"
                print(f"[edge] env {env_id} robot {rid}: {cur} -> {nxt} ({state})")

            # ---- 4) Move exactly ONE edge for allowed robots ----
            for rid in range(self.K):
                if rid not in allowed:
                    continue
                if rid not in next_nodes:
                    continue

                route = self._route_locs[env_id][rid]
                nxt = int(route[0])

                cur = self._get_robot_current_node(env_id, rid)

                # Jump to next node immediately
                tx, ty = loc_xy[nxt]
                self._robot_xy[env_id, rid, 0] = float(tx)
                self._robot_xy[env_id, rid, 1] = float(ty)

                # Yaw: face from cur -> nxt (if cur known)
                yaw = 0.0
                if cur is not None and 0 <= int(cur) < len(loc_xy):
                    cx, cy = loc_xy[int(cur)]
                    yaw = math.atan2(float(ty) - float(cy), float(tx) - float(cx))

                # If an order is attached to this visited node, mark delivered
                order = self._route_orders[env_id][rid][0] if self._route_orders[env_id][rid] else None
                if order is not None:
                    order.state = OrderState.DELIVERED
                    if order in self._robot_cargo[env_id][rid]:
                        self._robot_cargo[env_id][rid].remove(order)
                    # (optional) if you track delivered count:
                    # self._delivered_count[env_id] += 1

                # Pop one node => traversed one edge
                self._route_locs[env_id][rid].pop(0)
                if self._route_orders[env_id][rid]:
                    self._route_orders[env_id][rid].pop(0)

                # Update last_node to the node we just reached
                self._last_node[env_id][rid] = int(nxt)

                # Apply pose to USD prim
                prim_path = f"/World/envs/env_{env_id}/Robots/Robot{rid+1}"
                self._set_robot_pose_xy(prim_path, float(tx), float(ty), z=0.0, yaw_rad=float(yaw))


    def _orders_to_cuopt(self, orders):
        return [{"id": o.id, "location": o.location} for o in orders]
    def _pick_batch_orders_for_robot(self, env_id: int, rid: int, k: int) -> List[Order]:
        # lấy orders hợp lệ (đúng type + chưa delivered + chưa assigned)
        type_map = {0: OrderType.TYPE1, 1: OrderType.TYPE2, 2: OrderType.TYPE3}
        want_type = type_map[rid]

        candidates = [
            o for o in self._orders[env_id]
            if o.state != OrderState.DELIVERED
            and o.state != OrderState.ASSIGNED
            and getattr(o, "order_type", None) == want_type
        ]

        # pick k cái đầu (hoặc theo priority nếu bạn có)
        batch = candidates[:k]

        # mark assigned ngay để lần sau không pick lại
        for o in batch:
            o.state = OrderState.ASSIGNED
            o.assigned_robot = rid

        return batch