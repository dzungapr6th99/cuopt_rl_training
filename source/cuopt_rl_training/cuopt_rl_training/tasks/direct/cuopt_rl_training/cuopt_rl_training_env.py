from __future__ import annotations

import math

from sympy import root
from cuopt_rl_training.order.order_generator import OrderGenerator
import torch
from collections.abc import Sequence
from typing import Any, Dict, List, Tuple
import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
import isaaclab.sim.utils.prims as prim_utils
from isaaclab.assets import Articulation, ArticulationData
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab.sim.spawners.from_files import (
    GroundPlaneCfg,
    spawn_ground_plane,
    spawn_from_usd,
)
from cuopt_rl_training.cuopt_helper.cuopt_planner import CuOptPlanner
from .cuopt_rl_training_env_cfg import CuoptRlTrainingEnvCfg
import omni.usd
from cuopt_rl_training.cuopt_helper.cuopt_extract_from_scene import (
    extract_nodes_from_stage,
    extract_edges_from_stage,
)
import traceback
from cuopt_rl_training.order.order_generator import OrderGenerator
from cuopt_rl_training.order.order_queue import OrderQueue
from cuopt_rl_training.order.order_types import OrderType, OrderPriority, OrderState
from cuopt_rl_training.robots.robot_specs import spawn_all_robots_base, ROBOT_SPECS
from typing import Sequence
import random

from pxr import UsdGeom, Gf


class CuoptRlTrainingEnv(DirectRLEnv):
    """Server-training env (centralized agent) + cuOpt routing solve.

    - Action is NOT robot control.
    - Action modifies scenario costs (e.g., congestion penalties).
    - cuOpt is called each step (or on reset), producing last_cost -> reward/obs.
    - Robots are VISUAL ONLY (USD references). No articulation, no wheel joints.
    """

    cfg: CuoptRlTrainingEnvCfg

    def __init__(
        self, cfg: CuoptRlTrainingEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)
        self.viewport_camera_controller.update_view_location(
            eye=(20.0, 20.0, 35.0),
            lookat=(20.0, 20.0, 0.0),
        )

        # cuOpt solver
        self.planner = CuOptPlanner(
            time_limits=float(getattr(self.cfg, "cuopt_time_limit", 5.0)),
            seed=int(getattr(self.cfg, "cuopt_seed", 0)),
        )
        # 3 visual robots
        self.K = 3
        self._rng = random.Random(int(getattr(self.cfg, "order_seed", 0)))
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
        self._xform_ops_cache: dict[str, tuple] = {}
        # Visual robot state per env
        self._robot_xy = torch.zeros(
            (self.num_envs, self.K, 2), device=self.device, dtype=torch.float32
        )
        self._robot_yaw = torch.zeros(
            (self.num_envs, self.K), device=self.device, dtype=torch.float32
        )
        self._route_locs: List[List[List[int]]] = [[[] for _ in range(self.K)] for _ in range(self.num_envs)]
        self._route_idx: List[List[int]] = [[0 for _ in range(self.K)] for _ in range(self.num_envs)]
        # Scenario state per env (python lists ok for now)
        self._cost_matrices: List[List[List[float]]] = [None] * self.num_envs  # type: ignore
        self._orders: List[List[Dict[str, Any]]] = [None] * self.num_envs  # type: ignore
        self._vehicle_starts: List[List[int]] = [None] * self.num_envs  # type: ignore
        self._vehicle_returns: List[List[int]] = [None] * self.num_envs  # type: ignore
        self._blocked_edges: List[List[Tuple[int, int]]] = [
            [] for _ in range(self.num_envs)
        ]
        self._rl_step_counter = 0
        self._cuopt_interval_rl = max(
            1, int(round((1 / float(self.cfg.sim.dt)) / int(self.cfg.decimation) / 2))
        )
        # Location -> (x,y) lookup per env
        self._loc_xy: List[List[Tuple[float, float]]] = [None] * self.num_envs  # type: ignore
        self._order_gen = OrderGenerator(seed=int(getattr(self.cfg, "order_seed", 0)))
        self._order_queue_type1 = OrderQueue()
        self._order_queue_type2 = OrderQueue()
        self._order_queue_type3 = OrderQueue()

    # ----------------------- Scene -----------------------

    def _setup_scene(self):
        # Load warehouse USD in source env_0 so clone works

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        spawn_from_usd(
            prim_path="/World/envs/env_0/Warehouse",
            cfg=self.cfg.warehouse_cfg,
        )
        # Create parent prim for robot visuals (must exist before spawn)
        prim_utils.create_prim("/World/envs/env_0/Robots", prim_type="Xform")

        self.robot1 = Articulation(self.cfg.robot1_cfg)
        self.robot2 = Articulation(self.cfg.robot2_cfg)
        self.robot3 = Articulation(self.cfg.robot3_cfg)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot1"] = self.robot1
        self.scene.articulations["robot2"] = self.robot2
        self.scene.articulations["robot3"] = self.robot3

        self.robots = {
            "robot1": self.robot1,
            "robot2": self.robot2,
            "robot3": self.robot3,
        }
        for i in range(len(ROBOT_SPECS)):
            prim_path = f"/World/envs/env_0/Robots/Robot{i+1}"
            self._init_robot_xform_ops(prim_path)
        # Clone/replicate envs
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # Light (optional)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _init_robot_xform_ops(self, prim_path: str) -> None:
        if not hasattr(self, "_xform_ops_cache"):
            self._xform_ops_cache = {}
        prim = prim_utils.get_prim_at_path(prim_path)
        if not prim or not prim.IsValid():
            return

        xform = UsdGeom.Xformable(prim)
        # Clear once to make ops compatible.
        xform.ClearXformOpOrder()
        t_op = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        r_op = xform.AddRotateXYZOp(UsdGeom.XformOp.PrecisionDouble)
        s_op = xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
        self._xform_ops_cache[prim_path] = (t_op, r_op, s_op)

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
            n_orders=10,
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
        # No robot control here. Solve cuOpt so rewards/obs use updated last_cost.
        self._rl_step_counter += 1
        if self._rl_step_counter % self._cuopt_interval_rl != 0:
            return
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._solve_for_env_ids(env_ids)

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
        fail_penalty = float(getattr(self.cfg, "solve_fail_penalty", 1000.0))
        r = -self.last_cost
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

        # Reset per-env solve buffers
        self.last_cost[env_ids] = 0.0
        self.last_status_ok[env_ids] = True

        # Optional initial solve
        if bool(getattr(self.cfg, "solve_on_reset", True)):
            self._solve_for_env_ids(env_ids)

    def _sample_orders_for_cuopt(self):
        pools = [
            (self._order_queue_type1, 1),
            (self._order_queue_type2, 2),
            (self._order_queue_type3, 3),
        ]
        picked = []
        for q, cap in pools:
            cand = q.list_active()
            self._rng.shuffle(cand)
            picked.extend(cand[:cap])
        return self._orders_to_cuopt(picked)

    def _extract_graph_from_scene(self, env_id: int):
        stage = omni.usd.get_context().get_stage()
        root_path = f"/World/envs/env_{env_id}/Warehouse/Warehouse/Transportation/WaypointGraph"  # đổi nếu node/edge nằm nơi khác
        nodes_xyz = extract_nodes_from_stage(
            stage=stage,
            root_prim_path=f"{root_path}/Nodes",
            node_prefix="Node_",
        )
        print(nodes_xyz[:5])        
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

    # ----------------------- Scenario -----------------------

    def _build_scenario(self, env_id: int) -> None:
        nodes_xyz, edges = self._extract_graph_from_scene(env_id)
        cm = self._build_cost_matrix_from_edges(nodes_xyz, edges)
        n_locations = len(nodes_xyz)
        step = (
            int(self.common_step_counter) if hasattr(self, "common_step_counter") else 0
        )

        orders1, orders2, orders3 = self._spawn_fixed_orders_for_episode(
            n_locations, step
        )
        self._orders_type1, self._orders_type2, self._orders_type3 = (
            orders1,
            orders2,
            orders3,
        )

        # dùng XY từ scene
        self._loc_xy[env_id] = [(x, y) for (x, y, _z) in nodes_xyz]

        starts = [self._rng.randrange(0, n_locations) for _ in range(3)]
        self._blocked_edges[env_id] = []
        self._cost_matrices[env_id] = cm
        self._orders[env_id] = orders1 + orders2 + orders3
        self._vehicle_starts[env_id] = starts
        self._vehicle_returns[env_id] = list(starts)

    # ----------------------- Visual robots -----------------------

    def _reset_visual_robots(self, env_id: int) -> None:
        """Place 3 visual robots at start locations (0,5,10)."""
        nloc = len(self._loc_xy[env_id])
        starts = [self._rng.randrange(0, nloc) for _ in range(self.K)]

        loc_xy = self._loc_xy[env_id]
        for i, loc in enumerate(starts):
            x, y = loc_xy[loc]
            self._robot_xy[env_id, i, 0] = x
            self._robot_xy[env_id, i, 1] = y
            self._robot_yaw[env_id, i] = 0.0

            prim_path = f"/World/envs/env_{env_id}/Robots/Robot{i+1}"
            self._set_robot_pose_xy(f"robot{i+1}", x, y, z=0.01, yaw_rad=0.0)

    def _set_robot_pose_xy(
        self,
        robot_id: str,
        x: float,
        y: float,
        z: float = 0.0,
        yaw_rad: float = 0.0,
    ) -> None:
        robot = self.robots[robot_id]
        root_state = robot.data.default_root_state.clone()
        root_state[:, 0] = x
        root_state[:, 1] = y
        root_state[:, 2] = z

        yaw = torch.tensor([yaw_rad], device=self.device)
        zero = torch.tensor([0.0], device=self.device)
        quat = quat_from_euler_xyz(zero, zero, yaw)
        root_state[:, 3:7] = quat

        robot.write_root_pose_to_sim(root_state[:, :7])

    # ----------------------- Solve -----------------------

    def _apply_action_to_cost_matrix(
        self, env_id: int, base_cost_matrix: List[List[float]]
    ) -> List[List[float]]:
        cm = [row[:] for row in base_cost_matrix]
        blocked_value = float(getattr(self.cfg, "blocked_edge_cost", 1e6))

        a0 = float(self.actions[env_id, 0].item()) if self.actions.numel() > 0 else 0.0
        penalty_scale = max(0.0, a0)

        for u, v in self._blocked_edges[env_id]:
            if 0 <= u < len(cm) and 0 <= v < len(cm):
                cm[u][v] = min(blocked_value, cm[u][v] + penalty_scale * blocked_value)
        return cm

    def _extract_next_targets(
        self, plan_rows: List[Dict[str, Any]], n_vehicles: int
    ) -> List[int]:
        targets = [0] * n_vehicles
        by_v = [[] for _ in range(n_vehicles)]
        for r in plan_rows:
            v = int(r["vehicle"])
            if 0 <= v < n_vehicles:
                by_v[v].append(r)

        for v in range(n_vehicles):
            rows = sorted(by_v[v], key=lambda x: x["seq"])
            if len(rows) >= 2:
                targets[v] = int(rows[1]["location"])
            elif len(rows) == 1:
                targets[v] = int(rows[0]["location"])
            else:
                targets[v] = 0
        return targets
    def _build_routes_from_plan(self, plan_rows: List[Dict[str, Any]], n_vehicles: int) -> List[List[int]]:
        routes = [[] for _ in range(n_vehicles)]
        by_v = [[] for _ in range(n_vehicles)]
        for r in plan_rows:
            v = int(r["vehicle"])
            if 0 <= v < n_vehicles:
                by_v[v].append(r)

        for v in range(n_vehicles):
            rows = sorted(by_v[v], key=lambda x: x["seq"])
            routes[v] = [int(r["location"]) for r in rows]
        return routes
    def _solve_for_env_ids(self, env_ids: Sequence[int]) -> None:
        dt = float(getattr(self.cfg.sim, "dt", 1 / 24)) * int(
            getattr(self.cfg, "decimation", 1)
        )

        for eid in env_ids:
            env_id = int(eid)
            cm0 = self._cost_matrices[env_id]
            starts = self._vehicle_starts[env_id]
            rets = self._vehicle_returns[env_id]

            orders = self._sample_orders_for_cuopt()
            cm = self._apply_action_to_cost_matrix(env_id, cm0)

            try:
                plan_rows = self.planner.plan(
                    cost_matrix=cm,
                    orders=orders,
                    vehicle_starts=starts,
                    vehicle_returns=rets,
                )
                print("[solve] plan_rows:", len(plan_rows), plan_rows[:3])

                cost = (
                    0.0
                    if len(plan_rows) == 0
                    else max(float(r["eta"]) for r in plan_rows)
                )
                self.last_cost[env_id] = float(cost)
                self.last_status_ok[env_id] = True

                n_veh = len(starts) if starts is not None else self.K
                routes = self._build_routes_from_plan(plan_rows, n_vehicles=n_veh)
                for v, locs in enumerate(routes):
                    print(f"[route] v{v}:", locs)
                self._route_locs[env_id] = routes
                self._route_idx[env_id] = [0 for _ in range(self.K)]
                loc_xy = self._loc_xy[env_id]

                for i in range(self.K):
                    route = self._route_locs[env_id][i]
                    if not route:
                        continue

                    idx = self._route_idx[env_id][i]
                    if idx >= len(route):
                        idx = len(route) - 1

                    tgt_loc = route[idx]
                    tx, ty = loc_xy[tgt_loc]

                    cx = float(self._robot_xy[env_id, i, 0].item())
                    cy = float(self._robot_xy[env_id, i, 1].item())

                    dx, dy = tx - cx, ty - cy
                    dist = math.sqrt(dx * dx + dy * dy)

                    # if reached current node, advance to next
                    reach_eps = 0.2
                    if dist < reach_eps and idx + 1 < len(route):
                        self._route_idx[env_id][i] = idx + 1
                        tgt_loc = route[idx + 1]
                        tx, ty = loc_xy[tgt_loc]
                        dx, dy = tx - cx, ty - cy
                        dist = math.sqrt(dx * dx + dy * dy)

                    theta = math.atan2(dy, dx)

                    step_dist = getattr(ROBOT_SPECS[i], "speed", 1.0) * 3.0
                    if dist > 1e-6:
                        s = min(1.0, step_dist / dist)
                        nx = cx + dx * s
                        ny = cy + dy * s
                    else:
                        nx, ny = cx, cy

                    print(
                        f"[move] robot {i + 1}: from node {route[idx]} ({cx:.2f},{cy:.2f}) to node {tgt_loc} ({tx:.2f},{ty:.2f}) -> new ({nx:.2f},{ny:.2f}) step_dist {step_dist:.3f}"
                    )

                    self._robot_xy[env_id, i, 0] = nx
                    self._robot_xy[env_id, i, 1] = ny
                    self._set_robot_pose_xy(f"robot{i+1}", nx, ny, z=0.0, yaw_rad=theta)

            except Exception:
                self.last_cost[env_id] = float(
                    getattr(self.cfg, "solve_fail_cost", 1e6)
                )
                self.last_status_ok[env_id] = False
                traceback.print_exc()

    def robot_active_priority(self, robot_id: int) -> int:
        cargo = self._robot_cargo[robot_id]  # list of Order
        if not cargo:
            return 0
        return max(int(o.priority) for o in cargo)

    def _orders_to_cuopt(self, orders):
        return [{"id": o.id, "location": o.location} for o in orders]

    def resolve_conflicts(self, proposals):
        # proposals: {rid: (u, v)}
        edge_claims = {}
        node_claims = {}

        for rid, (u, v) in proposals.items():
            edge_claims.setdefault((u, v), []).append(rid)
            node_claims.setdefault(v, []).append(rid)

        allowed = set(proposals.keys())
        blocked = set()

        # edge conflicts
        for edge, rids in edge_claims.items():
            if len(rids) > 1:
                winner = max(rids, key=lambda r: self.robot_active_priority(r))
                for r in rids:
                    if r != winner:
                        blocked.add(r)

        # node conflicts (arrive same node)
        for node, rids in node_claims.items():
            if len(rids) > 1:
                winner = max(rids, key=lambda r: self.robot_active_priority(r))
                for r in rids:
                    if r != winner:
                        blocked.add(r)

        allowed -= blocked
        return allowed, blocked
    def _routes_finished(self, env_id: int) -> bool:
        for i in range(self.K):
            route = self._route_locs[env_id][i]
            if route and self._route_idx[env_id][i] < len(route) - 1:
                return False
        return True