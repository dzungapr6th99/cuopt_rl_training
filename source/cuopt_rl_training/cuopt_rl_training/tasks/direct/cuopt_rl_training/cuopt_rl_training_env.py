from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any, Dict, List, Tuple

import omni.usd
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
import isaaclab.sim.utils.prims as prim_utils
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane, spawn_from_usd
from isaaclab.utils.math import quat_from_euler_xyz

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
            eye=(20.0, 20.0, 25.0),
            lookat=(20.0, 20.0, 0.0),
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

    # ----------------------- Scene -----------------------

    def _setup_scene(self):
        #spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        spawn_from_usd(
            prim_path="/World/envs/env_0/Warehouse",
            cfg=self.cfg.warehouse_cfg,
        )

        # Parent prim for robot assets (required by regex prim paths).
        prim_utils.create_prim("/World/envs/env_0/Robots", prim_type="Xform")

        # Articulations for 3 robots
        self.robot1 = Articulation(self.cfg.robot1_cfg)
        self.robot2 = Articulation(self.cfg.robot2_cfg)
        self.robot3 = Articulation(self.cfg.robot3_cfg)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot1"] = self.robot1
        self.scene.articulations["robot2"] = self.robot2
        self.scene.articulations["robot3"] = self.robot3
        self.robots = [self.robot1, self.robot2, self.robot3]

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

        # plan only once per episode
        to_plan = (~self._planned).nonzero(as_tuple=False).squeeze(-1)
        if to_plan.numel() > 0:
            self._plan_routes_for_env_ids(to_plan)
            self._planned[to_plan] = True

        # move every step
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
        root_path = f"/World/envs/env_{env_id}/Warehouse/Warehouse/Transportation/WaypointGraph"
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
            self._set_robot_pose_xy(i, x, y, z=0.3, yaw_rad=0.0)

    def _set_robot_pose_xy(
        self,
        robot_id: int,
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

                cost = (
                    0.0
                    if len(plan_rows) == 0
                    else max(float(r["eta"]) for r in plan_rows)
                )
                self.last_cost[env_id] = float(cost)
                self.last_status_ok[env_id] = True

                n_veh = len(starts) if starts is not None else self.K
                routes = self._build_routes_from_plan(plan_rows, n_vehicles=n_veh)
                self._route_locs[env_id] = routes
                self._assign_orders_to_routes(env_id, routes)
                self._last_node[env_id] = [
                    r[0] if r else None for r in self._route_locs[env_id]
                ]
                veh_ids = sorted({int(r["vehicle"]) for r in plan_rows}) if plan_rows else []
                print(f"[plan] env {env_id} vehicles: {veh_ids} route_lens: {[len(r) for r in routes]}")

            except Exception:
                self.last_cost[env_id] = float(getattr(self.cfg, "solve_fail_cost", 1e6))
                self.last_status_ok[env_id] = False
                traceback.print_exc()

    def _advance_robots(self, env_ids: Sequence[int]) -> None:
        dt = float(getattr(self.cfg.sim, "dt", 1 / 24)) * int(
            getattr(self.cfg, "decimation", 1)
        )

        for eid in env_ids:
            env_id = int(eid)
            loc_xy = self._loc_xy[env_id]

            next_nodes: Dict[int, int] = {}
            for i in range(self.K):
                route = self._route_locs[env_id][i]
                if not route:
                    continue
                next_nodes[i] = route[0]

            allowed = self._resolve_next_node_conflicts(env_id, next_nodes)

            for i in range(self.K):
                route = self._route_locs[env_id][i]
                nxt = route[0] if route else None
                state = "allow" if i in allowed else "wait"
                print(f"[next] env {env_id} robot {i}: next {nxt} ({state})")

            for i in range(self.K):
                if i not in next_nodes:
                    continue
                if i not in allowed:
                    continue

                route = self._route_locs[env_id][i]

                tgt_loc = route[0]
                tx, ty = loc_xy[tgt_loc]

                cx = float(self._robot_xy[env_id, i, 0].item())
                cy = float(self._robot_xy[env_id, i, 1].item())

                dx, dy = tx - cx, ty - cy
                dist = math.sqrt(dx * dx + dy * dy)

                reach_eps = 0.2
                if dist < reach_eps:
                    prev_loc = self._route_locs[env_id][i][0]
                    order = self._route_orders[env_id][i][0] if self._route_orders[env_id][i] else None
                    if order is not None:
                        order.state = OrderState.DELIVERED
                        if order in self._robot_cargo[env_id][i]:
                            self._robot_cargo[env_id][i].remove(order)

                    # pop current node from route queue
                    self._route_locs[env_id][i].pop(0)
                    if self._route_orders[env_id][i]:
                        self._route_orders[env_id][i].pop(0)

                    if not self._route_locs[env_id][i]:
                        continue
                    self._last_node[env_id][i] = prev_loc
                    tgt_loc = self._route_locs[env_id][i][0]
                    tx, ty = loc_xy[tgt_loc]
                    dx, dy = tx - cx, ty - cy
                    dist = math.sqrt(dx * dx + dy * dy)
                    print(f"[step] env {env_id} robot {i}: node {prev_loc} -> {tgt_loc}")

                theta = math.atan2(dy, dx)

                step_dist = self._robot_speeds[i] * dt
                if dist > 1e-6:
                    s = min(1.0, step_dist / dist)
                    nx = cx + dx * s
                    ny = cy + dy * s
                else:
                    nx, ny = cx, cy

                self._robot_xy[env_id, i, 0] = nx
                self._robot_xy[env_id, i, 1] = ny
                self._set_robot_pose_xy(i, nx, ny, z=0.0, yaw_rad=theta)

    def _orders_to_cuopt(self, orders):
        return [{"id": o.id, "location": o.location} for o in orders]
