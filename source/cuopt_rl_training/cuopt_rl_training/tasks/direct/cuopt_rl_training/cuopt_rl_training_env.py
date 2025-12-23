from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import Any, Dict, List, Tuple
import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
import isaaclab.sim.utils.prims as prim_utils
import isaaclab.sim.simulation_context
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane, spawn_from_usd
from .cuopt_planner import CuOptPlanner
from .cuopt_rl_training_env_cfg import CuoptRlTrainingEnvCfg

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

        # cuOpt solver
        self.planner = CuOptPlanner(
            time_limits=float(getattr(self.cfg, "cuopt_time_limit", 5.0)),
            seed=int(getattr(self.cfg, "cuopt_seed", 0)),
        )

        # 3 visual robots
        self.K = 3

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

        # Scenario state per env (python lists ok for now)
        self._cost_matrices: List[List[List[float]]] = [None] * self.num_envs  # type: ignore
        self._orders: List[List[Dict[str, Any]]] = [None] * self.num_envs  # type: ignore
        self._vehicle_starts: List[List[int]] = [None] * self.num_envs  # type: ignore
        self._vehicle_returns: List[List[int]] = [None] * self.num_envs  # type: ignore
        self._blocked_edges: List[List[Tuple[int, int]]] = [
            [] for _ in range(self.num_envs)
        ]

        # Location -> (x,y) lookup per env
        self._loc_xy: List[List[Tuple[float, float]]] = [None] * self.num_envs  # type: ignore
    # ----------------------- Scene -----------------------

    def _setup_scene(self):
        # Load warehouse USD in source env_0 so clone works

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        
        spawn_from_usd(
            prim_path="/World/envs/env_0/Warehouse",
            cfg=self.cfg.warehouse_cfg,
        )
        # Create parent prim for robot visuals (must exist before spawn)
        #prim_utils.create_prim("/World/envs/env_0/Robots", prim_type="Xform")

        # Spawn visual robots (USD references). These do NOT need ArticulationRootAPI.
        r1 = sim_utils.UsdFileCfg(usd_path=self.cfg.nova_carter_usd_path, visible=True)
        r2 = sim_utils.UsdFileCfg(usd_path=self.cfg.iw_hub_usd_path, visible=True)
        r3 = sim_utils.UsdFileCfg(usd_path=self.cfg.jetbot_usd_path, visible=True)
        
        r1.func(
            "/World/envs/env_0/Robot1",
            r1,
            translation=[17.98, 4.16, 0.0],
            orientation=[1, 0, 0, 0],
            scale=[1, 1, 1],
        )
        r2.func(
            "/World/envs/env_0/Robot2",
            r2,
            translation=[24.72, 4.16, 0.0],
            orientation=[1, 0, 0, 0],
            scale=[1, 1, 1],
        )
        r3.func(
            "/World/envs/env_0/Robot3",
            r3,
            translation=[11.14, 4.16, 0.0],
            orientation=[1, 0, 0, 0],
            scale=[1, 1, 1],
        )

        # Clone/replicate envs
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # Light (optional)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

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
        if bool(getattr(self.cfg, "solve_every_step", True)):
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            self._solve_for_env_ids(env_ids)
        return

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

    # ----------------------- Scenario -----------------------

    def _build_scenario(self, env_id: int) -> None:
        n_locations = int(getattr(self.cfg, "n_locations", 30))
        n_vehicles = int(getattr(self.cfg, "n_vehicles", 3))
        n_orders = int(getattr(self.cfg, "n_orders", 20))

        # Location -> XY (demo: straight line)
        self._loc_xy[env_id] = [(float(i) * 1.0, 0.0) for i in range(n_locations)]

        # Dummy symmetric cost matrix
        cm = [[0.0 for _ in range(n_locations)] for _ in range(n_locations)]
        for i in range(n_locations):
            for j in range(n_locations):
                cm[i][j] = 0.0 if i == j else float(abs(i - j))

        # 3 different starts
        starts = [0, min(5, n_locations - 1), min(10, n_locations - 1)]
        starts = starts[:n_vehicles]
        vehicle_starts = starts
        vehicle_returns = list(vehicle_starts)

        # 20 tasks (deterministic demo). You can randomize later.
        orders = [
            {"id": k, "location": int((k + 1) % n_locations)} for k in range(n_orders)
        ]

        self._blocked_edges[env_id] = []
        self._cost_matrices[env_id] = cm
        self._orders[env_id] = orders
        self._vehicle_starts[env_id] = vehicle_starts
        self._vehicle_returns[env_id] = vehicle_returns

    # ----------------------- Visual robots -----------------------

    def _reset_visual_robots(self, env_id: int) -> None:
        """Place 3 visual robots at start locations (0,5,10)."""
        nloc = int(getattr(self.cfg, "n_locations", 30))
        starts = [0, min(5, nloc - 1), min(10, nloc - 1)]
        starts = starts[: self.K]

        loc_xy = self._loc_xy[env_id]
        for i, loc in enumerate(starts):
            x, y = loc_xy[loc]
            self._robot_xy[env_id, i, 0] = x
            self._robot_xy[env_id, i, 1] = y
            self._robot_yaw[env_id, i] = 0.0

            prim_path = f"/World/envs/env_{env_id}/Robot{i+1}"
            self._set_robot_pose_xy(prim_path, x, y, z=0.01, yaw_rad=0.0)

    def _set_robot_pose_xy(
        self,
        prim_path: str,
        x: float,
        y: float,
        z: float = 0.0,
        yaw_rad: float = 0.0,
    ) -> None:
        """
        Teleport a visual USD prim by authoring its transform ops (no respawn).
        Requires standalone runtime (pxr available).

        - Uses XformCommonAPI: Translate + Rotate (Z) + Scale (optional keep 1).
        - Rotation is set as Euler degrees around Z (Yaw).
        """
        # Lazy-import to avoid breaking non-standalone runs
        from pxr import UsdGeom, Gf, Usd

        # Cache stage once (recommended). If you already store self._stage, use it.
        

        prim = prim_utils.get_prim_at_path(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim not found or invalid: {prim_path}")

        # Ensure prim is Xformable
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            raise ValueError(f"Prim is not Xformable: {prim_path}")

        # Use XformCommonAPI for clean TR (and optional S)
        common = UsdGeom.XformCommonAPI(prim)

        # Translation
        common.SetTranslate(Gf.Vec3d(float(x), float(y), float(z)))

        # Rotation: XformCommonAPI uses XYZ Euler degrees
        yaw_deg = float(yaw_rad * 180.0 / math.pi)
        common.SetRotate(Gf.Vec3f(0.0, 0.0, yaw_deg), UsdGeom.XformCommonAPI.RotationOrderXYZ)

        # Keep scale = 1 (optional but prevents inherited weirdness if authoring exists)
        common.SetScale(Gf.Vec3f(1.0, 1.0, 1.0))

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

    def _solve_for_env_ids(self, env_ids: Sequence[int]) -> None:
        # Move step for visual robots
        dt = float(getattr(self.cfg.sim, "dt", 1 / 24)) * int(
            getattr(self.cfg, "decimation", 1)
        )
        speed = float(getattr(self.cfg, "marker_speed", 1.0))
        step_dist = speed * dt

        for eid in env_ids:
            env_id = int(eid)

            cm0 = self._cost_matrices[env_id]
            orders = self._orders[env_id]
            starts = self._vehicle_starts[env_id]
            rets = self._vehicle_returns[env_id]

            cm = self._apply_action_to_cost_matrix(env_id, cm0)

            try:
                plan_rows = self.planner.plan(
                    cost_matrix=cm,
                    orders=orders,
                    vehicle_starts=starts,
                    vehicle_returns=rets,
                )

                # Scalar cost proxy: makespan (max ETA)
                cost = (
                    0.0
                    if len(plan_rows) == 0
                    else max(float(r["eta"]) for r in plan_rows)
                )
                self.last_cost[env_id] = float(cost)
                self.last_status_ok[env_id] = True

                # Update visual robots toward next targets
                n_veh = len(starts) if starts is not None else self.K
                targets = self._extract_next_targets(plan_rows, n_vehicles=n_veh)
                loc_xy = self._loc_xy[env_id]

                for i in range(self.K):
                    tgt_loc = int(targets[i]) if i < len(targets) else 0
                    tx, ty = loc_xy[tgt_loc]

                    cx = float(self._robot_xy[env_id, i, 0].item())
                    cy = float(self._robot_xy[env_id, i, 1].item())

                    dx = tx - cx
                    dy = ty - cy
                    dist = math.sqrt(dx * dx + dy * dy)

                    if dist > 1e-6:
                        s = min(1.0, step_dist / dist)
                        nx = cx + dx * s
                        ny = cy + dy * s
                    else:
                        nx, ny = cx, cy

                    self._robot_xy[env_id, i, 0] = nx
                    self._robot_xy[env_id, i, 1] = ny

                    prim_path = f"/World/envs/env_{env_id}/Robots/Robot{i+1}"
                    self._set_robot_pose_xy(prim_path, nx, ny, z=0.0, yaw_rad=0.0)

            except Exception:
                self.last_cost[env_id] = float(
                    getattr(self.cfg, "solve_fail_cost", 1e6)
                )
                self.last_status_ok[env_id] = False
