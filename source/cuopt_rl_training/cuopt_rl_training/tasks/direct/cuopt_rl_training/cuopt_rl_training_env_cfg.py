from cuopt_rl_training.robots.nova_carter import NOVA_CARTER_CONFIG
from cuopt_rl_training.robots.iw_hub import IW_ROBOT_CONFIG
from cuopt_rl_training.robots.carter_v1 import CARTER_V1_CONFIG

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

@configclass
class CuoptRlTrainingEnvCfg(DirectRLEnvCfg):
    # paths
    scene_usd_path: str = "/mnt/hdd/isaac/Isaac-Object/Warehouse_cuopt_graph_only.usd"
    nova_carter_usd_path: str = "/mnt/hdd/isaac/Isaac-Object/Robot/nova_carter_removecam.usd"
    iw_hub_usd_path: str = "/mnt/hdd/isaac/Isaac-Object/Robot/Collected_Idealworks/iw_hub.usd"
    carter_v1_usd_path: str = "/mnt/hdd/isaac/Isaac-Object/Robot/carter_v1_removecam.usd"
    # env
    decimation: int = 2
    episode_length_s: float = 5.0

    # ====== Fleet scenario (server-training) ======
    n_locations: int = 30
    n_vehicles: int = 3
    n_orders: int = 20

    # cuOpt settings
    cuopt_time_limit: float = 5.0
    cuopt_seed: int = 0

    # Solve timing
    solve_on_reset: bool = True
    solve_every_step: bool = True  # call cuOpt each RL step (simple + stable)

    # spaces:
    # Observation vector in env below:
    # base = [n_vehicles, n_orders, blocked_cnt, last_cost, status_ok] => 5
    # obs_dim = 5 + action_space
    action_space: int = 1
    observation_space: int = 6  # = 5 + action_space
    state_space: int = 0

    # action behavior
    action_clip: float = 1.0
    blocked_edge_cost: float = 1e6
    solve_fail_cost: float = 1e6
    solve_fail_penalty: float = 1000.0
    terminate_on_solve_fail: bool = False

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1/24, render_interval=decimation)

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=4.0,
        replicate_physics=True,
    )

    # optional warehouse USD (not spawned in current env code yet)
    warehouse_cfg = sim_utils.UsdFileCfg(usd_path=scene_usd_path)
    # (Optional) robot configs if you later spawn for visualization
    #robot_cfg: ArticulationCfg = NOVA_CARTER_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")
    robot1_cfg = NOVA_CARTER_CONFIG.replace(prim_path="/World/envs/env_0.*/Robots/Robot1")
    robot2_cfg = IW_ROBOT_CONFIG.replace(prim_path="/World/envs/env_0.*/Robots/Robot2")
    robot3_cfg = CARTER_V1_CONFIG.replace(prim_path="/World/envs/env_0.*/Robots/Robot3")

    dof_names = ["left_wheel_joint", "right_wheel_joint"]
