from dataclasses import dataclass
from typing import List, Tuple, Dict
import isaaclab.sim as sim_utils

@dataclass(frozen = True)
class RobotSpec:
    """Specifications for different robot types."""
    name: str
    capacity: int
    speed: float  # meters per second
    #battery_life: float  # in hours
    #dimensions: Tuple[float, float, float]  # (length, width, height) in meters
    order_type: int
    usd_path: str


ROBOT_SPECS: Dict[int, RobotSpec] = {
    0: RobotSpec(
        name="carter_v1",
        usd_path="/mnt/hdd/isaac/Isaac-Object/Robot/carter_v1_removecam.usd",  # TODO: chỉnh path đúng
        order_type=1,
        capacity=1,
        speed= 1.0
    ),
    1: RobotSpec(
        name="nova_carter",
        usd_path="/mnt/hdd/isaac/Isaac-Object/Robot/nova_carter_removecam.usd",
        order_type=2,
        capacity=2,
        speed= 1.0
    ),
    2: RobotSpec(
        name="iw_hub",
        usd_path="/mnt/hdd/isaac/Isaac-Object/Robot/Collected_Idealworks/iw_hub.usd",  # TODO: chỉnh path đúng
        order_type=3,
        capacity=3,
        speed= 1.0
    ),
}

def spawn_robot(
    prim_path: str,
    spec: RobotSpec,
    translation: Tuple[float, float, float],
    orientation: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> None:
    cfg = sim_utils.UsdFileCfg(usd_path=spec.usd_path, visible=True)
    cfg.func(
        prim_path,
        cfg,
        translation=list(translation),
        orientation=list(orientation),
        scale=list(scale),
    )

def spawn_all_robots_base(
    base_path: str,
    start_poses: List[Tuple[float, float, float]],
) -> None:
    for rid, spec in ROBOT_SPECS.items():
        prim_path = f"{base_path}/Robot{rid + 1}"
        spawn_robot(
            prim_path=prim_path,
            spec=spec,
            translation=start_poses[rid],
        )
