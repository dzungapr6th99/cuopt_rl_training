import isaaclab.sim  as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

IW_ROBOT_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path="D:/Isaac-Project/iw_hub.usd"),
        actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            damping=None, stiffness=None
        )
    },
)