#!/usr/bin/env python3

from isaaclab.app import AppLauncher

# chỉ cần ít CLI: headless/renderer nếu muốn
import argparse
parser = argparse.ArgumentParser(description="Test spawning a scene in Isaac Sim.")
args, _ = parser.parse_known_args()
# nếu muốn chắc chắn có viewer, đặt mặc định:
args.headless = False

# khởi động Isaac Sim (load omni.*)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# chỉ cần ít CLI: headless/renderer nếu muốn
args, _ = parser.parse_known_args()
# nếu muốn chắc chắn có viewer, đặt mặc định:
args.headless = False

from isaaclab.envs import DirectRLEnv
from cuopt_rl_training.tasks.direct.cuopt_rl_training.cuopt_rl_training_env_cfg import CuoptRlTrainingEnvCfg


def main():
    cfg = CuoptRlTrainingEnvCfg()
    # Use small number of envs for quick visualization
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 30.0

    env = DirectRLEnv(cfg, headless=False)
    obs, _ = env.reset()
    print(f"Reset OK. Obs type: {type(obs)}, keys: {list(obs.keys()) if isinstance(obs, dict) else None}")

    steps = 120
    for _ in range(steps):
        env.step(env.action_space.sample() if hasattr(env, "action_space") else None)
    print("Steps OK.")
    env.close()
    

if __name__ == "__main__":
    main()
