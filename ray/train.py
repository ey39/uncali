from common.reachEnv import * 

env_config = {
    "robots": ["UR5e"],         # 机器人
    "controller_configs": load_composite_controller_config(controller="BASIC"),
    "has_renderer": False,
    "has_offscreen_renderer": False,
    "reward_shaping": True,
    "horizon": 200,
    "control_freq": 20,
    "seed": 42,
    "train_type": "pose",
    "reset_policy": 2,
    "reward_scale": 1.0,
    "use_object_obs": False
}

env = ReachEnv(**env_config)