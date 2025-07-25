import os
import gymnasium as gym
from .transUtils import *
from .reachEnv import *

class ReachEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        # 将robosuite的动作空间和观察空间转换成gym的
        low, high = env.action_spec
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        # self.observation_space = gym.spaces.Dict({
        #     'achieved_goal': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(env.robots[0].dof,), dtype=np.float32),    
        #     'desired_goal': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),    
        #     'observation': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),      
        # })
        obs_dim = env.robots[0].dof + 7 + 7
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs = self._process_obs(obs)
        infos = {
            'success': False
        }
        return obs, infos

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # 修改 observation
        obs = self._process_obs(obs)
        info['success'] = self.unwrapped._check_success()

        return obs, reward, terminated, truncated, info

    def _process_obs(self, obs):
        # 当前时刻基座坐标系下机械臂末端的实际位姿
        joint_positions = self.unwrapped.get_joint_pos()
        # 当前episode相机坐标系下机械臂工具的目标位姿
        quat_g_c, pos_g_c = homogeneous_matrix_to_quaternion(self.unwrapped.get_goal_pose_c())
        # 当前时刻相机坐标系下机械臂工具的实际位姿
        quat_t_c, pos_t_c = homogeneous_matrix_to_quaternion(self.unwrapped.get_tool_pose_c())
        # return {
        #     "achieved_goal": joint_positions,
        #     "desired_goal": np.concatenate([pos_g_c, quat_g_c]),
        #     "observation": np.concatenate([pos_t_c, quat_t_c]),
        # }
        return np.concatenate([joint_positions, pos_g_c, quat_g_c, pos_t_c, quat_t_c])

if os.path.exists("reachController.json"):
    controller_fpath = "./reachController.json"
else:
    controller_fpath = "./common/reachController.json"
    
reach_env_config = {
    "robots": ["UR5e"],         # 机器人
    "controller_configs": load_composite_controller_config(controller=controller_fpath),
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
def make_reach_env(**kwargs):
    env = ReachEnv(**reach_env_config)
    env = GymWrapper(env)
    return ReachEnvWrapper(env)

class ReachEnvGym(gym.Env):
    def __init__(self, env_config):
        self.env = make_reach_env()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.reset()

    def reset(self, **kwargs):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def close(self):
        self.env.close()


