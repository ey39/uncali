import os
import gymnasium as gym
from .transUtils import *
from .reachEnv import *

class ReachEnvSimpleWrapper(gym.Wrapper):
    def __init__(self, env, env_type="pos"):
        super().__init__(env)
        self.env_type = env_type
        # 原始动作空间 --> 简化动作空间
        low, high = env.action_spec
        if self.env_type == "pos":
            self.action_space = gym.spaces.Box(low=low[:3], high=high[:3], dtype=np.float32)
        elif self.env_type == "rot":
            self.action_space = gym.spaces.Box(low=low[3:], high=high[3:], dtype=np.float32)
        else:
            self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        
        # 原始观测空间 --> 新观测空间
        if env_type == "pos":
            obs_dim = 3 + 3
        elif env_type == "rot":
            obs_dim = 4 + 4
        else:
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
        if self.env_type == "pos":
            action = np.concatenate([
                action,
                np.zeros((3,),dtype=np.float32),
            ])
        elif self.env_type == "rot":
            action = np.concatenate([
                np.zeros((3,),dtype=np.float32),
                action,
            ])
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
        
        if self.env_type == "pos":
            return np.concatenate([pos_g_c, pos_t_c])
        elif self.env_type == "rot":
            return np.concatenate([quat_g_c, quat_t_c])
        else:
            return np.concatenate([joint_positions, pos_g_c, quat_g_c, pos_t_c, quat_t_c])

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
    
train_env_config = {
    "robots": ["UR5e"],         # 机器人
    "controller_configs": load_composite_controller_config(controller=controller_fpath),
    "has_renderer": False,
    "has_offscreen_renderer": False,
    "reward_shaping": True,
    "horizon": 200,
    "control_freq": 20,
    "seed": 42,
    "train_type": "pos",
    "reset_policy": 2,
    "reward_scale": 1.0,
    "use_object_obs": False,
}

eval_env_config = {
    "robots": ["UR5e"],         # 机器人
    "controller_configs": load_composite_controller_config(controller=controller_fpath),
    "has_renderer": False,
    "has_offscreen_renderer": False,
    "reward_shaping": True,
    "horizon": 200,
    "control_freq": 20,
    "seed": 42,
    "train_type": "pos",
    "reset_policy": 2,
    "reward_scale": 1.0,
    "use_object_obs": False,
}

def make_reach_env(render=False, **kwargs):
    log_dir=kwargs.get("log_dir")
    if render:
        env = ReachEnv(**train_env_config, log_dir=log_dir)
    else:
        env = ReachEnv(**eval_env_config, log_dir=log_dir)
    env = GymWrapper(env)
    return ReachEnvWrapper(env)

class ReachEnvGym(gym.Env):
    def __init__(self, env_config):
        self.env = make_reach_env(log_dir=env_config.get("log_dir"))
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

def make_reach_simple_env(render=False, **kwargs):
    log_dir=kwargs.get("log_dir")
    if render:
        env = ReachEnv(**train_env_config, log_dir=log_dir)
    else:
        env = ReachEnv(**eval_env_config, log_dir=log_dir)
    env = GymWrapper(env)
    return ReachEnvSimpleWrapper(env, env_type=train_env_config['train_type'])

class ReachEnvSimpleGym(gym.Env):
    def __init__(self, env_config):
        self.env = make_reach_simple_env(log_dir=env_config.get("log_dir"))
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        _, _ = self.reset()

    def reset(self, **kwargs):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def close(self):
        self.env.close()
