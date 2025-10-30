import os
import copy
import gymnasium as gym
from .transUtils import *
from .reachEnv import *
import random
from ray.rllib.core.rl_module import RLModule
from pathlib import Path

class ReachEnvHERWrapper(gym.Wrapper):
    def __init__(self, env, env_type="pos",checkpoint_path=""):
        super().__init__(env)
        self.her_ratio = 0.1
        random.seed(42)
        np.random.default_rng(seed=42)
        
        self.env_type = env_type
        # 原始动作空间 --> 简化动作空间
        low, high = env.action_spec
        if self.env_type == "pos":
            self.action_space = gym.spaces.Box(low=low[:3], high=high[:3], dtype=np.float32)
        elif self.env_type == "rot":
            self.action_space = gym.spaces.Box(low=low[3:], high=high[3:], dtype=np.float32)
            self.rl_module = RLModule.from_checkpoint(
                Path(checkpoint_path)
                / "learner_group"
                / "learner"
                / "rl_module"
                / "default_policy"
            )
        else:
            self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        
        # 原始观测空间 --> 新观测空间
        if env_type == "pos":
            obs_dim = 3 + 3
        elif env_type == "rot":
            obs_dim = 3 + 3
        else:
            obs_dim = env.robots[0].dof + 6 + 6
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
            action_pos = self.get_pos_action(self.get_pos_obs())

            action = np.concatenate([
                action_pos,
                action,
            ])
        obs, reward, terminated, truncated, info = self.env.step(action)
        # 修改 observation
        
        if self.env_type != "pose":
            if (random.random() < self.her_ratio) and (reward < 0.0):
                obs = self._process_obs_her(obs)
                reward = 90.0
                info['success'] = True
                # print(f"her_obs:{obs}")
            else:
                obs = self._process_obs(obs)
                info['success'] = self.unwrapped._check_success()

        return obs, reward, terminated, truncated, info
    
    def get_pos_action(self, obs):
        obs_batch = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
        model_outputs = self.rl_module.forward_inference({"obs": obs_batch})
        logits = model_outputs["action_dist_inputs"]
        dist_class = self.rl_module.get_inference_action_dist_cls()
        dist = dist_class.from_logits(logits)
        action_sample = dist.sample()
        action_pos = action_sample.squeeze(0).detach().numpy().astype(np.float32)
        return action_pos
    
    def get_pos_obs(self):
        # 当前episode相机坐标系下机械臂工具的目标位姿
        _, pos_g_c = homogeneous_matrix_to_quaternion(self.unwrapped.get_goal_pose_c())
        # 当前时刻相机坐标系下机械臂工具的实际位姿
        _, pos_t_c = homogeneous_matrix_to_quaternion(self.unwrapped.get_tool_pose_c())
        
        return np.concatenate([pos_g_c, pos_t_c])

    def _get_reward_masac(self, t="combine"):
        return self.unwrapped.cal_reward_value_masac(t)

    def _process_obs_masac(self, obs, obs_type):
        # 当前episode相机坐标系下机械臂工具的目标位姿
        # quat_g_c, pos_g_c = homogeneous_matrix_to_quaternion(self.unwrapped.get_goal_pose_c())
        rvec_g_c, pos_g_c = homogeneous_matrix_to_axisangle(self.unwrapped.get_goal_pose_c())
        # 当前时刻相机坐标系下机械臂工具的实际位姿
        # quat_t_c, pos_t_c = homogeneous_matrix_to_quaternion(self.unwrapped.get_tool_pose_c())
        rvec_t_c, pos_t_c = homogeneous_matrix_to_axisangle(self.unwrapped.get_tool_pose_c())
        
        if obs_type == "pos":
            return np.concatenate([pos_g_c, pos_t_c])
        elif obs_type == "rot":
            return np.concatenate([rvec_g_c, rvec_t_c])
        elif obs_type == "joint_pos":
            return np.concatenate([pos_g_c, pos_t_c, self.unwrapped.get_joint_pos()])
        elif obs_type == "joint_rot":
            return np.concatenate([rvec_g_c, rvec_t_c, self.unwrapped.get_joint_pos()])
        else:
            return np.concatenate([pos_g_c, rvec_g_c, pos_t_c, rvec_t_c])

    def _process_obs_her_masac(self, obs, obs_type):
        # 当前episode相机坐标系下机械臂工具的目标位姿
        # quat_g_c, pos_g_c = homogeneous_matrix_to_quaternion(self.unwrapped.get_goal_pose_c())
        rvec_g_c, pos_g_c = homogeneous_matrix_to_axisangle(self.unwrapped.get_goal_pose_c())
        # 当前时刻相机坐标系下机械臂工具的实际位姿
        # quat_t_c, pos_t_c = homogeneous_matrix_to_quaternion(self.unwrapped.get_tool_pose_c())
        rvec_t_c, pos_t_c = homogeneous_matrix_to_axisangle(self.unwrapped.get_tool_pose_c())
        
        pos_g_c = np.array(
            pos_t_c + np.random.uniform(0, 0.01, size=pos_t_c.shape)
        )
        rvec_g_c = np.array(
            rvec_t_c + np.random.uniform(0, 0.01, size=rvec_t_c.shape)
        )
        
        if obs_type == "pos":
            return np.concatenate([pos_g_c, pos_t_c])
        elif obs_type == "rot":
            return np.concatenate([rvec_g_c, rvec_t_c])
        else:
            return np.concatenate([pos_g_c, rvec_g_c, pos_t_c, rvec_t_c])

    def _process_obs_her(self, obs):
        # 当前时刻基座坐标系下机械臂末端的实际位姿
        joint_positions = self.unwrapped.get_joint_pos()
        # 当前episode相机坐标系下机械臂工具的目标位姿
        # quat_g_c, pos_g_c = homogeneous_matrix_to_quaternion(self.unwrapped.get_goal_pose_c())
        rvec_g_c, pos_g_c = homogeneous_matrix_to_axisangle(self.unwrapped.get_goal_pose_c())
        # 当前时刻相机坐标系下机械臂工具的实际位姿
        # quat_t_c, pos_t_c = homogeneous_matrix_to_quaternion(self.unwrapped.get_tool_pose_c())
        rvec_t_c, pos_t_c = homogeneous_matrix_to_axisangle(self.unwrapped.get_tool_pose_c())
        
        pos_g_c = np.array(
            pos_t_c + np.random.uniform(0, 0.01, size=pos_t_c.shape)
        )
        rvec_g_c = np.array(
            rvec_t_c + np.random.uniform(0, 0.01, size=rvec_t_c.shape)
        )
        
        if self.env_type == "pos":
            return np.concatenate([pos_g_c, pos_t_c])
        elif self.env_type == "rot":
            return np.concatenate([rvec_g_c, rvec_t_c])
        else:
            return np.concatenate([joint_positions, pos_g_c, rvec_g_c, pos_t_c, rvec_t_c])

    def _process_obs(self, obs):
        # 当前时刻基座坐标系下机械臂末端的实际位姿
        joint_positions = self.unwrapped.get_joint_pos()
        # 当前episode相机坐标系下机械臂工具的目标位姿
        # quat_g_c, pos_g_c = homogeneous_matrix_to_quaternion(self.unwrapped.get_goal_pose_c())
        rvec_g_c, pos_g_c = homogeneous_matrix_to_axisangle(self.unwrapped.get_goal_pose_c())
        # 当前时刻相机坐标系下机械臂工具的实际位姿
        # quat_t_c, pos_t_c = homogeneous_matrix_to_quaternion(self.unwrapped.get_tool_pose_c())
        rvec_t_c, pos_t_c = homogeneous_matrix_to_axisangle(self.unwrapped.get_tool_pose_c())
        
        if self.env_type == "pos":
            return np.concatenate([pos_g_c, pos_t_c])
        elif self.env_type == "rot":
            return np.concatenate([rvec_g_c, rvec_t_c])
        else:
            return np.concatenate([joint_positions, pos_g_c, rvec_g_c, pos_t_c, rvec_t_c])

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
    "train_type": "rot",
    "reset_policy": 2,
    "reward_scale": 1.0,
    "use_object_obs": False,
    "sim2real": False,
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
    "train_type": "rot",
    "reset_policy": 2,
    "reward_scale": 1.0,
    "use_object_obs": False,
}

def make_reach_env(render=False, **kwargs):
    log_dir=kwargs.get("log_dir")
    if render:
        env = ReachEnv(**eval_env_config, log_dir=log_dir)
    else:
        env = ReachEnv(**train_env_config, log_dir=log_dir)
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

def make_reach_her_env(render=False, **kwargs):
    log_dir=kwargs.get("log_dir")
    if render:
        env = ReachEnv(**eval_env_config, log_dir=log_dir)
    else:
        env = ReachEnv(**train_env_config, log_dir=log_dir)
    env = GymWrapper(env)
    checkpoint_path = "/home/ey/rl/src/rlreach2/rlreach/ray/db/ray_results/Reach_2025-08-19_20-10-32/reach/SAC_ReachEnvHERGym_81d54_00000_0_2025-08-19_20-10-33/checkpoint_000155"
    return ReachEnvHERWrapper(env, env_type=train_env_config['train_type'],checkpoint_path=checkpoint_path)

def make_reach_her_env_masac(render=False, **kwargs):
    log_dir=kwargs.get("log_dir")
    train_env_config["train_type"] = 'pose'
    train_env_config["has_renderer"] = kwargs.get("has_renderer")
    train_env_config["sim2real"] = kwargs.get("sim2real")
    if render:
        env = ReachEnv(**eval_env_config, log_dir=log_dir)
    else:
        env = ReachEnv(**train_env_config, log_dir=log_dir)
    env = GymWrapper(env)
    return ReachEnvHERWrapper(env, env_type='pose')

class ReachEnvHERGym(gym.Env):
    def __init__(self, env_config):
        self.env = make_reach_her_env(log_dir=env_config.get("log_dir"))
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