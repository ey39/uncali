# ===================gym=========================
import gym
# ===================datatime====================
from datetime import datetime
# ===================numpy=======================
import numpy as np
# ===================robosuite===================
import robosuite as suite
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.tasks import ManipulationTask
from robosuite.models.arenas import TableArena
from robosuite.models.objects import CylinderObject
from robosuite.models.robots import Panda
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.controllers import load_controller_config
import xml.etree.ElementTree as ET
import robosuite.utils.transform_utils as T
from robosuite.utils.observables import Observable, sensor
from robosuite.wrappers import GymWrapper
# ===================torch======================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# ===================skrl=======================
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
# ===================users======================
from reachUtils import *

# ===============================================
# |               environment                   |
# ===============================================
class ReachEnv(SingleArmEnv):
    def __init__(self,
                 controller_config=None,
                 has_renderer=True,
                 has_offscreen_renderer=False,
                 use_camera_obs=False,
                 control_freq=20,
                 horizon=200,
                 reward_shaping=True,
                 seed=123,
                 **kwargs):
        
        # 创建 Panda 机械臂
        robots = ["UR5e"]
        # 使用robosuite创建环境
        super().__init__(
            robots=robots,
            controller_configs=controller_config,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            use_camera_obs=use_camera_obs,
            control_freq=control_freq,
            horizon=horizon,
            gripper_types=None,
            **kwargs
        )
        
        # 参数
        self.reward_scale = 1.0
        self.use_object_obs = False
        self.seed = seed
        self.reward_shaping = reward_shaping
        self.best_pos_err = 999.9
        self.best_rot_err = 999.9
        self.init_pos_err = 999.9
        self.init_rot_err = 999.9
        self.episode_ctr = 0
        self.reward_value = 0.0

        # 初始化世界坐标系下相机位姿
        self.T_camera_world = generate_random_homogeneous_transform(
            translation_range=[(-0.5, 0.5), (-0.5, 0.5), (0.8, 1.3)],
            rotation_mode='euler',
            rotation_range=(-90, 90),
            seed=self.seed
        )
        # 初始化末端坐标系下工具位姿
        self.T_tool_end = generate_random_homogeneous_transform(
            translation_range=[(-0.1, 0.1), (-0.1, 0.1), (0.1, 0.1)],
            rotation_mode='euler',
            rotation_range=(-45, 45),
        )
        # 初始化世界坐标系下目标位姿
        self.pos_range = 0.3
        self.rot_range = 30
        self.T_target_world = generate_perturbed_transform(
            base_transform=self.get_robot_pose(),
            translation_error_range=(-self.pos_range, self.pos_range),
            rotation_error_range=(-self.rot_range, self.rot_range),
            rotation_mode='euler',
            seed=42,   
        )
        self.pos_err_threshold = 0.05   # m
        self.rot_err_threshold = 0.3    # rad
        self.camera_reset_episodes = 10
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # 构造日志目录
        log_dir = f"runs_reach/torch/reach_task/reach_task_{timestamp}"
        self.writer = SummaryWriter(log_dir=log_dir)
        self.total_steps = 0
        self.reset_policy_choice = 1

    def _load_model(self):
        self.frame_definitions = {}
        super()._load_model()
        # 桌子
        # Adjust base pose accordingly
        self.table_full_size=(0.8, 0.8, 0.05)
        self.table_friction=(1.0, 5e-3, 1e-4)
        self.table_offset = np.array((0, 0, 0.8))
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[],
        )
    
    def _setup_observables(self):
        '''
        自定义观测空间
        '''
        observables = super()._setup_observables()
        return observables
    
    def get_robot_joint_positions(self):
        '''
        获取机械臂关节位置
        '''
        joint_positions = self.robots[0].controller.joint_pos
        return joint_positions

    def get_robot_joint_velocities(self):
        '''
        获取机械臂关节速度
        '''
        joint_velocities = self.robots[0].controller.joint_vel
        return joint_velocities

    def get_robot_pose(self):
        """
        获取世界坐标系下机械臂末端tcp的位姿
        """
        body_id = self.sim.model.body_name2id(self.robots[0].robot_model.eef_name)
        pos = self.sim.data.body_xpos[body_id]
        quat = T.convert_quat(self.sim.data.body_xquat[body_id], to="xyzw")
        trans = quaternion_to_homogeneous_matrix(quat, pos)
        return trans
    
    def get_base_pose(self):
        """
        获取世界坐标系下机械臂基座的位姿
        """
        body_id = self.sim.model.body_name2id('robot0_base')
        pos = self.sim.data.body_xpos[body_id]
        quat = T.convert_quat(self.sim.data.body_xquat[body_id], to="xyzw")
        trans = quaternion_to_homogeneous_matrix(quat, pos)
        return trans
    
    def get_robot_pose_base(self):
        """
        获取基座坐标系下机械臂末端tcp的位姿
        """
        T_base_world = self.get_base_pose()
        T_end_base = compose_transforms(invert_homogeneous_matrix(T_base_world), self.get_robot_pose())
        return T_end_base
    
    def get_tool_pose(self):
        """
        获取相机坐标系下机械臂末端工具的位姿
        """
        T_end_world = self.get_robot_pose()
        T_tool_camera = compose_transforms(invert_homogeneous_matrix(self.T_camera_world), T_end_world, self.T_tool_end)
        return T_tool_camera

    def set_tool_pose(self, T_tool_end):
        """
        设置工具坐标系与末端坐标系的齐次变换
        """
        self.T_tool_end = T_tool_end

    def get_target_pose(self):
        """
        获取相机坐标系下目标位姿
        """
        T_target_camera = compose_transforms(invert_homogeneous_matrix(self.T_camera_world), self.T_target_world)
        return T_target_camera
    
    def set_target_pose(self, T_target_camera):
        """
        设置相机坐标系下目标位姿
        """
        self.T_target_world = self.T_camera_world @ T_target_camera
    
    def update_target_pose(self):
        """
        设置新的目标位姿
        """
        while True:
            self.T_target_world = generate_perturbed_transform(
                base_transform=self.get_robot_pose(),
                translation_error_range=(-self.pos_range, self.pos_range),
                rotation_error_range=(-self.rot_range, self.rot_range),
                rotation_mode='euler',
            )
            T_target_camera = self.get_target_pose()
            T_tool_camera = self.get_tool_pose()
            error_info = calculate_pose_error(T_target_camera, T_tool_camera, angle_unit='radians')
            self.init_pos_err = error_info['translation_magnitude']
            self.init_rot_err = error_info['rotation_error_angle']
            if (self.init_pos_err > self.pos_err_threshold) and (self.init_rot_err > self.rot_err_threshold):
                break

    def update_camera_pose(self):
        """
        设置新的相机位姿
        """
        while True:
            self.T_camera_world = generate_perturbed_transform(
                base_transform=self.T_camera_world,
                translation_error_range=(-0.5, 0.5),
                rotation_error_range=(-45, 45),
                rotation_mode='euler',
            )
            T_target_camera = self.get_target_pose()
            T_tool_camera = self.get_tool_pose()
            error_info = calculate_pose_error(T_target_camera, T_tool_camera, angle_unit='radians')
            self.init_pos_err = error_info['translation_magnitude']
            self.init_rot_err = error_info['rotation_error_angle']
            if (self.init_pos_err > self.pos_err_threshold) and (self.init_rot_err > self.rot_err_threshold):
                break
        
    
    def set_camera_pose(self, T_camera_world):
        """
        设置世界坐标系下相机位姿
        """
        self.T_camera_world = T_camera_world   

    def get_all_pose(self):
        """
        获取所有位姿
        """
        poses = {
            "base": self.get_base_pose(),
            "eef": self.get_robot_pose(),
            "cam": self.T_camera_world,
            "target": self.T_target_world,
        }
        return poses 


    def _setup_observables(self):
        observables = super()._setup_observables()

        return observables

    def reset(self, seed=None, options=None):
        """
        自定义reset方法
        """
        # 调用父类reset
        obs_dict = super().reset()
        # 转换为gym风格的observation
        observation = obs_dict
        self.episode_ctr = self.episode_ctr + 1
        print("\n=============================================================")
        print(f" | episode: {self.episode_ctr} | reward: {self.reward_value} ")
        print(f" | init_pos_err: {self.init_pos_err} | best_pos_err: {self.best_pos_err}")
        print(f" | init_rot_err: {self.init_rot_err} | best_rot_err: {self.best_rot_err}")
        print("=============================================================")
        self.best_pos_err = None
        self.best_rot_err = None
        self.reward_value = 0.0

        if self.reset_policy_choice==1:
            # 重置目标位姿
            self.update_target_pose()
            # 重置相机位姿
            if self.episode_ctr % self.camera_reset_episodes == 0:
                self.update_camera_pose()
                print("\n=============================================================")
                print(f" reset camera pose | episode: {self.episode_ctr} | steps: {self.total_steps}")
                print("=============================================================")
        elif self.reset_policy_choice==2:
            # 重置相机位姿
            self.update_camera_pose()
            # 重置目标位姿
            if self.episode_ctr % self.camera_reset_episodes == 0:
                self.update_target_pose()
                print("\n=============================================================")
                print(f" reset target pose | episode: {self.episode_ctr} | steps: {self.total_steps}")
                print("=============================================================")

        return observation 

    def _reset_internal(self):
        super()._reset_internal()

    def reward(self, action=None):
        '''
        自定义奖励函数
        '''
        # 计算位置和姿态误差
        pos_err_threshold = self.pos_err_threshold    
        rot_err_threshold = self.rot_err_threshold     
        T_target_camera = self.get_target_pose()
        T_tool_camera = self.get_tool_pose()
        error_info = calculate_pose_error(T_target_camera, T_tool_camera, angle_unit='radians')
        self.pos_err = error_info['translation_magnitude']
        self.rot_err = error_info['rotation_error_angle']
        if (self.best_pos_err is None) or (self.best_pos_err > self.pos_err):
            self.best_pos_err = self.pos_err
        if (self.best_rot_err is None) or (self.best_rot_err > self.rot_err):
            self.best_rot_err = self.rot_err
        if self._check_success():
            print(f'verygood pos_err:{self.pos_err} rot_err:{self.pos_err}')

        # reward items
        pos_err_reward, rot_err_reward, pose_err_reward = 0.0, 0.0, 0.0
        # panelty items
        action_penalty, vel_penalty = 0.0, 0.0

        # pos err
        if self.pos_err > self.init_pos_err:
            pos_err_reward -= 5.0
        if self.pos_err > pos_err_threshold:
            pos_err_reward += (5.0 * (1-np.tanh(self.pos_err)))
        else:
            pos_err_reward += (10.0 - np.log(0.1*self.pos_err+1e-7))
        
        # rot err
        if self.rot_err > self.init_rot_err:
            rot_err_reward -= 5.0
        if self.rot_err > rot_err_threshold:
            rot_err_reward += (5.0 * (1-np.tanh(0.5*self.rot_err)))
        else:
            rot_err_reward += (10.0 - np.log(0.1*self.rot_err+1e-7))
        
        # pose err
        if (self.rot_err < rot_err_threshold and self.pos_err < pos_err_threshold):
            pose_err_reward = 1000.0
        
        # action Penalty
        action_penalty = -0.1 * np.linalg.norm(action)

        # joint vel penalty
        vel_penalty = -0.1 * np.linalg.norm(self.get_robot_joint_velocities())

        # reward
        reward_value = (
            pos_err_reward + 
            rot_err_reward + 
            pose_err_reward + 
            action_penalty + 
            vel_penalty
        )

        self.total_steps = self.total_steps + 1
        self.writer.add_scalar("Error/Position", pos_err_reward, self.total_steps)
        self.writer.add_scalar("Error/Orientation", rot_err_reward, self.total_steps)
        self.writer.add_scalar("Error/ActionPenalty", action_penalty, self.total_steps)
        self.writer.add_scalar("Error/VelocityPenalty", vel_penalty, self.total_steps)

        self.reward_value += reward_value
        return reward_value

    def _check_success(self):
        '''
        自定义完成函数
        '''
        if self.pos_err < 0.01 and self.rot_err < 0.1:
            return True
        else:
            return False
        
    def _test_matrixs(self):
        # members
        print("\n=============================================================")
        T_te = self.T_tool_end
        print("T_te:")
        print(T_te)

        T_ew = self.get_robot_pose()
        print("T_ew:")
        print(T_ew)

        T_bw = self.get_base_pose()
        print("T_bw:")
        print(T_bw)

        T_cw = self.T_camera_world
        print("T_cw:")
        print(T_cw)

        T_gw = self.T_target_world
        print("T_gw:")
        print(T_gw)

        T_eb = self.get_robot_pose_base()
        print("T_eb:")
        print(T_eb)
        print("\n=============================================================")
        # Ttc
        T_tc = invert_homogeneous_matrix(T_cw) @ T_bw @ T_eb @ T_te
        print("T_tc:")
        print(T_tc)
        quat_t_c, pos_t_c = homogeneous_matrix_to_quaternion(T_tc)
        print("pose_tc:")
        print(quat_t_c)
        print(pos_t_c)
        quat_t_c, pos_t_c = homogeneous_matrix_to_quaternion(self.get_tool_pose())
        print("pose_gc_:")
        print(quat_t_c)
        print(pos_t_c)
        print("=============================================================")
        T_gc = invert_homogeneous_matrix(T_cw) @ T_gw
        print("T_gc:")
        print(T_gc)
        quat_g_c, pos_g_c = homogeneous_matrix_to_quaternion(T_gc)
        print("pose_gc:")
        print(quat_g_c)
        print(pos_g_c)
        quat_g_c, pos_g_c = homogeneous_matrix_to_quaternion(self.get_target_pose())
        print("pose_gc_:")
        print(quat_g_c)
        print(pos_g_c)
        print("=============================================================")

        error_info = calculate_pose_error(T_tc, T_gc, angle_unit='degrees')
        pos_err = error_info['translation_magnitude']
        rot_err = error_info['rotation_error_angle']
        print(f"pos_err:{pos_err} rot_err:{rot_err}")

        print("=============================================================")
        
class ReachEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        # 将robosuite的动作空间和观察空间转换成gym的
        low, high = env.action_spec
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            'achieved_goal': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),    # 当前时刻机械臂末端的 实际位置
            'desired_goal': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),     # 当前 episode 的 目标位置
            'observation': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),      # 当前时刻机械臂末端的 实际位置
        })
        

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs = self._process_obs(obs)
        info = {}
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # 修改 observation
        obs = self._process_obs(obs)

        return obs, reward, terminated, truncated, info

    def _process_obs(self, obs):
        # 当前时刻基座坐标系下机械臂末端的实际位姿
        # quat_e_b, pos_e_b = homogeneous_matrix_to_quaternion(self.unwrapped.get_robot_pose_base())
        joint_positions = self.unwrapped.get_robot_joint_positions()
        # 当前episode相机坐标系下机械臂工具的目标位姿
        quat_g_c, pos_g_c = homogeneous_matrix_to_quaternion(self.unwrapped.get_target_pose())
        # 当前时刻相机坐标系下机械臂工具的实际位姿
        quat_t_c, pos_t_c = homogeneous_matrix_to_quaternion(self.unwrapped.get_tool_pose())
        return {
            "achieved_goal": joint_positions,
            "desired_goal": np.concatenate([pos_g_c, quat_g_c]),
            "observation": np.concatenate([pos_t_c, quat_t_c]),
        }
    
    
        

def generate_env():
    controller_config = load_controller_config(default_controller="OSC_POSE")
    env = ReachEnv(controller_config=controller_config,seed=42)
    env = GymWrapper(env)
    env = ReachEnvWrapper(env)
    env = wrap_env(env, wrapper="gym")
    return env

# ===============================================
# |                 agent                       |
# ===============================================
# seed for reproducibility
set_seed(42)  # e.g. `set_seed(42)` for fixed seed

# define models (stochastic and deterministic models) using mixins
class Actor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-5, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.linear_layer_1 = nn.Linear(self.num_observations, 512)
        self.linear_layer_2 = nn.Linear(512, 512)
        # self.linear_layer_3 = nn.Linear(128, 128)
        self.action_layer = nn.Linear(512, self.num_actions)

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        x = F.relu(self.linear_layer_1(inputs["states"]))
        x = F.relu(self.linear_layer_2(x))
        # x = F.relu(self.linear_layer_3(x))
        
        return torch.tanh(self.action_layer(x)), self.log_std_parameter, {}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.linear_layer_1 = nn.Linear(self.num_observations + self.num_actions, 512)
        self.linear_layer_2 = nn.Linear(512, 512)
        # self.linear_layer_3 = nn.Linear(128, 128)
        self.linear_layer_4 = nn.Linear(512, 1)

    def compute(self, inputs, role):
        x = F.relu(self.linear_layer_1(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)))
        x = F.relu(self.linear_layer_2(x))
        # x = F.relu(self.linear_layer_3(x))
        return self.linear_layer_4(x), {}
    
def generate_agent(env):
    device = env.device
    # instantiate a memory as experience replay
    memory = RandomMemory(memory_size=1000000, num_envs=env.num_envs, device=device, replacement=False)

    # instantiate the agent's models (function approximators).
    # SAC requires 5 models, visit its documentation for more details
    # https://skrl.readthedocs.io/en/latest/api/agents/sac.html#models
    models = {}
    models["policy"] = Actor(env.observation_space, env.action_space, device)
    models["critic_1"] = Critic(env.observation_space, env.action_space, device)
    models["critic_2"] = Critic(env.observation_space, env.action_space, device)
    models["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
    models["target_critic_2"] = Critic(env.observation_space, env.action_space, device)

    # initialize models' parameters (weights and biases)
    for model in models.values():
        model.init_parameters(method_name="normal_", mean=0.0, std=0.1)

    # configure and instantiate the agent (visit its documentation to see all the options)
    # https://skrl.readthedocs.io/en/latest/api/agents/sac.html#configuration-and-hyperparameters
    cfg = SAC_DEFAULT_CONFIG.copy()
    cfg["gradient_steps"] = 1
    cfg["batch_size"] = 100
    cfg["discount_factor"] = 0.99
    cfg["polyak"] = 0.005
    cfg["actor_learning_rate"] = 1.0e-4
    cfg["critic_learning_rate"] = 1.0e-4
    from skrl.resources.schedulers.torch import KLAdaptiveLR
    cfg["learning_rate_scheduler"] = KLAdaptiveLR
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.01}
    from skrl.resources.preprocessors.torch import RunningStandardScaler
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    # cfg["grad_norm_clip"] = 1.0
    cfg["random_timesteps"] = 0
    cfg["learning_starts"] = 1000
    cfg["learn_entropy"] = True
    cfg["entropy_learning_rate"] = 1.0e-4
    # cfg["initial_entropy_value"] = 1.0
    # cfg["target_entropy"] = -3.0
    # cfg["target_entropy"] = -models["policy"].num_actions
    # cfg["mixed_precision"] = False
    # logging to TensorBoard and write checkpoints (in timesteps)
    cfg["experiment"]["write_interval"] = 75
    cfg["experiment"]["checkpoint_interval"] = 10000
    cfg["experiment"]["directory"] = "runs_reach/torch/reach"

    agent = SAC(models=models,
                memory=memory,
                cfg=cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)
    
    return agent

def train():
    # configure and instantiate the RL trainer
    cfg_trainer = {"timesteps": 100000, "headless": True}
    agent = generate_agent()
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])
    # start training
    trainer.train() 

def eval(model_path):
    test_env = generate_env()
    test_agent = generate_agent()
    test_agent.load(path=model_path)
    cfg_trainer = {"timesteps": 10000, "headless": False}
    test_trainer = SequentialTrainer(cfg=cfg_trainer, env=test_env, agents=[test_agent])
    test_trainer.eval()




















