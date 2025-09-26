# ===================numpy==========================
import numpy as np
# ===================robosuite======================
import robosuite as suite
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.tasks import ManipulationTask
from robosuite.models.arenas import TableArena
from robosuite.models.objects import CylinderObject
from robosuite.models.robots import Panda
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.controllers import load_part_controller_config
from robosuite.controllers import load_composite_controller_config
import xml.etree.ElementTree as ET
import robosuite.utils.transform_utils as T
from robosuite.utils.observables import Observable, sensor
from robosuite.wrappers import GymWrapper
# ===================user=========================
from .transUtils import *
from .shmUtils import SharedMemoryChannel, shared_memory_exists
# ===================torch========================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# ===================datatime=====================
from datetime import datetime
# ===================global variable==============
FinalPosErrThreshold=0.001                     
FinalRotErrThreshold=0.001

# ===============================================
# |               environment                   |
# ===============================================
class ReachEnv(ManipulationEnv):
    def __init__(self,
                 robots=["UR5e"],               # 机器人
                 controller_configs=None,        # 控制器
                 has_renderer=True,             # ui渲染
                 has_offscreen_renderer=False,
                 reward_shaping=True,           # 稀疏奖励/连续奖励
                 horizon=200,                   # 每回合时间步
                 control_freq=20,               # 控制频率
                 seed=42,                       # 随机种子
                 train_type="pose",             # 训练类型
                 reset_policy=2,                # 重置策略
                 reward_scale=1.0,              # 奖励放缩尺度
                 use_object_obs=False,          # 相机观察障碍物
                 n_env=1,
                 log_dir="db/default",
                 **kwargs):
        
        # 使用robosuite创建环境
        super().__init__(
            robots=robots,
            controller_configs=controller_configs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            use_camera_obs=False,
            control_freq=control_freq,
            horizon=horizon,
            gripper_types=None,
            **kwargs
        )
        
        self.reward_shaping=reward_shaping      
        self.seed=seed
        self.train_type = train_type
        self.reset_policy = reset_policy
        self.reward_scale = reward_scale
        self.use_object_obs = use_object_obs

        np.random.seed(self.seed) 
        # 初始化世界坐标系下相机位姿
        self.T_camera_world = generate_random_homogeneous_transform(
            translation_range=[(-0.5, 0.5), (-0.5, 0.5), (0.8, 1.3)],
            rotation_mode='euler',
            rotation_range=(-90, 90),
            seed=self.seed
        )
        # 初始化末端坐标系下工具位姿
        self.T_tool_end = generate_random_homogeneous_transform(
            translation_range=[(-0.1, 0.1), (-0.1, 0.1), (0.0, 0.1)],
            rotation_mode='euler',
            rotation_range=(-45, 45),
        )
        # 初始化世界坐标系下目标位姿
        self.pos_range = 0.3
        self.rot_range = 30
        self.T_target_world = generate_perturbed_transform(
            base_transform=self.get_base_pose_w(),
            translation_error_range=(-self.pos_range, self.pos_range),
            rotation_error_range=(-self.rot_range, self.rot_range),
            rotation_mode='euler',
            seed=self.seed,   
        )
        
        # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")    # 构造日志目录
        while shared_memory_exists(f"chatbus_{n_env}"):
            n_env += 1
        self.log_dir = f"{log_dir}/user/env_{n_env}"
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.pos_err_threshold=0.1                     
        self.final_pos_err_threshold = FinalPosErrThreshold
        self.rot_err_threshold=0.1
        self.final_rot_err_threshold = FinalRotErrThreshold
        self.channel = SharedMemoryChannel(f"chatbus_{n_env}")   #
        self.episodes_ctr = 0       # 计数器
        self.epochs_ctr = 0
        self.steps_ctr = 0
        self.init_pos_err = None    #
        self.init_rot_err = None
        self.best_pos_err = None
        self.best_rot_err = None
        self.cur_best_pos_err = 1.0
        self.cur_best_rot_err = 1.0
        self.episode_reward = 0.0
        
        self.reset_episodes_num = 5
        self.pos_tracker = SuccessTracker(10)
        self.rot_tracker = SuccessTracker(10)
        self.pos_episode_succ = 0
        self.rot_episode_succ = 0

        self.joint_low = np.array([-0.470 - 1.0, -1.735, 2.480, -0.785, 1.590, -0.5])
        self.joint_high = np.array([-0.470 + 1.0, -1.735, 2.480, -0.785, 1.590, 0.5])
        self.margin = 0.1
        # 给关节上下限预留 margin，避免到极限
        self.joint_low = self.joint_low + self.margin * (self.joint_high - self.joint_low)
        self.joint_high = self.joint_high - self.margin * (self.joint_high - self.joint_low)

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
    
    def get_joint_pos(self, verbose=False):
        '''
        获取机械臂关节位置
        '''
        joint_positions = self.robots[0]._joint_positions
        if verbose:
            print(f"机械臂关节位置:\n{joint_positions}")
        return joint_positions

    def get_joint_vel(self, verbose=False):
        '''
        获取机械臂关节速度
        '''
        joint_velocities = self.robots[0]._joint_velocities
        if verbose:
            print(f"机械臂关节速度:\n{joint_velocities}")
        return joint_velocities

    def get_end_pose_w(self, verbose=False):
        """
        获取世界坐标系下机械臂末端tcp的位姿
        """
        end_name = self.robots[0].robot_model.eef_name['right']
        body_id = self.sim.model.body_name2id(end_name)
        pos = self.sim.data.body_xpos[body_id]
        quat = T.convert_quat(self.sim.data.body_xquat[body_id], to="xyzw")
        trans = quaternion_to_homogeneous_matrix(quat, pos)
        if verbose:
            print(f"机械臂末端名称:{end_name}")
            print(f"机械臂末端序号:{body_id}")
            print(f"世界坐标系末端位置:\n{pos}")
            print(f"世界坐标系末端姿态:\n{quat}")
            print(f"世界坐标系末端变换:\n{trans}")
        return trans
    
    def get_base_pose_w(self, verbose=False):
        """
        获取世界坐标系下机械臂基座的位姿
        """
        body_id = self.sim.model.body_name2id('robot0_base')
        pos = self.sim.data.body_xpos[body_id]
        quat = T.convert_quat(self.sim.data.body_xquat[body_id], to="xyzw")
        trans = quaternion_to_homogeneous_matrix(quat, pos)
        if verbose:
            print(f"机械臂基座序号:{body_id}")
            print(f"世界坐标系基座位置:\n{pos}")
            print(f"世界坐标系基座姿态:\n{quat}")
            print(f"世界坐标系基座变换:\n{trans}")
        return trans
    
    def get_end_pose_b(self, verbose=False):
        """
        获取基座坐标系下机械臂末端tcp的位姿
        """
        T_end_base = compose_transforms(
            invert_homogeneous_matrix(self.get_base_pose_w(verbose=verbose)), 
            self.get_end_pose_w(verbose=verbose)
        )
        if verbose:
            print(f"基座坐标系末端变换:\n{T_end_base}")
        return T_end_base
    
    def get_tool_pose_c(self, verbose=False):
        """
        获取相机坐标系下机械臂末端工具的位姿
        """
        T_end_world = self.get_end_pose_w(verbose=verbose)
        T_tool_camera = compose_transforms(invert_homogeneous_matrix(self.T_camera_world), T_end_world, self.T_tool_end)
        if verbose:
            print(f"相机坐标系工具变换:{T_tool_camera}")
        return T_tool_camera

    def get_goal_pose_c(self, verbose=False):
        """
        获取相机坐标系下目标位姿
        """
        T_target_camera = compose_transforms(invert_homogeneous_matrix(self.T_camera_world), self.T_target_world)
        if verbose:
            print(f"相机坐标系目标变换:{T_target_camera}")
        return T_target_camera
    
    def update_goal_pose(self, verbose=False):
        """
        设置新的目标位姿
        """
        T_target_world = self.T_target_world
        T_base_world = self.get_base_pose_w(verbose=verbose)
        _, pos_b_w = homogeneous_matrix_to_quaternion(T_base_world)
        i = 0
        while True:
            i += 1
            self.T_target_world = generate_random_homogeneous_transform(
                translation_range=[
                    (pos_b_w[0]-0.8, pos_b_w[0]+0.8), 
                    (pos_b_w[1]-0.8, pos_b_w[1]+0.8), 
                    (pos_b_w[2]+0.1, pos_b_w[2]+0.9)
                ],
                rotation_mode='euler',
                rotation_range=(-30, 30),
            )
            _, pos_g_w = homogeneous_matrix_to_quaternion(self.T_target_world)
            if is_point_in_cylinder(
                base_center=pos_b_w, 
                axis_vector=np.array([0, 0, 1]),
                radius=0.2, 
                height=1.0, 
                point=pos_g_w,
            ):
                continue
            # self.T_target_world = generate_perturbed_transform(
            #     base_transform=self.get_end_pose_w(verbose=verbose),
            #     translation_error_range=(-self.pos_range, self.pos_range),
            #     rotation_error_range=(-self.rot_range, self.rot_range),
            #     rotation_mode='euler',
            # )
            T_target_camera = self.get_goal_pose_c(verbose=verbose)
            T_tool_camera = self.get_tool_pose_c(verbose=verbose)
            error_info = calculate_pose_error(T_target_camera, T_tool_camera, angle_unit='radians')
            self.init_pos_err = error_info['translation_magnitude']
            self.init_rot_err = error_info['rotation_error_angle']
            if (self.init_pos_err > self.pos_err_threshold) and (self.init_rot_err > self.rot_err_threshold):
                break
            if i>100:
                print(f"get goal out of range:{self.steps_ctr}")
                break
        if verbose:
            print(f"原世界坐标系目标变换:\n{T_target_world}")
            print(f"新世界坐标系目标变换:\n{self.T_target_world}")
            print(f"变换位置初始误差:{self.init_pos_err}")
            print(f"变换角度初始误差:{self.init_rot_err}")

    def update_camera_pose(self, verbose=False):
        """
        设置新的相机位姿
        """
        T_camera_world = self.T_camera_world
        i = 0
        while True:
            i += 1
            self.T_camera_world = generate_perturbed_transform(
                base_transform=self.get_base_pose_w(),
                translation_error_range=(0.4, 0.6),
                rotation_error_range=(-45, 45),
                rotation_mode='euler',
            )
            T_target_camera = self.get_goal_pose_c(verbose=verbose)
            T_tool_camera = self.get_tool_pose_c(verbose=verbose)
            error_info = calculate_pose_error(T_target_camera, T_tool_camera, angle_unit='radians')
            self.init_pos_err = error_info['translation_magnitude']
            self.init_rot_err = error_info['rotation_error_angle']
            if (self.init_pos_err > self.pos_err_threshold) and (self.init_rot_err > self.rot_err_threshold):
                break
            if i>100:
                print(f"get camera out of range:{self.steps_ctr}")
                break
        if verbose:
            print(f"原世界坐标系相机变换:\n{T_camera_world}")
            print(f"新世界坐标系相机变换:\n{self.T_camera_world}")
            print(f"变换位置初始误差:{self.init_pos_err}")
            print(f"变换角度初始误差:{self.init_rot_err}")

    def update_tool_pose(self, verbose=False):
        self.T_tool_end = generate_random_homogeneous_transform(
            translation_range=[(-0.1, 0.1), (-0.1, 0.1), (0.0, 0.1)],
            rotation_mode='euler',
            rotation_range=(-45, 45),
        )

    def update_tf(self):
        """
        将当前坐标树传给ros2包
        """
        self.get_joint_pos(verbose=False)
        self.get_joint_vel(verbose=False)
        T_bw = self.get_base_pose_w(verbose=False)
        T_gc = self.get_goal_pose_c(verbose=False)
        T_eb = self.get_end_pose_b(verbose=False)
        self.channel.send({
            'T_base_world': T_bw,
            'T_end_base': T_eb,
            'T_tool_end': self.T_tool_end,
            'T_camera_world': self.T_camera_world,
            'T_goal_camera': T_gc,
        })

    def random_qpos(self):
        return np.random.uniform(self.joint_low, self.joint_high)

    def write_tensorboard(self, str, y_value):
        self.writer.add_scalar(str, y_value, self.steps_ctr)

    @staticmethod
    def cal_pos_reward(error):
        # pos_err_reward = 0.0
        pos_err = error["pos"]
        return combined_reward(pos_err)
        # if pos_err > error["pos_init"]:
        #     pos_err_reward -= 5.0
        # elif pos_err > FinalPosErrThreshold:
        #     pos_err_reward -= 2.0 * pos_err
        # else:
        #     pos_err_reward += 10.0 * (1 - np.tanh(10.0 * pos_err))
        # return pos_err_reward

    @staticmethod
    def cal_rot_reward(error):
        # rot_err_reward = 0.0
        rot_err = error["rot"]
        return combined_reward(rot_err)
        # if rot_err > error["rot_init"]:
        #     rot_err_reward -= 5.0
        # elif rot_err > FinalRotErrThreshold:
        #     rot_err_reward -= 2.0 * rot_err
        # else:
        #     rot_err_reward += 10.0 * (1 - np.tanh(10.0 * rot_err))
        # return rot_err_reward 

    def cal_reward_value(self, action, error):
        # reward items
        pos_err_reward, rot_err_reward, pose_err_reward, action_penalty, vel_penalty  = self.cal_reward_value_masac()

        # reward
        reward_value = (
            pos_err_reward + rot_err_reward + pose_err_reward + 
            action_penalty + vel_penalty
        )

        self.writer.add_scalar("Reward/Position_Err_Threshold", self.pos_err_threshold, self.steps_ctr)
        self.writer.add_scalar("Reward/Rotation_Err_Threshold", self.rot_err_threshold, self.steps_ctr)
        self.writer.add_scalar("Reward/Reward_PosErr", pos_err_reward, self.steps_ctr)
        self.writer.add_scalar("Reward/Reward_RotErr", rot_err_reward, self.steps_ctr)
        self.writer.add_scalar("Reward/Reward_PoseErr", pose_err_reward, self.steps_ctr)
        self.writer.add_scalar("Reward/Penalty_Action", action_penalty, self.steps_ctr)
        self.writer.add_scalar("Reward/Penalty_Velocity", vel_penalty, self.steps_ctr)
        self.writer.add_scalar("Reward/InstaneousReward", reward_value, self.steps_ctr)
        self.writer.add_scalar("Error/Position", self.pos_err, self.steps_ctr)
        self.writer.add_scalar("Error/Orientation", self.rot_err, self.steps_ctr)
        self.writer.add_scalar("Error/EpisodeBestPosErr", self.best_pos_err, self.steps_ctr)
        self.writer.add_scalar("Error/EpisodeBestRotErr", self.best_rot_err, self.steps_ctr)
        self.writer.add_scalar("Error/BestPosErr", self.cur_best_pos_err, self.steps_ctr)
        self.writer.add_scalar("Error/BestRotErr", self.cur_best_rot_err, self.steps_ctr)
        self.writer.add_scalar("Info/PosSuccessRate", self.pos_tracker.success_rate(), self.steps_ctr)
        self.writer.add_scalar("Info/RotSuccessRate", self.rot_tracker.success_rate(), self.steps_ctr)
        return reward_value

    def cal_reward_value_masac(self):
        # 奖励和惩罚项
        pos_err_reward, rot_err_reward, pose_err_reward = 0.0, 0.0, 0.0
        action_penalty, vel_penalty = 0.0, 0.0


        # if self.pos_err > self.init_pos_err:
        #     pos_err_reward -= 5.0
        # elif self.pos_err > self.pos_err_threshold:
        #     pos_err_reward -= 2.0 * self.pos_err
        # else:
        #     pos_err_reward += 10.0 * (1 - np.tanh(10.0 * self.pos_err))
        #     self.pos_episode_succ += 1
        pos_err_reward = combined_reward(self.pos_err)
            
        # if self.rot_err > self.init_rot_err:
        #     rot_err_reward -= 5.0
        # elif self.rot_err > self.rot_err_threshold:
        #     rot_err_reward -= 2.0 * self.rot_err
        # else:
        #     rot_err_reward += 10.0 * (1 - np.tanh(10.0 * self.rot_err))
        #     self.rot_episode_succ += 1
        rot_err_reward = combined_reward(self.rot_err)
        
        # if (self.rot_err < self.rot_err_threshold and self.pos_err < self.pos_err_threshold):
        #     pose_err_reward = 20.0

        # action Penalty
        action_penalty = -0.001 * np.linalg.norm(self.action)
        # joint vel penalty
        # vel_penalty = -0.01 * np.linalg.norm(self.get_joint_vel())
        # reward
        return pos_err_reward, rot_err_reward, pose_err_reward, action_penalty, vel_penalty

    def reset(self, seed=None, options=None):
        """
        自定义reset方法
        """
        # 调用父类reset
        obs_dict = super().reset()
        # self.robots[0].set_joint_positions(self.random_qpos())
        self.sim.data.qpos[:] = self.random_qpos()
        self.sim.forward()
        # 重置参数
        if self.episodes_ctr > 1:
            self.writer.add_scalar("Reward/TotalReward", self.episode_reward, self.steps_ctr)
        self.episodes_ctr = self.episodes_ctr + 1
        if (self.best_pos_err is not None) and (self.best_pos_err < self.cur_best_pos_err):
            self.cur_best_pos_err = self.best_pos_err
        if (self.best_rot_err is not None) and (self.best_rot_err < self.cur_best_rot_err):
            self.cur_best_rot_err = self.best_rot_err

        if self.best_pos_err is not None:
            self.pos_tracker.add_result((self.pos_episode_succ > 3))
            if self.best_pos_err < self.cur_best_pos_err:
                self.cur_best_pos_err = self.best_pos_err
            if (self.pos_tracker.success_rate() > 0.7) and (self.final_pos_err_threshold < self.pos_err_threshold) and (self.episodes_ctr > 50):
                self.pos_err_threshold *= 0.95
        if self.best_rot_err is not None:
            self.rot_tracker.add_result((self.rot_episode_succ>3))
            if self.best_rot_err < self.cur_best_rot_err:
                self.cur_best_rot_err = self.best_rot_err
            if (self.rot_tracker.success_rate() > 0.7) and (self.final_rot_err_threshold < self.rot_err_threshold) and (self.episodes_ctr > 50):
                self.rot_err_threshold *= 0.95
        
        self.best_pos_err = None
        self.best_rot_err = None
        self.episode_reward = 0.0
        self.pos_episode_succ = 0
        self.rot_episode_succ = 0
        # 更新
        if self.reset_policy == 1:
            self.update_goal_pose()
            if self.episodes_ctr % self.reset_episodes_num == 0:
                self.update_camera_pose()
        elif self.reset_policy == 2:
            self.update_camera_pose()
            self.update_tool_pose()
            if self.episodes_ctr % self.reset_episodes_num == 0:
                self.update_goal_pose()

        return obs_dict 

    def reward(self, action=None):
        '''
        自定义奖励函数
        '''
        self.update_tf()
        self.steps_ctr = self.steps_ctr + 1

        error_info = calculate_pose_error(
            self.get_tool_pose_c(), 
            self.get_goal_pose_c(), 
            angle_unit='radians'
        )
        self.pos_err = error_info['translation_magnitude']
        self.rot_err = error_info['rotation_error_angle']
        if (self.best_pos_err is None) or (self.best_pos_err > self.pos_err):
            self.best_pos_err = self.pos_err
        if (self.best_rot_err is None) or (self.best_rot_err > self.rot_err):
            self.best_rot_err = self.rot_err

        reward_value = 0.0
        if self._check_success():
            reward_value = 5.0
        else:
            reward_value = -5.0

        if self.reward_shaping:
            self.action=action
            reward_value = self.cal_reward_value(action=action, error=error_info)            
        
        self.episode_reward += reward_value
        return reward_value

    def _check_success(self):
        '''
        自定义完成函数
        '''
        if self.pos_err < self.pos_err_threshold and self.rot_err < self.rot_err_threshold:
            return True
        else:
            return False
    
    def test(self):
        verbose = True
        self.get_joint_pos(verbose=verbose)
        self.get_joint_vel(verbose=verbose)
        T_bw = self.get_base_pose_w(verbose=verbose)
        T_ew = self.get_end_pose_w(verbose=verbose)
        T_gc = self.get_goal_pose_c(verbose=verbose)
        T_tc = self.get_tool_pose_c(verbose=verbose)
        T_eb = self.get_end_pose_b(verbose=verbose)
        print(self.sim.model.joint_names)
        print(self.sim.model.body_names)
        print(self.sim.model.site_names)

