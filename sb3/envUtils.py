from reachEnvSb3 import *
import types
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import GymWrapper
from stable_baselines3.common.callbacks import BaseCallback
# ===================gym=========================
import gym

class StopTrainingOnSuccessThreshold(BaseCallback):
    def __init__(self, success_threshold=0.8, eval_env=None, check_freq=5000, n_eval_episodes=5, verbose=1):
        super().__init__(verbose)
        self.success_threshold = success_threshold
        self.eval_env = eval_env
        self.check_freq = check_freq
        self.n_eval_episodes = n_eval_episodes
        self.success_history = []

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True

        successes = []
        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()  
            if isinstance(obs, tuple):
                obs = obs[0]
            dones_table = np.full(self.eval_env.num_envs, False)
            while not all(dones_table):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, infos = self.eval_env.step(action)
                for idx, info in enumerate(infos):
                    if info["success"]==True and dones_table[idx]==False:
                        successes.append(1)
                        dones_table[idx] = True   
                    if dones_table[idx]==False and done[idx]:
                        successes.append(0)
                        dones_table[idx] = True

        avg_success = np.mean(successes)
        self.logger.record("eval/success_rate", avg_success)
        self.success_history.append(avg_success)

        if self.verbose:
            print(f"[Callback] Success: {successes}")
            print(f"[Callback] Success rate over last {self.n_eval_episodes} episodes: {avg_success:.2f}")

        if avg_success >= self.success_threshold:
            print(f"[Callback] Success threshold of {self.success_threshold} reached. Stopping training.")
            return False  # Returning False stops training

        return True
    
class ReachEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        # 将robosuite的动作空间和观察空间转换成gym的
        low, high = env.action_spec
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            'achieved_goal': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(env.robots[0].dof,), dtype=np.float32),    
            'desired_goal': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),    
            'observation': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),      
        })

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
        return {
            "achieved_goal": joint_positions,
            "desired_goal": np.concatenate([pos_g_c, quat_g_c]),
            "observation": np.concatenate([pos_t_c, quat_t_c]),
        }

def make_env(config, n_env=1):
    config['n_env']=n_env
    config['seed']=config['seed']+n_env
    env = ReachEnv(**config)
    env = GymWrapper(env)
    env = ReachEnvWrapper(env)
    return env