
import torch
import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from envUtils import *
import sys

training_config = {
    # "composite_task": TASK,             # 任务名
    "total_timesteps": 10_000_000,      # 总时间步
    "policy": "MultiInputPolicy",       # 观测策略 MlpPolicy MultiInputPolicy
    "net_arch": [512,512,512],          # mlp结构
    "activation_fn": torch.nn.ReLU,     # 激活函数
    "learning_rate": 0.001,             # 学习率
    "batch_size": 128,                  #
    "target_update_interval": 5,        # 目标更新间隔
    "gamma": 0.99,                      # 
    "ent_coef": "auto",                 #
    "device": "cuda",                   # 设备
    "n_envs": 5,                        # 同时训练环境数
    "seed": 42,                         # 随机种子
}

controller_fpath = "./reachController.json"
env_config = {
    "robots": ["UR5e"],                 # 机器人
    "controller_configs": load_composite_controller_config(controller=controller_fpath),
    "has_renderer": False,              # 渲染
    "has_offscreen_renderer": False,    # 渲染
    "reward_shaping": True,             # 稀疏奖励
    "horizon": 200,                     # 每回合时间步
    "control_freq": 20,                 # 控制频率
    "seed": 42,                         # 随机种子
    "train_type": "pose",               # 训练类型
    "reset_policy": 2,                  # 重置策略
    "reward_scale": 1.0,                # 奖励放缩
    "use_object_obs": False,            # 障碍物
    "n_env": 1,                         # 环境编号
    "log_dir": "db/eval",
}

def eval(model_path, type):
    if model_path is None:
        print("Need ModelPath.")
        return 
    env_config["train_type"] = type            
    env_config["has_renderer"] = True
    env = make_env(env_config)
    # 构建模型
    model = SAC( 
        policy=training_config["policy"],
        env=env, 
        learning_rate=training_config["learning_rate"],
        batch_size=training_config["batch_size"],
        device=training_config["device"],
        gamma=training_config["gamma"],
        ent_coef=training_config["ent_coef"],
        policy_kwargs=dict(
            net_arch=training_config["net_arch"],
            activation_fn=training_config["activation_fn"],
        ),
        tensorboard_log=f"db/eval/train",
        seed=training_config["seed"],
        verbose=1,
    )
    
    state_size = env.unwrapped.sim.get_state().flatten().shape
    # 加载模型
    model = SAC.load(model_path, env=env)
    # 测试模型
    success = 0
    obs = env.reset()
    for i in range(10):
        obs = env.reset()
        obs = obs[0]
        for i in range(env_config["horizon"]):
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            if info['success']:
                success += 1
                break
            if terminated or truncated:
                obs = env.reset()
    success_rate = success / 10
    print(f"Success rate on Reach:", success_rate)
    
if __name__ == '__main__':
    eval(model_path=sys.argv[1], type=sys.argv[2])
    