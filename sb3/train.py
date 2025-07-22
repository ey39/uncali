import datetime
import wandb
from wandb.integration.sb3 import WandbCallback
import datetime
import torch
import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from envUtils import *

TASK="Reach"
experiment_name = TASK + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.environ["WANDB_MODE"]="offline"

training_config = {
    "composite_task": TASK,             # 任务名
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

controller_fpath = "/home/ey/rl/src/rlreach2/rlreach/sb3/reachController.json"
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
}

if __name__ == '__main__':
    # 数据库
    run = wandb.init(
        project="Reach-rl",
        group=experiment_name,
        config=training_config,
        sync_tensorboard=True,
        save_code=True,
        dir=f"db/wandb/{experiment_name}"
    )
    # 构建环境
    vec_env = SubprocVecEnv([lambda i=i: make_env(env_config,n_env=i) for i in range(training_config["n_envs"])])
    # 构建模型
    model = SAC( 
        policy=training_config["policy"],
        env=vec_env, 
        learning_rate=training_config["learning_rate"],
        batch_size=training_config["batch_size"],
        device=training_config["device"],
        gamma=training_config["gamma"],
        ent_coef=training_config["ent_coef"],
        policy_kwargs=dict(
            net_arch=training_config["net_arch"],
            activation_fn=training_config["activation_fn"],
        ),
        tensorboard_log=f"db/train/{experiment_name}/{run.id}",
        seed=training_config["seed"],
        verbose=1,
    )
    # 训练模型
    model.learn(
        total_timesteps=training_config["total_timesteps"], 
        log_interval=1,
        progress_bar=True,
        callback=[
            WandbCallback(
                model_save_path=f"db/models/{experiment_name}/{run.id}",
                model_save_freq=10_000,
                verbose=2,
            ), 
            EvalCallback(
                eval_env=vec_env,
                best_model_save_path=f"db/models/{experiment_name}/{run.id}",
                eval_freq=500_000,
                deterministic=True,
                render=False,
            ),
            StopTrainingOnSuccessThreshold(
                success_threshold=0.8,
                eval_env=vec_env,
                check_freq=500_000,
                n_eval_episodes=5,
                verbose=0
            )],
    )
    run.finish()
    # 验证训练效果
    env_config["has_renderer"] = True
    env = make_env(env_config)
    state_size = env.unwrapped.sim.get_state().flatten().shape
    # 加载模型
    model = SAC.load(f"db/models/{experiment_name}/{run.id}/best_model.zip", env=env)
    # 测试模型
    success = 0
    obs = env.reset()
    for i in range(10):
        obs = env.reset()
        obs = obs[0]
        for i in range(env_config["horizon"]):
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            if reward == 1:
                success += 1
                break
            if done:
                obs = env.reset()
    success_rate = success / 10
    print(f"Success rate on Reach:", success_rate)
    