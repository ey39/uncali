from ray.rllib.env.multi_agent_env import MultiAgentEnv
import gymnasium as gym
import numpy as np
from ray.tune.registry import register_env
from common.envUtils import *
import random


class MultiAgentReachEnv(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        has_renderer = False
        self.env_core = make_reach_her_env_masac(log_dir=config.get("log_dir"), has_renderer=has_renderer)
        
        # multi agent id
        self.possible_agents = ['agent_1', 'agent_2']
        self.agents = self.possible_agents = ['agent_1', 'agent_2']

        # multi agent obs_space & act_space
        self.observation_spaces = {
            "agent_1": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
            "agent_2": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
        }
        self.action_spaces = {
            "agent_1": gym.spaces.Box(low=-0.5, high=0.5, shape=(3,), dtype=np.float32),
            "agent_2": gym.spaces.Box(low=-0.5, high=0.5, shape=(3,), dtype=np.float32),
        }
    
    def get_action_space(self, agent_id):
        return self.action_spaces[agent_id]

    def get_observation_space(self, agent_id):
        return self.observation_spaces[agent_id]
    
    def reset(self, *, seed=None, options=None):
        self.env_core.reset()
        return {
            "agent_1": np.zeros(self.observation_spaces["agent_1"].shape, dtype=self.observation_spaces["agent_1"].dtype),
            "agent_2": np.zeros(self.observation_spaces["agent_2"].shape, dtype=self.observation_spaces["agent_2"].dtype),
        }, {'success': False}
    
    def step(self, action_dict):
        action = np.concatenate([
            action_dict["agent_1"], 
            action_dict["agent_2"],
        ])
        observation, reward, terminated, truncated, info = self.env_core.step(action)
        pos_err_reward, rot_err_reward, pose_err_reward, action_penalty, vel_penalty = self.env_core._get_reward_masac()

        observations = {
            "agent_1": self.env_core._process_obs_masac(obs=observation, obs_type='pos'),
            "agent_2": self.env_core._process_obs_masac(obs=observation, obs_type='rot'),
        }
        rewards = {
            "agent_1": pos_err_reward + pose_err_reward + action_penalty + vel_penalty,
            "agent_2": rot_err_reward + pose_err_reward + action_penalty + vel_penalty,
        }

        self.env_core.unwrapped.write_tensorboard("Reward/EnvOriginReward", reward)
        self.env_core.unwrapped.write_tensorboard("Reward/PosProcessReward", pos_err_reward)
        self.env_core.unwrapped.write_tensorboard("Reward/RotProcessReward", rot_err_reward)
        self.env_core.unwrapped.write_tensorboard("Reward/PosAgentReward", rewards["agent_1"])
        self.env_core.unwrapped.write_tensorboard("Reward/RotAgentReward", rewards["agent_2"])

        terminateds = {
            "agent_1": terminated, "agent_2": terminated,
        }

        truncateds = {
            "agent_1": truncated, "agent_2": truncated,
        }
        terminateds["__all__"] = all(terminateds.values())
        truncateds["__all__"] = any(truncateds.values())

        infos = {
            "agent_1": info, "agent_2": info,
        }

        return observations, rewards, terminateds, truncateds, infos
    
register_env("MultiReach-v0", MultiAgentReachEnv)

from datetime import datetime
import gymnasium as gym
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
import numpy as np
from pathlib import Path

TASK="MultiAgentReach_"
experiment_name = TASK + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOGDIR=f"/home/ey/rl/src/rlreach2/rlreach/ray/db/ray_results/{experiment_name}"

def policy_mapping_fn(agent_id, *args, **kwargs):
    if agent_id == "agent_1":
        p = "policy_1"
    elif agent_id == "agent_2":
        p = "policy_2"
    else:
        raise ValueError(f"Unknown agent_id: {agent_id}")
    # print(f"[mapping] {agent_id} -> {p}")
    return p

config = (
    SACConfig()
    .environment(
        env="MultiReach-v0",
        env_config={"log_dir": LOGDIR},        
    )
    .multi_agent(
        policies={
            "policy_1": (None, gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32), gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32), {}),
            "policy_2": (None, gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32), gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32), {}),
        },
        policy_mapping_fn=policy_mapping_fn,
        policies_to_train=["policy_1", "policy_2",]
    )
    .rl_module(
        rl_module_spec=MultiRLModuleSpec(
            rl_module_specs={
                "policy_1": RLModuleSpec(
                    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
                    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
                ),
                "policy_2": RLModuleSpec(
                    observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
                    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
                ),
            }
        )
    )
    .training(
        twin_q=True,
        initial_alpha=0.2,
        actor_lr=1e-4,
        critic_lr=1e-4,
        alpha_lr=1e-4,
        target_entropy="auto",
        n_step=1,
        tau=0.005,
        train_batch_size=128,
        target_network_update_freq=1,
        replay_buffer_config={
            "type": "MultiAgentEpisodeReplayBuffer",
            "capacity": 1000000,
            "learning_starts": 1000,
            "replay_batch_size": 200,
        },
        num_steps_sampled_before_learning_starts=1000,
        model={
            "fcnet_hiddens": [512, 512],
            "fcnet_activation": "relu",
            "post_fcnet_hiddens": [],
            "post_fcnet_activation": None,
            "post_fcnet_weights_initializer": "orthogonal_",
            "post_fcnet_weights_initializer_config": {"gain": 0.01},
        },
    )
    .resources(
        # num_cpus=10,
    #     num_gpus=0.25,      # 或 0.25 视机器配置
        num_cpus_per_worker=10,
    #     num_learner_workers=1,
    )
    .framework("torch")
    .reporting(
        metrics_num_episodes_for_smoothing=5,
        min_sample_timesteps_per_iteration=1000,
    )
    .evaluation(
        evaluation_interval=1,
        evaluation_num_env_runners=1,
        evaluation_duration=2,
        evaluation_config={"seed": 42},
    )
    .env_runners(
        rollout_fragment_length=200,
    )
    # .env_runners(
    #     num_env_runners=6,             # 进程数量
    #     num_envs_per_env_runner=1,     # 环境数量
    #     # gym_env_vectorize_mode="ASYNC"
    # )
)

from ray import train, tune, air

def _on_sample_end(env_runner, metrics_logger, samples, **kwargs):
    if not hasattr(_on_sample_end, "count"):
        _on_sample_end.count = 0  # 初始化静态变量
    agent_1_episode = samples[0].agent_episodes["agent_1"]
    agent_2_episode = samples[0].agent_episodes["agent_2"]
    # print(f"samples len:{len(samples)}")
    if len(agent_1_episode) == 200:
        if (random.random() < 0.5):
            err = {
                "pos_init": calculate_pos_error(
                    np.array(agent_1_episode.observations[1][:3]), 
                    np.array(agent_1_episode.observations[1][-3:])
                ),
                "rot_init": calculate_rot_error(
                    np.array(agent_2_episode.observations[1][:3]), 
                    np.array(agent_2_episode.observations[1][-3:]),
                    angle_unit='radians'
                ),
            }
        # pos
            for i in range(len(agent_1_episode.observations)):
                agent_1_episode.observations[i][:3] = agent_1_episode.observations[-1][-3:]
                err["pos"] = calculate_pos_error(
                    np.array(agent_1_episode.observations[i][:3]), 
                    np.array(agent_1_episode.observations[i][-3:])
                )
                agent_1_episode.rewards[i-1] = ReachEnv.cal_pos_reward(error=err)
        # rot
            # for i in range(len(agent_2_episode.observations))[1:]:
                agent_2_episode.observations[i][:3] = agent_2_episode.observations[-1][-3:]
                err["rot"] = calculate_rot_error(
                    np.array(agent_2_episode.observations[i][:3]), 
                    np.array(agent_2_episode.observations[i][-3:]),
                    angle_unit='radians'
                )
                agent_2_episode.rewards[i-1] = ReachEnv.cal_rot_reward(error=err)
    # for i in range(len(ep1.observations)):
    #     if ep1.observations[i][0] == 0 and  ep1.observations[i][1] == 0:
    #         print(_on_sample_end.count)
    #         _on_sample_end.count = 0
    #     else:
    #         _on_sample_end.count += 1
    #     print(f"ep1 obs:{ep1.observations[i]}")
    # for i in range(len(ep1.rewards)):
    #     print(f"ep1 r:{ep1.rewards[i]}")
    # for i in range(len(ep1.actions)):
    #     print(f"ep1 act:{ep1.actions[i]}")

    
config.callbacks(on_sample_end=_on_sample_end)

tuner = tune.Tuner(
    trainable=config.algo_class,
    param_space=config,
    run_config=train.RunConfig(
        name="multi_agent_reach",
        storage_path=LOGDIR,
        log_to_file=True,
        checkpoint_config=air.CheckpointConfig(
            checkpoint_frequency=100,
            checkpoint_at_end=True,
        ),
        stop={"env_runners/num_env_steps_sampled_lifetime": 10_000_000},
        # stop={"evaluation/env_runners/episode_return_mean": 1800000.0},
        # callbacks=[MyCheckpointCallback()],  # 挂上去
    )
)

results = tuner.fit()

