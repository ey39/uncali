import datetime
from ray import train, tune, air
from ray.rllib.algorithms.sac import SACConfig
from common.envUtils import *

TASK="Reach_"
experiment_name = TASK + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
LOGDIR=f"/home/ey/rl/src/rlreach2/rlreach/ray/db/ray_results/{experiment_name}"

config = (
    SACConfig()
    .environment(
        env=ReachEnvGym,
        env_config={"log_dir": LOGDIR},        
    )
    .training(
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
            "type": "EpisodeReplayBuffer",
            "capacity": 1000000,
            "learning_starts": 1000,
            # HER 专用参数
            "replay_mode": "independent",
            "replay_sequence_length": 1,
            "replay_burn_in": 0,
            "replay_zero_init_states": False,
            "storage_unit": "episodes",
            # 关键：HER wrapper 配置
            "wrap_buffer": True,
            "wrapped_buffer": {
                "type": "HindsightExperienceReplayBuffer",
                "replay_mode": "independent",
                "her_strategy": "future",      # 可选: future, final, episode
                "replay_k": 4,                 # 每个 transition 生成多少个 HER 样本
                "goal_fn": None,               # 你可以自定义 goal extraction function
            },
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
        num_gpus=0.25,      # 或 0.25 视机器配置
        num_cpus_per_worker=1,
        num_learner_workers=1,
    )
    .framework("torch")
    .reporting(
        metrics_num_episodes_for_smoothing=5,
        min_sample_timesteps_per_iteration=1000,
    )
    .evaluation(
        evaluation_interval=1,
        evaluation_num_env_runners=1,
        evaluation_config={"seed": 42},
    )
    .env_runners(
        num_env_runners=6,             # 进程数量
        num_envs_per_env_runner=1,     # 环境数量
        # gym_env_vectorize_mode="ASYNC"
    )
)


tunner = tune.Tuner(
    trainable=config.algo_class,
    param_space=config,
    run_config=train.RunConfig(
        name="reach",
        storage_path=LOGDIR,
        log_to_file=True,
        checkpoint_config=air.CheckpointConfig(
            checkpoint_frequency=10,
            checkpoint_at_end=True,
        ),
        stop={"evaluation/env_runners/episode_return_mean": 8000.0}
    ),
)

results = tunner.fit()