from src.rlkit_ppo import rlkit
from env.pigeon_gym import PigeonEnv3Joints

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.ppo.ppo_env_replay_buffer import PPOEnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.ppo.ppo_path_collector import PPOMdpPathCollector
from rlkit.torch.ppo.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.ppo.ppo import PPOTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.ppo.ppo_torch_batch_rl_algorithm import PPOTorchBatchRLAlgorithm

import torch


def experiment(variant):
    torch.autograd.set_detect_anomaly(True)
    expl_env = PigeonEnv3Joints()
    eval_env = PigeonEnv3Joints()
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    vf = FlattenMlp(
        input_size=obs_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )
    eval_policy = MakeDeterministic(policy)
    eval_step_collector = PPOMdpPathCollector(
        eval_env,
        eval_policy,
        calculate_advantages=False
    )
    expl_step_collector = PPOMdpPathCollector(
        expl_env,
        policy,
        calculate_advantages=True,
        vf=vf,
        gae_lambda=0.97,
        discount=0.995,
    )
    replay_buffer = PPOEnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    trainer = PPOTrainer(
        env=eval_env,
        policy=policy,
        vf=vf,
        **variant['trainer_kwargs']
    )
    algorithm = PPOTorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_step_collector,
        evaluation_data_collector=eval_step_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()




if __name__ == "__main__":
    # noinspection PyTypeChecker
    T = 2048
    max_ep_len = 1000
    epochs = 10
    minibatch_size = 64

    variant = dict(
        algorithm="PPO",
        version="normal",
        layer_size=64,
        replay_buffer_size=T,
        algorithm_kwargs=dict(
            num_iter=int(1e6 // T),
            num_eval_steps_per_epoch=max_ep_len,
            num_trains_per_train_loop=T // minibatch_size * epochs,
            num_expl_steps_per_train_loop=T,
            min_num_steps_before_training=0,
            max_path_length=max_ep_len,
            minibatch_size=minibatch_size,
        ),
        trainer_kwargs=dict(
            epsilon=0.2,
            reward_scale=1.0,
            lr=3e-4,
        ),
    )
    setup_logger('pigeon_3_joints', variant=variant)
    #ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)
