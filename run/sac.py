from gym_env.pigeon_gym import PigeonEnv3Joints
from gym_env.pigeon_gym_retinal import PigeonRetinalEnv
import sys
sys.path.append('src/rlkit_ppo')

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

import argparse

def experiment(expl_env, eval_env, variant, args):
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', '--environment', type=str,
                        default = "PigeonEnv3Joints",
                        help = 'name of environment: \n' + \
                               '  PigeonEnv3Joints \n' + \
                               '  PigeonRetinalEnv')
    parser.add_argument('-bs', '--body_speed', type=float, default=1.0,
                        help='pigeon body speed')
    parser.add_argument('-rc', '--reward_code', type=str,
                        default="head_stable_manual_reposition",
                        help='specify reward function \n' + \
                             'PigeonEnv3Joints: \n' + \
                             '  head_stable_manual_reposition \n' + \
                             '  head_stable_manual_reposition_strict_angle \n' + \
                             'PigeonRetinalEnv \n' + \
                             '  motion_parallax \n' + \
                             '  retinal_stabilization \n' + \
                             '  fifty_fifty'
                             )
    parser.add_argument('-mo', '--max_offset', type=float,
                        default=1.0,
                        help='specify max offset for aligning head to target')
    args = parser.parse_args()

    # noinspection PyTypeChecker
    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=3000,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
    )

    # Select environment
    env_code = None
    expl_env = None
    eval_env = None
    if args.environment == "PigeonEnv3Joints":
        env_code = 'sac_pigeon_3_joints_'
        expl_env = NormalizedBoxEnv(PigeonEnv3Joints(args.body_speed, args.reward_code, args.max_offset))
        eval_env = NormalizedBoxEnv(PigeonEnv3Joints(args.body_speed, args.reward_code, args.max_offset))
    elif args.environment == "PigeonRetinalEnv":
        env_code = 'sac_pigeon_retinal_env_'
        expl_env = NormalizedBoxEnv(PigeonRetinalEnv(args.body_speed, args.reward_code))
        eval_env = NormalizedBoxEnv(PigeonRetinalEnv(args.body_speed, args.reward_code))
    else:
        raise ValueError("Unknown pigeon gym environment")

    # setting up argparse params (body speed and reward function)
    setup_logger(env_code + args.reward_code + \
                 '_body_speed_' + str(args.body_speed) + \
                 '_max_offset_' + str(args.max_offset),
                 variant=variant)
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(expl_env, eval_env, variant, args)
