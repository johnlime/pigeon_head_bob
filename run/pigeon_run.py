from gym_env.pigeon_gym import PigeonEnv3Joints
from gym_env.pigeon_gym_retinal import PigeonRetinalEnv
import argparse
import torch
import numpy as np
import sys
sys.path.append('src/rlkit_ppo')

def run_rand_policy(env):
    observation = env.reset()
    for t in range(1000):
        env.render()
        action = env.action_space.sample()
        _, reward, _, _ = env.step(action)
        # print(env.head.angle)
        print(reward)
    env.close()

def run_trained_policy(policy, env, video_path = None):
    if video_path is not None:
        from gym import wrappers
        env = wrappers.RecordVideo(env, video_path)
        # Possible alternate method (not tested)
        # env = wrappers.Monitor(env, video_path,
        #                        video_callable = lambda episode_id: True,
        #                        force = True)

    observation = env.reset()
    for t in range(1000):
        env.render()
        action = policy.get_action(torch.from_numpy(observation))[0]
        env.step(action)
        observation, reward, done, info = env.step(action)
        # if reward != 0:
        #     print(reward)
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', '--environment', type=str,
                        default = "PigeonEnv3Joints",
                        help = 'name of environment: \n' + \
                               '  PigeonEnv3Joints \n' + \
                               '  PigeonRetinalEnv')
    parser.add_argument('-dir', '--snapshot_directory', type=str,
                        help='path to the snapshot directory')
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
                             '  retinal_stabilization'
                             )
    parser.add_argument('-mo', '--max_offset', type=float,
                        default=1.0,
                        help='specify max offset for aligning head to target')
    parser.add_argument('-v', '--video', action = 'store_true',
                        help='export to video')
    args = parser.parse_args()

    # Select environment
    if args.environment == "PigeonEnv3Joints":
        env = PigeonEnv3Joints(args.body_speed, args.reward_code, args.max_offset)
    elif args.environment == "PigeonRetinalEnv":
        env = PigeonRetinalEnv(args.body_speed, args.reward_code)
    else:
        raise ValueError("Unknown pigeon gym environment")


    # Check to see if user wants to test a specific trained policy
    if args.snapshot_directory is None:
        run_rand_policy(env)

    else:
        if args.snapshot_directory[-1] == '/':
            args.snapshot_directory = args.snapshot_directory[:-1]

        policy = torch.load(args.snapshot_directory + \
                            "/evaluation/policy/params.pt",
                            map_location=torch.device('cpu'))
        if args.video:
            video_path = args.snapshot_directory + '/return_per_epoch/'

            # check to see if destination path exists
            import os
            try:
                os.stat(video_path)
            except:
                os.mkdir(video_path)

            run_trained_policy(policy, env, video_path = video_path)

        else:
            run_trained_policy(policy, env)
