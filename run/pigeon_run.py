from gym_env.pigeon_gym import PigeonEnv3Joints#, PigeonEnv3JointsHeadstart
import argparse
import torch
import numpy as np
import sys
sys.path.append('src/rlkit_ppo')

def run_rand_policy(body_speed, reward_code, max_offset):
    env = PigeonEnv3Joints(body_speed, reward_code, max_offset)
    observation = env.reset()
    for t in range(1000):
        env.render()
        action = env.action_space.sample()
        _, reward, _, _ = env.step(action)
        # print(env.head_target_location)
        # print(reward)
    env.close()

def run_trained_policy(policy, body_speed, reward_code,
                       max_offset, video_path = None):
    env = PigeonEnv3Joints(body_speed, reward_code, max_offset)

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
    parser.add_argument('-dir', '--snapshot_directory', type=str,
                        help='path to the snapshot directory')
    parser.add_argument('-bs', '--body_speed', type=float, default=1.0,
                        help='pigeon body speed')
    parser.add_argument('-rc', '--reward_code', type=str,
                        default="head_stable_manual_reposition",
                        help='specify reward function')
    parser.add_argument('-mo', '--max_offset', type=float,
                        default=1.0,
                        help='specify max offset for reward function 03+')
    parser.add_argument('-v', '--video', action = 'store_true',
                        help='export to video')
    args = parser.parse_args()

    if args.snapshot_directory is None:
        run_rand_policy(args.body_speed, args.reward_code, args.max_offset)

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

            run_trained_policy(policy, args.body_speed, args.reward_code,
                               args.max_offset, video_path = video_path)

        else:
            run_trained_policy(policy, args.body_speed, args.reward_code, args.max_offset)
