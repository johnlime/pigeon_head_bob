from gym_env.pigeon_gym import PigeonEnv3Joints
from gym_env.pigeon_gym_retinal import PigeonRetinalEnv
import argparse
import torch
import numpy as np
import sys
sys.path.append('src/rlkit_ppo')

import os
import pickle

def headtrack(policy, env, dest_path):
    head_trajectory = [] #np.empty((1000, 2))
    observation = env.reset()
    for t in range(1000):
        action = policy.get_action(torch.from_numpy(observation))[0]
        env.step(action)
        observation, reward, done, info = env.step(action)
        head_trajectory.append(observation[:2])
    env.close()
    filehandler = open(dest_path, 'wb')
    pickle.save(head_trajectory, filehandler)

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
    # where to store the resulting array containing all of the head tracking
    parser.add_argument('-dest', '--destination_directory', type=str,
                        help='path to the destination directory')

    args = parser.parse_args()

    # Select environment
    if args.environment == "PigeonEnv3Joints":
        env = PigeonEnv3Joints(args.body_speed, args.reward_code, args.max_offset)
    elif args.environment == "PigeonRetinalEnv":
        env = PigeonRetinalEnv(args.body_speed, args.reward_code)
    else:
        raise ValueError("Unknown pigeon gym environment")

    if args.snapshot_directory[-1] == '/':
        args.snapshot_directory = args.snapshot_directory[:-1]

    policy = torch.load(args.snapshot_directory + \
                        "/evaluation/policy/params.pt",
                        map_location=torch.device('cpu'))

    if args.destination_directory[-1] != '/':
        args.destination_directory += '/'

    # determine pickle name (same as snapshot directory name)
    pickle_index = 0
    for i, char in enumerate(args.snapshot_directory):
        if char == '/':
            pickle_index = i
    pickle_index += 1

    try:
        os.stat(args.destination_directory + '/body_trajectory')
    except:
        os.mkdir(args.destination_directory + '/body_trajectory')

    dest_path = args.destination_directory + '/body_trajectory/' + \
        args.snapshot_directory[pickle_index:] + ".pkl"

    headtrack(policy, env, dest_path)
