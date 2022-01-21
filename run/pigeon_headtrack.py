from gym_env.pigeon_gym import PigeonEnv3Joints
from gym_env.pigeon_gym_retinal import PigeonRetinalEnv
import argparse
import torch
import numpy as np
import sys
sys.path.append('src/rlkit_ppo')

import os
import pandas as pd

def headtrack(policy, env, dest_path):
    head_position_trajectory = [] #np.empty((1000, 2))
    head_angle_trajectory = []
    body_trajectory = []
    observation = env.reset()
    for t in range(1000):
        action = policy.get_action(torch.from_numpy(observation))[0]
        env.step(action)
        observation, reward, done, info = env.step(action)
        head_position_trajectory.append(observation[:2])
        head_angle_trajectory.append(observation[2])
        body_trajectory.append(observation[9])
    env.close()

    df = pd.DataFrame()
    df["Head Position"] = head_position_trajectory
    df["Head Angle"] = head_angle_trajectory
    df["Body"] = body_trajectory
    df.to_csv(dest_path + "trajectory.csv")

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
    args = parser.parse_args()

    # Select environment
    if args.environment == "PigeonEnv3Joints":
        env = PigeonEnv3Joints(args.body_speed)
    elif args.environment == "PigeonRetinalEnv":
        env = PigeonRetinalEnv(args.body_speed)
    else:
        raise ValueError("Unknown pigeon gym environment")

    if args.snapshot_directory[-1] == '/':
        args.snapshot_directory = args.snapshot_directory[:-1]

    policy = torch.load(args.snapshot_directory + \
                        "/evaluation/policy/params.pt",
                        map_location=torch.device('cpu'))

    # where to store the resulting array containing all of the head tracking
    try:
        os.stat(args.snapshot_directory + '/body_trajectory')
    except:
        os.mkdir(args.snapshot_directory + '/body_trajectory')

    dest_path = args.snapshot_directory + '/body_trajectory/'

    headtrack(policy, env, dest_path)
