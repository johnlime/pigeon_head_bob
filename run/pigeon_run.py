from gym_env.pigeon_gym import PigeonEnv3Joints
import argparse
import torch
import numpy as np
import sys
sys.path.append('src/rlkit_ppo')

def run_rand_policy(body_speed, reward_code):
    env = PigeonEnv3Joints(body_speed, reward_code)
    observation = env.reset()
    for t in range(1000):
        env.render()
        action = env.action_space.sample()
        _, reward, _, _ = env.step(action)
        print(reward)
    env.close()

def run_trained_policy(policy, body_speed, reward_code):
    env = PigeonEnv3Joints(body_speed, reward_code)
    observation = env.reset()
    for t in range(1000):
        env.render()
        action = policy.get_action(torch.from_numpy(observation))[0]
        env.step(action)
        observation, reward, done, info = env.step(action)
        print(reward)
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy_file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--body_speed', type=float, default=0.0,
                        help='pigeon body speed')
    parser.add_argument('--reward_code', type=str,
                        default="head_stable_manual_reposition",
                        help='specify reward function')
    args = parser.parse_args()

    if args.policy_file is None:
        run_rand_policy(args.body_speed, args.reward_code)

    else:
        policy = torch.load(args.policy_file,
                            map_location=torch.device('cpu'))
        run_trained_policy(policy, args.body_speed, args.reward_code)
