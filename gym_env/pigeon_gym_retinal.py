from gym_env.pigeon_gym import PigeonEnv3Joints
import numpy as np
import gym
from gym import spaces

class PigeonRetinalEnv(PigeonEnv3Joints):
    def __init__(self,
                 body_speed = 0,
                 reward_code = "placeholder"):

        super().__init__(body_speed, reward_code)

        """
        Redefining Observation space
        """
        # 2-dim head location;
        # 1-dim head angle;
        # 3x2-dim joint angle and angular velocity;
        # 1-dim x-axis of the body
        high = np.array([np.inf] * 10).astype(np.float32) # formally 10
        self.observation_space = spaces.Box(-high, high)

        """
        Reassigning a Reward Function
        """
        self._assign_reward_func(reward_code)

    def _assign_reward_func(self, reward_code, max_offset = None):
        if "placeholder" in reward_code:
            self.reward_function = self._placeholder

        else:
            raise ValueError("Unknown reward_code")

    def _placeholder(self):
        return 0

    def _get_obs(self):
        # (self.head{relative}, self.joints -> obs) operation
        obs = np.array(self.head.position) - np.array(self.body.position)
        obs = np.concatenate((obs, self.head.angle), axis = None)
        for i in range(len(self.joints)):
            obs = np.concatenate((obs, self.joints[i].angle), axis = None)
            obs = np.concatenate((obs, self.joints[i].speed), axis = None)
        obs = np.concatenate((obs, self.body.position[0]), axis = None)
        obs = np.float32(obs)
        assert self.observation_space.contains(obs)
        return obs
