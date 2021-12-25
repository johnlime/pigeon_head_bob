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
        Object Location Init
        """
        object = np.array([-100.0, 100.0])


    """
    Retinal coords (angles); Within [-np.pi, np.pi]
    """
    def _get_retinal(object):
        # normalized direction of object from head
        object_direction = object - np.array(self.head.position)
        object_direction = object_direction / np.linalg.norm(object_direction)

        # is the object above or below the head?
        sign = 1
        if object_direction[1] < 0:
            sign = -1

        # calculate sine angle of object relative to head (positive if above, negative if below)
        sine_angle = sign * np.arcsin( \
            np.sqrt(1 - np.dot(np.array([-1.0, 0.0]), object_direction) ** 2))

        # differnce in angle between the head angle and sine_angle of head
        angular_difference = sine_angle + self.head.angle

        # angular_difference should be within [-np.pi, np.pi]
        if angular_difference < -np.pi:
            k = 1
            while angular_difference < (k + 1) * -np.pi:
                k += 1
            angular_difference = -angular_difference + (-1) ** (k - 1) * 2 * np.pi

        elif angular_difference > np.pi:
            k = 1
            while angular_difference > (k + 1) * np.pi:
                k += 1
            angular_difference = angular_difference + (-1) ** k * 2 * np.pi

        return angular_difference


    def _assign_reward_func(self, reward_code, max_offset = None):
        if "placeholder" in reward_code:
            self.reward_function = self._placeholder

        elif reward_code == "motion_parallax":
            self.reward_function = self._motion_parallax

        else:
            raise ValueError("Unknown reward_code")

    def _placeholder(self):
        return 0

    def _motion_parallax(self):
        return np.linalg.norm(np.array(self.head.position))

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

    def step(self, action):
        # alter object
        return super().step(action)
