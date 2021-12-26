from gym_env.pigeon_gym import PigeonEnv3Joints, VIEWPORT_SCALE
import numpy as np
import gym
from gym import spaces

class PigeonRetinalEnv(PigeonEnv3Joints):

    def __init__(self,
                 body_speed = 0,
                 reward_code = "motion_parallax"):

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
        Object Location Init (2D Tensor)
        """
        self.objects_position = np.array([[-30.0, 30.0]])


    """
    Retinal coords (angles); Within [-np.pi, np.pi]
    """
    def _get_retinal(self, object_position):
        # normalized direction of object from head
        object_direction = object_position - np.array(self.head.position)
        object_direction = object_direction / np.linalg.norm(object_direction)

        # is the object above or below the head?
        sign = 1
        if object_direction[1] < 0:
            sign = -1

        # calculate COSINE angle of object relative to head (positive if above, negative if below)
        # cosine_angle is of size [num_objects,]
        cosine_angle = sign * np.arccos( \
            np.dot(object_direction, np.array([-1.0, 0.0])))

        # differnce in angle between the head angle and sine_angle of head
        relative_angle = cosine_angle + self.head.angle

        # relative_angle should be within [-np.pi, np.pi]
        for i in range(relative_angle.shape[0]):
            if relative_angle[i] < -np.pi:
                k = 1
                while relative_angle[i] < (k + 1) * -np.pi:
                    k += 1
                relative_angle[i] = relative_angle[i] + 2 * np.pi * ((k + 1) // 2)

            elif relative_angle[i] > np.pi:
                k = 1
                while relative_angle[i] > (k + 1) * np.pi:
                    k += 1
                relative_angle[i] = relative_angle[i] - 2 * np.pi * ((k + 1) // 2)

        return relative_angle

    def _get_angular_velocity(self, prev_ang, current_ang):
        angle_velocity = current_ang - prev_ang
        angle_speed = np.absolute(angle_velocity)
        for i in range(angle_velocity.size):
            if angle_speed[i] > np.pi:
                angle_velocity[i] = 2 * np.pi - angle_velocity[i]
            elif angle_speed[i] < -np.pi
                angle_velocity[i] = 2 * np.pi + angle_velocity[i]
            else:
                pass
        return angle_velocity


    def _assign_reward_func(self, reward_code, max_offset = None):
        if "motion_parallax" in reward_code:
            self.prev_angle = self._get_retinal(self.objects_position)
            self.reward_function = self._motion_parallax
        else:
            raise ValueError("Unknown reward_code")

    def _motion_parallax(self):
        current_angle = self._get_retinal(self.objects_position)

        parallax_velocities = \
            self._get_angular_velocity(current_angle, self.prev_angle)

        reward = 0
        # sum of motion parallax magnitudes
        for i in range(parallax_velocities.size):
            for j in range(i, parallax_velocities.size):
                reward += np.abs(parallax_velocities[i], parallax_velocities[j])
        return reward

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
        self._get_retinal(self.object)
        # alter object
        return super().step(action)

    def render(self, mode = "human"):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.render_object = None

        super().render(mode)
        if self.render_object is None:
            self.render_object = rendering.make_circle( \
                radius=VIEWPORT_SCALE * 0.1,
                res=30,
                filled=True)
            self.object_translate = rendering.Transform(
                translation = VIEWPORT_SCALE * self.object - self.camera_trans,
                rotation = 0.0,
                scale = VIEWPORT_SCALE * np.ones(2)
            )
            self.render_object.add_attr(self.object_translate)
            self.render_object.set_color(0.0, 1.0, 0.0)
            self.viewer.add_geom(self.render_object)

        new_object_translate = VIEWPORT_SCALE * self.object - self.camera_trans
        self.object_translate.set_translation(new_object_translate[0], new_object_translate[1])

        return self.viewer.render(return_rgb_array = mode == "rgb_array")
