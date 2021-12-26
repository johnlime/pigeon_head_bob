from gym_env.pigeon_gym import PigeonEnv3Joints, VIEWPORT_SCALE
import numpy as np
import gym
from gym import spaces

class PigeonRetinalEnv(PigeonEnv3Joints):

    def __init__(self,
                 body_speed = 0,
                 reward_code = "motion_parallax"):

        """
        Object Location Init (2D Tensor)
        """
        self.objects_position = np.array([[-30.0, 30.0],
                                          [-30.0, 60.0],])
                                          [-60.0, 30.0],])
        self.objects_velocity = np.array([[0.0, 0.0],
                                          [1.0, 0.0],
                                          [-1.0, 0.0],])

        """
        Init based on superclass
        Reward function is defined here
        """
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
    Retinal coords (angles); Within [-np.pi, np.pi]
    """
    def _get_retinal(self, object_position):
        # normalized direction of object from head
        object_direction = object_position - np.array(self.head.position)
        object_direction = object_direction / np.linalg.norm(object_direction)

        sign = np.ones(object_direction.shape[0])
        for i in range(sign.size):
            # is the object above or below the head?
            if object_direction[i][1] < 0:
                sign[i] = -1

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
            elif angle_speed[i] < -np.pi:
                angle_velocity[i] = 2 * np.pi + angle_velocity[i]
            else:
                pass
        return angle_velocity


    def _assign_reward_func(self, reward_code, max_offset = None):
        self.prev_angle = self._get_retinal(self.objects_position)
        if "motion_parallax" in reward_code:
            self.reward_function = self._motion_parallax
        elif "retinal_stabilization" in reward_code:
            self.reward_function = self._retinal_stabilization
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
                reward += np.abs(parallax_velocities[i] - parallax_velocities[j])
            # reward += parallax_velocities[i]
        return reward

    def _retinal_stabilization(self):
        reward = 0
        current_angle = self._get_retinal(self.objects_position)
        relative_speeds = \
            np.absolute(self._get_angular_velocity(current_angle, self.prev_angle))
        reward -= np.sum(relative_speeds)
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
        self.prev_angle = self._get_retinal(self.objects_position)
        # alter object
        self.objects_position += self.objects_velocity
        return super().step(action)

    def render(self, mode = "human"):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.render_objects_list = None
            self.render_objects_translate_list = None

        super().render(mode)
        # initialize object rendering pointers
        if self.render_objects_list is None:
            self.render_objects_list = []
            self.render_objects_translate_list = []
            for i in range(self.objects_position.shape[0]):
                object_render_instance = rendering.make_circle( \
                    radius=0.6,
                    res=30,
                    filled=True)
                object_render_instance_translate = rendering.Transform(
                    translation = VIEWPORT_SCALE * \
                        (self.objects_position[i] - self.camera_trans),
                    rotation = 0.0,
                    scale = VIEWPORT_SCALE * np.ones(2)
                )
                object_render_instance.add_attr(object_render_instance_translate)
                object_render_instance.set_color(0.0, 1.0, 0.0)
                self.render_objects_list.append(object_render_instance)
                self.render_objects_translate_list.append(object_render_instance_translate)
                self.viewer.add_geom(object_render_instance)

        # update object translation
        new_object_translate = VIEWPORT_SCALE * self.objects_position - self.camera_trans
        for i in range(self.objects_position.shape[0]):
            self.render_objects_translate_list[i].set_translation( \
                new_object_translate[i][0], new_object_translate[i][1])

        return self.viewer.render(return_rgb_array = mode == "rgb_array")
