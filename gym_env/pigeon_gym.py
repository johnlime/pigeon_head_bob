from Box2D import *
import gym
from gym import spaces

from math import sin, pi, sqrt
import numpy as np
from copy import deepcopy

# anatomical variables ("macros")
BODY_WIDTH = 10
BODY_HEIGHT = 5

LIMB_WIDTH = 5
LIMB_HEIGHT = 2

HEAD_WIDTH = 3

# control variables/macros
MAX_JOINT_TORQUE = 10.0 ** 6
MAX_JOINT_SPEED = 10
TORQUE_WEIGHT = 0.1

VIEWPORT_SCALE = 6.0

class PigeonEnv3Joints(gym.Env):
    def __init__(self, body_speed = 0, reward_code = "head_stable_manual_reposition"):
        """
        Action and Observation space
        """

        # 3-dim joints' torque ratios
        self.action_space = spaces.Box(
            np.array([-1.0] * 3).astype(np.float32),
            np.array([1.0] * 3).astype(np.float32),
        )
        # 2-dim head location;
        # 1-dim head angle;
        # 3x2-dim joint angle and angular velocity;
        # 1-dim x-axis of the body
        high = np.array([np.inf] * 10).astype(np.float32)
        self.observation_space = spaces.Box(-high, high)

        """
        Box2D Pigeon Model Params and Initialization
        """
        self.world = b2World()                          # remove in Framework
        self.body = None
        self.joints = []
        self.head = None
        self.bodyRef = [] # for destruction
        self.body_speed = body_speed
        self.pigeon_model()

        """
        Box2D Simulation Params
        """
        self.timeStep = 1.0 / 60
        self.vel_iters, self.pos_iters = 10, 10

        self.viewer = None

        """
        Assigning a Reward Function
        """
        if reward_code == "head_stable_01":
            # justified since this can be regarded as a constructor
            self.head_prev_pos = np.array([0.0, 0.0])       # head tracking
            self.head_prev_ang = 0                          # head tracking

            # can call method unless it's virtual (abstract)
            self.reward_function = self._head_stable_01

        elif reward_code == "head_stable_manual_reposition_01" or \
                reward_code == "head_stable_manual_reposition":
            self.relative_head_target_location = np.array(self.head.position)
            self.head_target_location = np.array(self.head.position)
            self.head_target_angle = self.head.angle
            self.reward_function = self._head_stable_manual_reposition

        else:
            raise ValueError("Unknown reward_code")

    """
    Box2D Pigeon Model
    """
    def pigeon_model(self):
        # params
        body_anchor = np.array([float(-BODY_WIDTH), float(BODY_HEIGHT)])
        limb_width_cos = LIMB_WIDTH / sqrt(2)

        self.bodyRef = []
        # body definition
        self.body = self.world.CreateKinematicBody(
            position = (0, 0),
            shapes = b2PolygonShape(box = (BODY_WIDTH, BODY_HEIGHT)), # x2 in direct shapes def
            linearVelocity = (-self.body_speed, 0),
            angularVelocity = 0,
            )
        self.bodyRef.append(self.body)

        # neck as limbs + joints definition
        self.joints = []
        current_center = deepcopy(body_anchor)
        current_anchor = deepcopy(body_anchor)
        offset = np.array([-limb_width_cos, limb_width_cos])
        prev_limb_ref = self.body
        for i in range(2):
            if i == 0:
                current_center += offset

            else:
                current_center += offset * 2
                current_anchor += offset * 2

            tmp_limb = self.world.CreateDynamicBody(
                position = (current_center[0], current_center[1]),
                fixtures = b2FixtureDef(density = 2.0,
                                        friction = 0.6,
                                        shape = b2PolygonShape(
                                            box = (LIMB_WIDTH, LIMB_HEIGHT)),
                                        ),
                angle = -pi / 4
            )
            self.bodyRef.append(tmp_limb)

            tmp_joint = self.world.CreateRevoluteJoint(
                bodyA = prev_limb_ref,
                bodyB = tmp_limb,
                anchor = current_anchor,
                lowerAngle = -0.5 * b2_pi, # -90 degrees
                upperAngle = 0.5 * b2_pi,  #  90 degrees
                enableLimit = True,
                maxMotorTorque = MAX_JOINT_TORQUE * (TORQUE_WEIGHT ** i),
                motorSpeed = 0.0,
                enableMotor = True,
            )

            self.joints.append(tmp_joint)
            prev_limb_ref = tmp_limb

        # head def + joints
        current_center += offset
        current_anchor += offset * 2
        self.head = self.world.CreateDynamicBody(
            position = (current_center[0] - HEAD_WIDTH, current_center[1]),
            fixtures = b2FixtureDef(density = 2.0,
                                    friction = 0.6,
                                    shape = b2PolygonShape(
                                        box = (HEAD_WIDTH, LIMB_HEIGHT)),
                                    ),
        )
        self.bodyRef.append(self.head)

        head_joint = self.world.CreateRevoluteJoint(
            bodyA = prev_limb_ref,
            bodyB = self.head,
            anchor = current_anchor,
            lowerAngle = -0.5 * b2_pi, # -90 degrees
            upperAngle = 0.5 * b2_pi,  #  90 degrees
            enableLimit = True,
            maxMotorTorque = MAX_JOINT_TORQUE * (TORQUE_WEIGHT ** 2),
            motorSpeed = 0.0,
            enableMotor = True,
        )
        self.joints.append(head_joint)

        # head tracking
        self.head_prev_pos = np.array(self.head.position)
        self.head_prev_ang = self.head.angle

    def destroy(self):
        for body in self.bodyRef:
            # all associated joints are destroyed implicitly
            self.world.DestroyBody(body)

    def get_obs(self):
        # (self.head, self.joints, self.body -> obs) operations
        obs = np.array(self.head.position)
        obs = np.concatenate((obs, self.head.angle), axis = None)
        for i in range(len(self.joints)):
            obs = np.concatenate((obs, self.joints[i].angle), axis = None)
            obs = np.concatenate((obs, self.joints[i].speed), axis = None)
        obs = np.concatenate((obs, self.body.position[0]), axis = None)

        obs = np.float32(obs)
        assert self.observation_space.contains(obs)
        return obs

    def reset(self):
        self.destroy()
        self.pigeon_model()
        return self.get_obs()

    # modular reward functions
    def _head_stable_01(self):
        head_dif_loc = np.linalg.norm(np.array(self.head.position) - self.head_prev_pos)
        head_dif_ang = abs(self.head.angle - self.head_prev_ang)

        reward = 0
        #threshold function
        if head_dif_loc < 0.5:
            reward += 1

            if head_dif_ang < np.pi / 6: # 30 deg
                reward += 1

        else:
            reward += 0

        # head tracking
        self.head_prev_pos = np.array(self.head.position)
        self.head_prev_ang = self.head.angle
        return reward

    # ONLY WORKS ON body_speed = 0
    def _head_stable_manual_reposition_01(self):
        head_dif_loc = np.linalg.norm(np.array(self.head.position) - self.head_target_location)
        head_dif_ang = abs(self.head.angle - self.head_target_angle)

        reward = 0
        if head_dif_loc < 0.5:
            reward += 1 - head_dif_loc / 0.5

            if head_dif_ang < np.pi / 6: # 30 deg
                reward += 1 - head_dif_ang/ np.pi

        return reward

    def _head_stable_manual_reposition(self):
        # detect whether the target head position is behind the body edge or not
        if self.head_target_location[0] > self.body.position[0] + float(-BODY_WIDTH):
            self.head_target_location = np.array(self.body.position) + \
                self.relative_head_target_location

        return self._head_stable_manual_reposition_01()

    def _head_stable_movement_minimizer(self):
        reward = 0
        return reward

    def step(self, action):
        assert self.action_space.contains(action)
        # self.world.Step(self.timeStep, self.vel_iters, self.pos_iters)
        # Framework handles this differently
        # Copied from bipedal_walker
        # self.world.Step(1.0 / 50, 6 * 30, 2 * 30)
        self.world.Step(1.0 / 50, self.vel_iters, self.pos_iters)
        obs = self.get_obs()

        # MOTOR CONTROL
        for i in range(len(self.joints)):
            # Copied from bipedal_walker
            self.joints[i].motorSpeed = float(MAX_JOINT_SPEED * np.sign(action[i]))
            self.joints[i].maxMotorTorque = float(
                MAX_JOINT_TORQUE * (TORQUE_WEIGHT ** i)
                    * np.clip(np.abs(action[i]), 0, 1)
            )

        reward = self.reward_function()

        done = False
        info = {}
        return obs, reward, done, info

    def render(self, mode = "human"):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
        background = rendering.FilledPolygon(
            [
                (0, 0),
                (500, 0),
                (500, 500),
                (0, 500)
            ]
        )
        background.set_color(1.0, 1.0, 1.0)
        self.viewer.add_geom(background)

        # Set ORIGIN POINT relative to camera
        camera_trans = b2Vec2(-250, -200) \
        + VIEWPORT_SCALE * self.bodyRef[0].position # camera moves with body

        for body in self.bodyRef:
            polygon = rendering.FilledPolygon(
                body.fixtures[0].shape.vertices
            )
            rotate = rendering.Transform(
                translation = (0.0, 0.0),
                rotation = body.angle,
            )
            translate = rendering.Transform(
                translation = VIEWPORT_SCALE * body.position - camera_trans,
                rotation = 0.0,
                scale = VIEWPORT_SCALE * np.ones(2)
            )
            polygon.set_color(1.0, 0.0, 0.0)
            polygon.add_attr(rotate)
            polygon.add_attr(translate)
            self.viewer.add_geom(polygon)
        return self.viewer.render(return_rgb_array = mode == "rgb_array")

    def close(self):
        self.destroy()
        self.world = None

        if self.viewer:
            self.viewer.close()
            self.viewer = None
