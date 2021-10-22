from gym_env.pigeon_gym import PigeonEnv3Joints

env = PigeonEnv3Joints(body_speed = 10)
observation = env.reset()
for t in range(1000):
    env.render()
    # print(observation)
    action = env.action_space.sample()
    env.step(action)
    # observation, reward, done, info = env.step(action)
env.close()
