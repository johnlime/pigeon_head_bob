from gym_env.pigeon_gym_retinal import PigeonRetinalEnv
env = PigeonRetinalEnv()

observation = env.reset()
for t in range(1000):
    env.render()
    action = env.action_space.sample()
    _, reward, _, _ = env.step(action)
    print(reward)
env.close()
