import HAL_Gymnasium  # needed for the env namespace to work
import gymnasium as gym


env = gym.make("HAL_Gymnasium/HAL_run-v0", render_mode="human")
observation, info = env.reset()

episode_over = False
while not episode_over:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    episode_over = terminated or truncated

env.close()
