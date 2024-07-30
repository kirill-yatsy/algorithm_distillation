from time import sleep
import gymnasium as gym

from dark_room import DarkRoom
import PIL.Image
import numpy as np

env = DarkRoom(goal=(0,0))

observation, info = env.reset(seed=42)
for i in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
    
    # print_grid(env.render())
    # sleep(0.1)
env.close()