import gymnasium as gym
import numpy as np
from gymnasium import spaces

 
class DarkRoom(gym.Env):
    metadata = {"render_modes": ["rgb_array", "2d_array"], "render_fps": 1}

    def __init__(self, size=9, goal=(0,0), terminate_on_goal=False, render_mode="rgb_array"):
        self.size = size
        self.agent_pos = None

        if goal is not None:
            self.goal_pos = np.asarray(goal)
            assert self.goal_pos.ndim == 1
        else:
            raise ValueError("goal must be specified")

        self.observation_space = spaces.Box(low=0, high=self.size - 1, shape=(2,), dtype=int)
        self.action_space = spaces.Discrete(5)

         
        self.center_pos = (self.size // 2, self.size // 2) 
        self.render_mode = render_mode
        if self.agent_pos is None:
            self.reset()
 
    def reset(self):  
        self.agent_pos = [self.size // 2, self.size // 2]
        return self.agent_pos


    def step(self, action):
        if action == 0:  # left
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 1:  # right
            self.agent_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)
        elif action == 2:  # up
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 3:  # down
            self.agent_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)
        
        done = (self.agent_pos == self.goal_pos).all()
        reward = 1 if done else -1
        if(done):
            print("Goal reached")
        return self.agent_pos, reward, done 
    
    def render(self):
        if self.render_mode == "rgb_array":
            # Create a grid representing the dark room
            grid = np.full((self.size, self.size, 3), fill_value=(255, 255, 255), dtype=np.uint8)
            grid[self.goal_pos[0], self.goal_pos[1]] = (255, 0, 0)
            grid[int(self.agent_pos[0]), int(self.agent_pos[1])] = (0, 255, 0) 
            return grid
        if self.render_mode == "2d_array":
            # Create a grid representing the dark room
            grid = np.full((self.size, self.size), fill_value=255, dtype=np.uint8)
            grid[self.goal_pos[0], self.goal_pos[1]] = 1
            grid[int(self.agent_pos[0]), int(self.agent_pos[1])] = 2
            return grid