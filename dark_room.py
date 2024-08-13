import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DarkRoom(gym.Env):
    metadata = {"render_modes": ["rgb_array", "2d_array"], "render_fps": 1}

    def __init__(self, size=9, goal=(0, 0), use_wall=False, render_mode="rgb_array"):
        self.size = size
        self.agent_pos = None

        if goal is not None:
            self.goal_pos = np.asarray(goal)
            assert self.goal_pos.ndim == 1
        else:
            raise ValueError("goal must be specified")

        self.observation_space = spaces.Box(
            low=0, high=self.size - 1, shape=(2,), dtype=int
        )
        self.action_space = spaces.Discrete(5)

        self.center_pos = (self.size // 2, self.size // 2)
        self.render_mode = render_mode
        self.use_wall = use_wall
        self.wall = self.create_curvy_wall()
        self.step_count = 0

        if self.agent_pos is None:
            self.reset()

    def create_curvy_wall(self):
        wall = np.zeros((self.size, self.size), dtype=bool)
        mid = self.size // 2

        if self.use_wall:
            # Create a sinusoidal curve wall
            for x in range(3, self.size - 3):
                y = int(mid + np.sin((x - mid) / 2) * 2)
                if 0 <= x < self.size:
                    wall[y, x] = True

        return wall

    def reset(self, use_random_agent_pos=False, agent_pos=None):
        self.step_count = 0

        if use_random_agent_pos:
            while True:
                start_x = np.random.choice([0, 2])
                x = np.random.randint(start_x, self.size)
                y = np.random.randint(0 if start_x == 2 else 2, self.size)
                if not self.wall[y, x]:
                    self.agent_pos = [y, x]
                    break
        elif agent_pos is not None:
            self.agent_pos = agent_pos
        else:
            self.agent_pos = [self.size // 2, self.size // 2]

            # Ensure the agent doesn't start on the wall
            while self.wall[self.agent_pos[0], self.agent_pos[1]]:
                self.agent_pos[1] = (self.agent_pos[1] + 1) % self.size

        return self.agent_pos

    def step(self, action):
        new_pos = self.agent_pos.copy()
        self.step_count += 1

        if action == 0:  # left
            new_pos[1] = max(0, new_pos[1] - 1)
        elif action == 1:  # right
            new_pos[1] = min(self.size - 1, new_pos[1] + 1)
        elif action == 2:  # up
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == 3:  # down
            new_pos[0] = min(self.size - 1, new_pos[0] + 1)

        # Check if the new position is valid (not a wall)
        if not self.wall[new_pos[0], new_pos[1]]:
            self.agent_pos = new_pos

        if self.agent_pos == new_pos:
            # this is my try to make agent not to stuck next to wall
            reward = -2

        done = (self.agent_pos == self.goal_pos).all()
        reward = -1
        if done: 
            reward = 1
        elif self.step_count > self.size * 2:
            # another one try to force transformer to find optimal path
            reward = -2

        return self.agent_pos, reward, done

    def render(self):
        if self.render_mode == "rgb_array":
            # Create a grid representing the dark room
            grid = np.full(
                (self.size, self.size, 3), fill_value=(255, 255, 255), dtype=np.uint8
            )

            # Draw the wall
            grid[self.wall] = (128, 128, 128)  # Gray color for the wall

            # Draw the goal and agent
            grid[self.goal_pos[0], self.goal_pos[1]] = (255, 0, 0)  # Red for the goal
            grid[int(self.agent_pos[0]), int(self.agent_pos[1])] = (
                0,
                255,
                0,
            )  # Green for the agent

            return grid
