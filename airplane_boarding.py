import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class Robot:
    def __init__(self, robot_id, start_pos):
        self.robot_id = robot_id
        self.pos = start_pos  # (x, y)
        self.carrying = None  # item_id if carrying, else None

class Item:
    def __init__(self, item_id, pickup_pos, dropoff_pos):
        self.item_id = item_id
        self.pickup_pos = pickup_pos
        self.dropoff_pos = dropoff_pos
        self.picked = False
        self.delivered = False

class WarehouseEnv(gym.Env):
    metadata = {'render_modes': ['human', 'terminal'], 'render_fps': 1}

    def __init__(self, grid_size=6, n_robots=2, n_items=2, render_mode=None):
        self.grid_size = grid_size
        self.n_robots = n_robots
        self.n_items = n_items
        self.render_mode = render_mode

        # Each robot: 0=up, 1=down, 2=left, 3=right, 4=pickup, 5=drop
        self.action_space = spaces.MultiDiscrete([6] * self.n_robots)

        # Observation: [robot1_x, robot1_y, robot1_carrying, ..., item1_x, item1_y, item1_picked, item1_delivered, ...]
        obs_len = self.n_robots * 3 + self.n_items * 4
        self.observation_space = spaces.Box(
            low=0, high=max(self.grid_size-1, 1), shape=(obs_len,), dtype=np.int32
        )
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robots = [Robot(i, self._random_empty_pos([])) for i in range(self.n_robots)]
        used = [r.pos for r in self.robots]
        self.items = []
        for i in range(self.n_items):
            pickup = self._random_empty_pos(used)
            used.append(pickup)
            dropoff = self._random_empty_pos(used)
            used.append(dropoff)
            self.items.append(Item(i, pickup, dropoff))
        self.steps = 0
        self.max_steps = self.grid_size * self.grid_size * 2
        return self._get_obs(), {}

    def _random_empty_pos(self, used):
        while True:
            pos = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
            if pos not in used:
                return pos

    def _get_obs(self):
        obs = []
        for r in self.robots:
            obs.extend([r.pos[0], r.pos[1], -1 if r.carrying is None else r.carrying])
        for item in self.items:
            obs.extend([
                item.pickup_pos[0], item.pickup_pos[1],
                int(item.picked), int(item.delivered)
            ])
        return np.array(obs, dtype=np.int32)

    def step(self, actions):
        assert len(actions) == self.n_robots
        reward = 0
        info = {}
        done = False
        self.steps += 1
        for i, action in enumerate(actions):
            robot = self.robots[i]
            if action == 0:  # up
                robot.pos = (robot.pos[0], max(0, robot.pos[1]-1))
            elif action == 1:  # down
                robot.pos = (robot.pos[0], min(self.grid_size-1, robot.pos[1]+1))
            elif action == 2:  # left
                robot.pos = (max(0, robot.pos[0]-1), robot.pos[1])
            elif action == 3:  # right
                robot.pos = (min(self.grid_size-1, robot.pos[0]+1), robot.pos[1])
            elif action == 4:  # pickup
                for item in self.items:
                    if (not item.picked and not item.delivered and
                        robot.pos == item.pickup_pos and robot.carrying is None):
                        item.picked = True
                        robot.carrying = item.item_id
                        reward += 10  # reward for picking up
            elif action == 5:  # drop
                if robot.carrying is not None:
                    item = self.items[robot.carrying]
                    if (robot.pos == item.dropoff_pos and item.picked and not item.delivered):
                        item.delivered = True
                        robot.carrying = None
                        reward += 50  # reward for delivery
        # Penalize collisions
        positions = [r.pos for r in self.robots]
        if len(set(positions)) < len(positions):
            reward -= 20
        reward -= 1  # Step penalty
        if all(item.delivered for item in self.items) or self.steps >= self.max_steps:
            done = True
        return self._get_obs(), reward, done, False, info

    def render(self):
        if self.render_mode is None:
            return
        if self.render_mode == 'terminal':
            self._render_terminal()

    def _render_terminal(self):
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for item in self.items:
            if not item.picked:
                x, y = item.pickup_pos
                grid[y][x] = 'I'  # Item to pick up
            elif not item.delivered:
                x, y = item.dropoff_pos
                grid[y][x] = 'D'  # Delivery location
        for r in self.robots:
            x, y = r.pos
            grid[y][x] = 'R' if r.carrying is None else 'C'
        print("Warehouse:")
        for row in grid:
            print(' '.join(row))
        print()


