import os
import gymnasium as gym
import numpy as np

import pygame

class MarsEnv2(gym.Env):
    def __init__(self, grid_size=10):
        super().__init__()
        self.grid_size = grid_size
        self.info = {}
        self.agent_state = np.array([0,0])
        self.goal_state = np.array([grid_size - 1, grid_size - 1])
        self.recharge_point = np.array([grid_size // 2, grid_size // 2])
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=grid_size - 1, shape=(2,), dtype=np.int32)
        self.craters = self.generate_craters()
        self.terrain = self.generate_terrain()

        pygame.init()
        self.cell_size = 50
        self.screen = pygame.display.set_mode((grid_size * self.cell_size, grid_size * self.cell_size))
        pygame.display.set_caption("Mars Rover Environment")
        self.rover_img = self._load_image("rover.png")
        self.goal_img = self._load_image("goal.png")
        self.sand_img = self._load_image("sand.png")
        self.rock_img = self._load_image("rock.png")
        self.crater_img = self._load_image("crater.png")
        self.recharge_img = self._load_image("recharge.png")

    def _load_image(self, file_name):
        base_path = os.path.dirname(__file__)
        img_path = os.path.join(base_path, file_name)
        img = pygame.image.load(img_path)
        return pygame.transform.scale(img, (self.cell_size, self.cell_size))
       
    def generate_craters(self):
        return np.array([[4, 2], [6, 2], [4, 7], [6, 7]])

    def generate_terrain(self):
        terrain = np.zeros((self.grid_size, self.grid_size))
        sand_positions = [(1, 1), (8, 1), (1, 8), (8, 8)]
        rock_positions = [(5, 0), (0, 5), (5, 9), (9, 5)]
        for pos in sand_positions:
            terrain[pos] = 1
        for pos in rock_positions:
            terrain[pos] = 2
        return terrain

    def reset(self):
        self.agent_state = np.array([0, 0])
        return self.agent_state, self.info

    def step(self, action):
        if action == 0 and self.agent_state[1] < self.grid_size - 1:  
            self.agent_state[1] += 1
        elif action == 1 and self.agent_state[1] > 0: 
            self.agent_state[1] -= 1
        elif action == 2 and self.agent_state[0] > 0: 
            self.agent_state[0] -= 1
        elif action == 3 and self.agent_state[0] < self.grid_size - 1:  
            self.agent_state[0] += 1

        reward = -1
        done = False

        if any(np.array_equal(self.agent_state, crater) for crater in self.craters):
            print("The rover fell into a crater!")
            reward = -50
            self.agent_state = np.array([0, 0]) 


        elif np.array_equal(self.agent_state, self.goal_state):
            reward = 100
            done = True

        elif np.array_equal(self.agent_state, self.recharge_point):
            print("The rover recharged its battery!")
            reward = 1
            self.recharged = True

        elif self.terrain[self.agent_state[0], self.agent_state[1]] == 1:
            reward = -10
        elif self.terrain[self.agent_state[0], self.agent_state[1]] == 2:
            reward = -20

        self.info["Distance to goal"] = np.sqrt(
            (self.agent_state[0]-self.goal_state[0])**2 +
            (self.agent_state[1]-self.goal_state[1])**2
        )


        return self.agent_state, reward, done, self.info, {"Agent Position": self.agent_state}

    def render(self):
        if not pygame.get_init():
            return
        self.screen.fill((255, 255, 255))
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                x, y = col * self.cell_size, row * self.cell_size
                pygame.draw.rect(self.screen, (200, 200, 200), (x, y, self.cell_size, self.cell_size), 1)
                if self.terrain[row, col] == 1:
                    self.screen.blit(self.sand_img, (x, y))
                elif self.terrain[row, col] == 2:
                    self.screen.blit(self.rock_img, (x, y))

        for crater in self.craters:
            self.screen.blit(self.crater_img, (crater[1] * self.cell_size, crater[0] * self.cell_size))
    
        self.screen.blit(self.rover_img, (self.agent_state[1] * self.cell_size, self.agent_state[0] * self.cell_size))
        self.screen.blit(self.goal_img, (self.goal_state[1] * self.cell_size, self.goal_state[0] * self.cell_size))
        self.screen.blit(self.recharge_img, (self.recharge_point[1] * self.cell_size, self.recharge_point[0] * self.cell_size))

        pygame.display.flip()
        pygame.time.delay(10)

    def close(self):
        pygame.quit()