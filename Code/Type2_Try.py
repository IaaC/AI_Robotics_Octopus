import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
import math


class ShapeCarvingEnv(gym.Env):
    def __init__(self, width=200, height=200):
        super().__init__()
        self.width, self.height = width, height
        self.grid_size = 10  # Size of each carving box
        self.grid_width = width // self.grid_size
        self.grid_height = height // self.grid_size

        # Initialize grid for the shape
        self.grid = np.ones(
            (self.grid_height, self.grid_width), dtype=np.int32)

        # Carving tool position
        self.tool_x = self.grid_width // 2
        self.tool_y = self.grid_height // 2

        # Target circle parameters
        self.center_x = self.grid_width // 2
        self.center_y = self.grid_height // 2
        self.radius = min(self.grid_width, self.grid_height) // 3

        # Game state
        self.steps_taken = 0
        self.max_steps = 200
        self.done = False

        # Define action and observation spaces
        # Keep same as original: left, stay, right
        self.action_space = spaces.Discrete(3)

        # Modified observation space to match original game's format (2 values)
        self.observation_space = spaces.Box(
            low=0, high=max(width, height),
            shape=(2,),
            dtype=np.int32
        )

        # Initialize pygame properly
        pygame.init()  # Initialize all pygame modules
        pygame.font.init()  # Specifically initialize the font module
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Shape Carving Game")
        self.clock = pygame.time.Clock()

        # Create initial deformed shape (random blob)
        self._create_deformed_shape()

        # Current similarity
        self.current_similarity = self._calculate_circle_similarity()

    def _create_deformed_shape(self):
        # Reset grid
        self.grid = np.ones(
            (self.grid_height, self.grid_width), dtype=np.int32)

        # Create a deformed blob by setting blocks to 1
        # Start with a filled grid
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                # Add some noise to make the shape deformed
                distance = math.sqrt((x - self.center_x) **
                                     2 + (y - self.center_y)**2)
                noise = np.random.uniform(0.7, 1.3)
                if distance * noise > self.radius:
                    # Outside the noisy radius - keep filled
                    self.grid[y, x] = 1
                else:
                    # Inside the noisy radius - carve out
                    self.grid[y, x] = 0

        # Add some random blocks for more deformation
        for _ in range(50):
            x = np.random.randint(0, self.grid_width)
            y = np.random.randint(0, self.grid_height)
            self.grid[y, x] = np.random.choice([0, 1])

    def _calculate_circle_similarity(self):
        """Calculate how similar the carved shape is to a perfect circle"""
        similarity = 0
        total_cells = 0

        for y in range(self.grid_height):
            for x in range(self.grid_width):
                distance = math.sqrt((x - self.center_x) **
                                     2 + (y - self.center_y)**2)

                # If inside the circle
                if distance <= self.radius:
                    # Should be carved (0)
                    if self.grid[y, x] == 0:
                        similarity += 1
                else:
                    # Should be solid (1)
                    if self.grid[y, x] == 1:
                        similarity += 1

                total_cells += 1

        return similarity / total_cells

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.tool_x = self.grid_width // 2
        self.tool_y = self.grid_height // 2
        self.steps_taken = 0
        self.done = False

        # Create a new deformed shape
        self._create_deformed_shape()
        self.current_similarity = self._calculate_circle_similarity()

        # Return state as [tool_x, current_similarity] to match expected format
        return np.array([self.tool_x, int(self.current_similarity * 100)]), {}

    def step(self, action):
        reward = 0
        self.steps_taken += 1

        # Store old tool position and similarity
        old_tool_x = self.tool_x
        old_similarity = self.current_similarity

        # Process action (adapted to match original game's 3 actions)
        if action == 0:  # Move left
            self.tool_x = max(0, self.tool_x - 1)
        elif action == 2:  # Move right
            self.tool_x = min(self.grid_width - 1, self.tool_x + 1)
        # Action 1 is "stay" in the original game

        # Always carve at current position (simplified mechanics)
        if self.grid[self.tool_y, self.tool_x] == 1:
            self.grid[self.tool_y, self.tool_x] = 0  # Carve out this block

            # Calculate if this carve improved the circle similarity
            new_similarity = self._calculate_circle_similarity()
            similarity_change = new_similarity - old_similarity
            self.current_similarity = new_similarity

            # Reward based on improvement in circle similarity
            reward = similarity_change * 100  # Scale up for more meaningful rewards

        # Check if done
        if self.steps_taken >= self.max_steps:
            self.done = True

            # Final reward based on circle similarity
            reward += self.current_similarity * 50  # Scale up the final similarity score

        # Return state as [tool_x, current_similarity] to match expected format
        return np.array([self.tool_x, int(self.current_similarity * 100)]), reward, self.done, False, {}

    def render(self):
        # Fill background
        self.screen.fill((200, 200, 200))

        # Draw grid
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                color = (0, 0, 0) if self.grid[y, x] == 1 else (255, 255, 255)
                pygame.draw.rect(
                    self.screen,
                    color,
                    (x * self.grid_size, y * self.grid_size,
                     self.grid_size, self.grid_size)
                )
                # Draw grid lines
                pygame.draw.rect(
                    self.screen,
                    (100, 100, 100),
                    (x * self.grid_size, y * self.grid_size,
                     self.grid_size, self.grid_size),
                    1
                )

        # Draw the target circle (faintly)
        pygame.draw.circle(
            self.screen,
            (0, 255, 0),  # Green
            (self.center_x * self.grid_size + self.grid_size//2,
             self.center_y * self.grid_size + self.grid_size//2),
            self.radius * self.grid_size,
            1  # Just the outline
        )

        # Draw the carving tool
        pygame.draw.rect(
            self.screen,
            (255, 0, 0),
            (self.tool_x * self.grid_size, self.tool_y * self.grid_size,
             self.grid_size, self.grid_size),
            3  # Thicker outline
        )

        # Display similarity score - handle font more carefully
        try:
            font = pygame.font.SysFont(None, 24)
            similarity = self.current_similarity * 100
            text = font.render(
                f"Circle Similarity: {similarity:.1f}%", True, (0, 0, 255))
            self.screen.blit(text, (10, 10))
        except pygame.error:
            # If font rendering fails, just skip it
            pass

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        pygame.quit()


env = ShapeCarvingEnv()

q_table = np.zeros((400, 400, 3))  # Q-values for (paddle_x, block_x, action)
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate

for episode in range(1000):
    state, _ = env.reset()
    done = False

    while not done:
        paddle_x, block_x = state
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[paddle_x, block_x])  # Exploit

        next_state, reward, done, _, _ = env.step(action)
        next_paddle_x, next_block_x = next_state

        # Q-learning update
        q_table[paddle_x, block_x, action] = (1 - alpha) * q_table[paddle_x, block_x, action] + \
            alpha * (reward + gamma *
                     np.max(q_table[next_paddle_x, next_block_x]))

print("Training complete!")

state, _ = env.reset()
done = False

while not done:
    paddle_x, block_x = state
    action = np.argmax(q_table[paddle_x, block_x])  # Use trained policy
    state, reward, done, _, _ = env.step(action)
    env.render()

env.close()
