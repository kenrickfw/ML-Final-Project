import pygame
import random
from enum import Enum  
from collections import namedtuple  
import numpy as np  

# Initialize Pygame and set font for score display
pygame.init()
font = pygame.font.Font('arial.ttf', 25)

# Define directions as an enumerated type for readability
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Define a Point class to store coordinates of the snake and food
Point = namedtuple('Point', 'x, y')

# Define colors for the game elements
WHITE = (255, 255, 255)  # Font color
RED = (200, 0, 0)  # Food color
BLUE1 = (0, 0, 255)  # Outer blue for the snake
BLUE2 = (0, 100, 255)  # Inner blue for the snake
BLACK = (0, 0, 0)  # Background color

# Constants for block size and game speed
BLOCK_SIZE = 20  # Size of each grid block
SPEED = 100  # Speed of the game

# Snake game class with AI capabilities
class SnakeGameAI:
    def __init__(self, w=640, h=480):
        # Initialize game window, dimensions, and reset game state
        self.w = w  # Width of the game window
        self.h = h  # Height of the game window
        # Initialize Pygame display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')  # Set game title
        self.clock = pygame.time.Clock()  # Clock to control game speed
        self.reset()  # Reset the game state

    def reset(self):
        # Reset the snake, direction, score, and place the first food
        self.direction = Direction.RIGHT  # Start with the snake moving right
        self.head = Point(self.w / 2, self.h / 2)  # Snake starts at the center
        self.snake = [self.head,  # Snake's initial body
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0  # Initialize score
        self.food = None
        self._place_food()  # Place the first food
        self.frame_iteration = 0  # Track game frames for timeout

    def _place_food(self):
        # Randomly place food on the grid, avoiding the snake's body
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:  # Ensure food is not placed on the snake
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1  # Increment frame counter

        # Handle Pygame events (e.g., quitting)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Move the snake
        self._move(action)  # Update the snake's head position
        self.snake.insert(0, self.head)  # Add the new head to the snake

        # Initialize reward and check game over conditions
        reward = 0
        game_over = False
        food_distance_before = self._calculate_distance(self.head, self.food)  # Distance before move
        if self.is_collision() or self.frame_iteration > 200 * len(self.snake):  # Check collision or timeout
            game_over = True
            reward = -10  # Penalty for losing
            return reward, game_over, self.score

        # Check if snake eats food
        if self.head == self.food:
            self.score += 1  # Increase score
            reward = 10  # Reward for eating food
            self._place_food()  # Place new food
        else:
            self.snake.pop()  # Remove tail segment to maintain size
            food_distance_after = self._calculate_distance(self.head, self.food)  # Distance after move

            # Reward or penalize based on distance to food
            if len(self.snake) <= 30:  # Encourage growth for smaller snakes
                if food_distance_after < food_distance_before:
                    reward = 0.1  # Small reward for getting closer
                else:
                    reward = -0.1  # Small penalty for getting farther

        # Update game visuals
        self._update_ui()
        self.clock.tick(SPEED)  # Control game speed
        return reward, game_over, self.score  # Return step results

    def _calculate_distance(self, point1, point2):
        # Calculate Manhattan distance between two points
        return abs(point1.x - point2.x) + abs(point1.y - point2.y)

    def is_collision(self, pt=None):
        # Check if the snake collides with the wall or itself
        if pt is None:
            pt = self.head  # Default to checking the head
        # Check if the head hits the boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # Check if the head hits itself
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        # Update the game display with snake, food, and score
        self.display.fill(BLACK)  # Clear screen with black background

        for pt in self.snake:
            # Draw snake blocks
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        # Draw food block
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        # Render and display the score
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # Determine the new direction of the snake based on the action
        # action = [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]  # Order of directions
        idx = clock_wise.index(self.direction)  # Find current direction index

        # Determine new direction
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # Continue straight
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4  # Turn right
            new_dir = clock_wise[next_idx]
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4  # Turn left
            new_dir = clock_wise[next_idx]

        self.direction = new_dir  # Update direction

        # Move the snake's head in the new direction
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)  # Update the head position
