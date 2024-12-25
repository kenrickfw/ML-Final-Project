import torch
import random
import numpy as np
from collections import deque
from snakegame import SnakeGameAI, Direction, Point  
from trainmodel import Linear_QNet, QTrainer, Dueling_Noisy_QNet  
from plotgraph import plot  
import math

# Constants for memory size, batch size, and learning rate
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001  # Learning rate
# LR = 0.001 -> n_games = 25 avg score = 0.03 --> fail
# LR = 0.01 -> n_games = 25 avg score = 0.03 --> fail

class Agent:
    def __init__(self):
        # Initialize the agent with game count, exploration rate, and replay memory
        self.n_games = 0
        self.epsilon = 0  # Randomness for exploration
        self.gamma = 0.9  # Discount rate for future rewards
        self.memory = deque(maxlen=MAX_MEMORY)  # Store transitions with limited size
        self.model = Linear_QNet(11, 256, 3)  # Neural network model
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)  # Q-learning trainer

    def get_state(self, game):
        # Create a representation of the game state for decision-making
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Define state as an array of boolean and positional values
        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # Food left
            game.food.x > game.head.x,  # Food right
            game.food.y < game.head.y,  # Food up
            game.food.y > game.head.y  # Food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        # Store the experience tuple in memory
        self.memory.append((state, action, reward, next_state, done))  # Remove oldest if full

    def train_long_memory(self):
        # Train the model using a batch of experiences from memory
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # Sample random batch
        else:
            mini_sample = self.memory

        # Extract states, actions, rewards, etc., for batch training
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        # Train the model immediately on a single step
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Random moves: tradeoff exploration vs. exploitation
        self.epsilon = 80 - self.n_games  # Decay exploration rate
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:  # Explore
            move = random.randint(0, 2)
            final_move[move] = 1
        else:  # Exploit learned knowledge
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)  # Predict action
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        # Initialize the replay buffer with priority parameters
        self.capacity = capacity
        self.buffer = []
        self.priorities = deque(maxlen=capacity)
        self.pos = 0
        self.alpha = alpha

    def add(self, state, action, reward, next_state, done):
        # Add experience to the buffer with maximum priority
        max_priority = max(self.priorities) if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities.append(max_priority)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        # Sample a batch with probabilities proportional to priorities
        priorities = np.array(self.priorities, dtype=np.float32)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        # Update the priorities of sampled experiences
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

class RainbowAgent:
    def __init__(self):
        # Initialize the Rainbow agent with its components
        self.n_games = 0
        self.epsilon = 1.0  # Start with high exploration
        self.memory = PrioritizedReplayBuffer(MAX_MEMORY)  # Prioritized replay buffer
        self.gamma = 0.99  # Discount rate
        self.n_step = 3  # Number of steps for multi-step learning
        self.n_step_buffer = deque(maxlen=self.n_step)
        self.model = Dueling_Noisy_QNet(11, 256, 3)  # Main network
        self.target_model = Dueling_Noisy_QNet(11, 256, 3)  # Target network
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def update_target_network(self):
        # Update the target network with the main network's weights
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state):
        # Choose an action based on exploration vs. exploitation
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        if random.random() < self.epsilon:  # Explore
            return random.randint(0, 2)
        else:  # Exploit
            with torch.no_grad():
                return torch.argmax(self.model(state_tensor)).item()

    def train_long_memory(self):
        # Train the model using a batch from the prioritized replay buffer
        if len(self.memory.buffer) > BATCH_SIZE:
            mini_batch, indices, weights = self.memory.sample(BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*mini_batch)

            # Convert to tensors
            states = torch.tensor(states, dtype=torch.float)
            next_states = torch.tensor(next_states, dtype=torch.float)
            actions = torch.tensor(actions, dtype=torch.long)
            rewards = torch.tensor(rewards, dtype=torch.float)
            dones = torch.tensor(dones, dtype=torch.bool)
            weights = torch.tensor(weights, dtype=torch.float)

            # Compute Q targets using the target network
            with torch.no_grad():
                next_q_values = self.target_model(next_states)
                next_actions = torch.argmax(next_q_values, dim=1)
                next_q_values = self.target_model(next_states)[torch.arange(BATCH_SIZE), next_actions]
                targets = rewards + (self.gamma ** self.n_step) * next_q_values * ~dones

            # Compute predicted Q values
            predicted_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            # Calculate loss
            td_errors = targets - predicted_q_values
            loss = (weights * td_errors.pow(2)).mean()

            # Optimize the model
            self.trainer.optimizer.zero_grad()
            loss.backward()
            self.trainer.optimizer.step()

            # Update priorities in the replay buffer
            priorities = td_errors.abs().detach().cpu().numpy() + 1e-5
            self.memory.update_priorities(indices, priorities)

def train():
    # Main training loop
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()  # Initialize the agent
    game = SnakeGameAI()  # Initialize the game environment
    while True:
        # Get old state
        state_old = agent.get_state(game)

        # Get move
        final_move = agent.get_action(state_old)

        # Perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # Remember the transition
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:  # Save model if record is broken
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    # Start training the agent
    train()
