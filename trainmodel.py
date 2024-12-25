# Import necessary PyTorch libraries
import torch
import torch.nn as nn  # For building neural network layers
import torch.optim as optim  # For optimization algorithms
import torch.nn.functional as F  # For activation functions and loss functions
import os  # For file operations

# Define a simple feedforward neural network for Q-learning
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Define two fully connected layers
        self.linear1 = nn.Linear(input_size, hidden_size)  # Input to hidden layer
        self.linear2 = nn.Linear(hidden_size, output_size)  # Hidden to output layer

    def forward(self, x):
        # Forward pass: compute the output of the network
        x = F.relu(self.linear1(x))  # Apply ReLU activation to the hidden layer
        x = self.linear2(x)  # Compute the final output without activation
        return x

    def save(self, file_name='model.pth'):
        # Save the model's weights to a file
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):  # Create directory if it doesn't exist
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)  # Save the model's state_dict to the file

# Define a Q-learning trainer class
class QTrainer:
    def __init__(self, model, lr, gamma):
        # Initialize the trainer with a model, learning rate, and discount factor
        self.lr = lr
        self.gamma = gamma  # Discount factor for future rewards
        self.model = model  # The neural network to be trained
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)  # Adam optimizer
        self.criterion = nn.MSELoss()  # Mean Squared Error loss function

    def train_step(self, state, action, reward, next_state, done):
        # Convert input data to PyTorch tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # Add batch dimension if only a single sample is provided
        if len(state.shape) == 1:
            # Reshape single sample to batch format
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )  # Convert to tuple

        # 1: Predict Q-values for the current state
        pred = self.model(state)

        # Clone predictions to create targets
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:  # If not a terminal state
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            # Update target for the selected action
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 2: Compute loss between target and predicted Q-values
        self.optimizer.zero_grad()  # Reset gradients
        loss = self.criterion(target, pred)  # Compute loss
        loss.backward()  # Backpropagate the loss
        self.optimizer.step()  # Update the model's parameters

# Define a Dueling Noisy Q-Network for Rainbow DQN
class Dueling_Noisy_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Shared feature extraction layer
        self.feature_layer = nn.Linear(input_size, hidden_size)
        # Separate streams for value and advantage
        self.value_stream = nn.Linear(hidden_size, 1)  # Computes state value
        self.advantage_stream = nn.Linear(hidden_size, output_size)  # Computes action advantages
        # List of noisy layers for prioritized exploration
        self.noisy_layers = [self.value_stream, self.advantage_stream]

    def forward(self, x):
        # Forward pass: compute the Q-value using the dueling architecture
        x = F.relu(self.feature_layer(x))  # Feature extraction
        value = self.value_stream(x)  # Compute state value
        advantage = self.advantage_stream(x)  # Compute action advantages
        # Combine value and advantage to compute Q-values
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

    def save(self, file_name='model.pth'):
        # Save the model's weights to a file
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):  # Create directory if it doesn't exist
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)  # Save the model's state_dict to the file
