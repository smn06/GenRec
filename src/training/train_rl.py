import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import pandas as pd
from collections import deque
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from src.models.dqn_model import DQN  # Assuming DQN model is defined in src/models/dqn_model.py

# Function to choose action based on epsilon-greedy policy
def choose_action(state, epsilon, dqn, device):
    if np.random.rand() <= epsilon:
        return np.random.uniform(1, 5)  # Random rating between 1 and 5
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    q_values = dqn(state)
    return q_values.cpu().detach().numpy()[0]

# Function to preprocess state
def preprocess_state(state, scaler):
    return scaler.transform(state.reshape(-1, 1)).flatten()

# Function to train RL agent
def train_rl(train_df_path, output_model_path, input_dim, num_episodes=1000, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, batch_size=64, memory_size=10000):
    # Check if CUDA is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the processed training data
    train_df = pd.read_csv(train_df_path)

    # Initialize the DQN, memory, and optimizer
    dqn = DQN(input_dim=input_dim - 1, output_dim=1).to(device)
    optimizer = optim.Adam(dqn.parameters(), lr=lr)
    memory = deque(maxlen=memory_size)
    criterion = nn.MSELoss()

    # Initialize MinMaxScaler for state normalization
    scaler = MinMaxScaler()
    train_data_scaled = scaler.fit_transform(train_df.iloc[:, :-1].values)

    # Training the RL Agent
    for episode in range(num_episodes):
        state = train_data_scaled[np.random.randint(0, len(train_data_scaled))]
        total_reward = 0
        done = False

        while not done:
            action = choose_action(state, epsilon, dqn, device)
            actual_rating = train_df.iloc[np.argmax(state), -1]  # Use the actual rating as the target for training
            reward = -abs(actual_rating - action)  # Reward based on the closeness of the action to the actual rating
            total_reward += reward

            next_state = train_data_scaled[np.random.randint(0, len(train_data_scaled))]

            memory.append((state, action, reward, next_state, done))

            state = next_state

            if len(memory) > batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.FloatTensor(states).to(device)
                actions = torch.FloatTensor(actions).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

                q_values = dqn(states)
                q_value = q_values.gather(1, actions.long())

                next_q_values = dqn(next_states)
                next_q_value = next_q_values.max(1)[0].unsqueeze(1)

                expected_q_value = rewards + (gamma * next_q_value * (1 - dones))

                loss = criterion(q_value, expected_q_value.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {epsilon}")

    # Save the trained DQN model
    torch.save(dqn.state_dict(), output_model_path)

    print(f"Training complete. Model saved to '{output_model_path}'")


if __name__ == "__main__":
    # Example usage
    train_df_path = '../data/processed/train_user_profiles.csv'
    output_model_path = '../models/dqn.pth'
    input_dim = 5  # Example input dimension, adjust according to your data

    train_rl(train_df_path, output_model_path, input_dim)
