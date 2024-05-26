import torch
import gym
import numpy as np
import pandas as pd
from stable_baselines3 import DQN, DDPG
from recommender_env import RecommenderEnv

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data.values

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test RL recommendation system.')
    parser.add_argument('--user_data', type=str, required=True, help='Path to user profiles data')
    parser.add_argument('--item_data', type=str, required=True, help='Path to item features data')
    parser.add_argument('--model_type', type=str, choices=['qlearning', 'ddpg'], required=True, help='Type of RL model to test')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained RL model')
    args = parser.parse_args()

    user_profiles = load_data(args.user_data)
    item_features = load_data(args.item_data)

    env = RecommenderEnv(user_profiles, item_features)

    if args.model_type == 'qlearning':
        model = DQN.load(args.model_path)
    elif args.model_type == 'ddpg':
        model = DDPG.load(args.model_path)

    obs = env.reset()
    total_reward = 0
    for _ in range(env.num_users):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            obs = env.reset()

    print(f"Total Reward: {total_reward}")
