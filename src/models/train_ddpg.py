import gym
import numpy as np
import pandas as pd
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from recommender_env import RecommenderEnv

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data.values

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train DDPG agent for recommendation system.')
    parser.add_argument('--user_data', type=str, required=True, help='Path to user profiles data')
    parser.add_argument('--item_data', type=str, required=True, help='Path to item features data')
    parser.add_argument('--timesteps', type=int, default=10000, help='Number of training timesteps')
    args = parser.parse_args()

    user_profiles = load_data(args.user_data)
    item_features = load_data(args.item_data)

    env = RecommenderEnv(user_profiles, item_features)

    # Add some noise for exploration
    n_actions = env.action_space.n
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = DDPG('MlpPolicy', env, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=args.timesteps)

    model.save("ddpg_recommender")
