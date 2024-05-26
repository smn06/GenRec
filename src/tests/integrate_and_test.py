import pandas as pd
import numpy as np
import torch
import argparse
from stable_baselines3 import DQN, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from recommender_env import RecommenderEnv
from gan_model import Generator

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data.values

def generate_synthetic_profiles(generator_path, latent_dim, num_profiles, output_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(latent_dim, 3).to(device)  # Adjust output_dim based on user profile features
    generator.load_state_dict(torch.load(generator_path))
    generator.eval()

    z = torch.randn(num_profiles, latent_dim).to(device)
    synthetic_profiles = generator(z).detach().cpu().numpy()
    
    columns = ['feature1', 'feature2', 'rating']  # Adjust based on user profile structure
    synthetic_df = pd.DataFrame(synthetic_profiles, columns=columns)
    synthetic_df.to_csv(output_file, index=False)
    return synthetic_profiles

def train_qlearning(env, timesteps):
    model = DQN('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=timesteps)
    model.save("qlearning_recommender")
    return model

def train_ddpg(env, timesteps):
    n_actions = env.action_space.n
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    model = DDPG('MlpPolicy', env, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=timesteps)
    model.save("ddpg_recommender")
    return model

def test_model(env, model):
    obs = env.reset()
    total_reward = 0
    for _ in range(env.num_users):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            obs = env.reset()
    return total_reward




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Integrate and test RL recommendation system.')
    parser.add_argument('--gan_model', type=str, required=True, help='Path to trained GAN generator model')
    parser.add_argument('--latent_dim', type=int, default=100, help='Dimensionality of latent space')
    parser.add_argument('--num_profiles', type=int, default=10000, help='Number of synthetic profiles to generate')
    parser.add_argument('--item_data', type=str, required=True, help='Path to item features data')
    parser.add_argument('--timesteps', type=int, default=10000, help='Number of training timesteps for RL models')
    args = parser.parse_args()

    # Generate synthetic user profiles
    synthetic_profiles = generate_synthetic_profiles(args.gan_model, args.latent_dim, args.num_profiles, 'synthetic_profiles.csv')
    item_features = load_data(args.item_data)

    # Create environment
    env = RecommenderEnv(synthetic_profiles, item_features)

    # Train Q-learning model
    qlearning_model = train_qlearning(env, args.timesteps)
    qlearning_reward = test_model(env, qlearning_model)
    print(f"Q-learning Total Reward: {qlearning_reward}")

    # Train DDPG model
    ddpg_model = train_ddpg(env, args.timesteps)
    ddpg_reward = test_model(env, ddpg_model)
    print(f"DDPG Total Reward: {ddpg_reward}")
