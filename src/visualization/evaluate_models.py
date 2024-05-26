import pandas as pd
import numpy as np
import torch
import argparse
from stable_baselines3 import DQN, DDPG
from recommender_env import RecommenderEnv
from sklearn.metrics import precision_score, recall_score

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data.values

def precision_at_k(y_true, y_pred, k):
    top_k = np.argsort(y_pred)[-k:]
    relevant = np.sum(y_true[top_k])
    return relevant / k

def recall_at_k(y_true, y_pred, k):
    top_k = np.argsort(y_pred)[-k:]
    relevant = np.sum(y_true[top_k])
    return relevant / np.sum(y_true)

def evaluate_model(env, model, k=10):
    obs = env.reset()
    total_reward = 0
    precisions = []
    recalls = []

    for _ in range(env.num_users):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        # Assuming the reward is a proxy for relevance
        y_true = env.user_profiles[env.current_user_idx]
        y_pred = env.item_features[action]

        precisions.append(precision_at_k(y_true, y_pred, k))
        recalls.append(recall_at_k(y_true, y_pred, k))

        if done:
            obs = env.reset()

    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    
    return total_reward, avg_precision, avg_recall

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate RL recommendation system.')
    parser.add_argument('--user_data', type=str, required=True, help='Path to user profiles data')
    parser.add_argument('--item_data', type=str, required=True, help='Path to item features data')
    parser.add_argument('--model_type', type=str, choices=['qlearning', 'ddpg'], required=True, help='Type of RL model to evaluate')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained RL model')
    parser.add_argument('--k', type=int, default=10, help='Top-K items for precision and recall')
    args = parser.parse_args()

    user_profiles = load_data(args.user_data)
    item_features = load_data(args.item_data)

    env = RecommenderEnv(user_profiles, item_features)

    if args.model_type == 'qlearning':
        model = DQN.load(args.model_path)
    elif args.model_type == 'ddpg':
        model = DDPG.load(args.model_path)

    total_reward, avg_precision, avg_recall = evaluate_model(env, model, k=args.k)
    
    print(f"Total Reward: {total_reward}")
    print(f"Average Precision@{args.k}: {avg_precision}")
    print(f"Average Recall@{args.k}: {avg_recall}")
