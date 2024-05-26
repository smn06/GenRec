import gym
from gym import spaces
import numpy as np
import pandas as pd

class RecommenderEnv(gym.Env):
    def __init__(self, user_profiles, item_features):
        super(RecommenderEnv, self).__init__()

        self.user_profiles = user_profiles
        self.item_features = item_features
        self.num_users = user_profiles.shape[0]
        self.num_items = item_features.shape[0]

        # Action space: recommend one of the items
        self.action_space = spaces.Discrete(self.num_items)

        # Observation space: user profile
        self.observation_space = spaces.Box(low=-1, high=1, shape=(user_profiles.shape[1],), dtype=np.float32)

        # Initialize the current user
        self.current_user_idx = 0

    def reset(self):
        # Reset the environment to an initial state
        self.current_user_idx = np.random.randint(0, self.num_users)
        user_profile = self.user_profiles[self.current_user_idx]
        return user_profile

    def step(self, action):
        # Execute one time step within the environment
        item_feature = self.item_features[action]
        user_profile = self.user_profiles[self.current_user_idx]

        # For simplicity, the reward is the dot product of user profile and item feature
        reward = np.dot(user_profile, item_feature)

        # Move to the next user
        self.current_user_idx = (self.current_user_idx + 1) % self.num_users

        # Check if the episode is done
        done = self.current_user_idx == 0

        return user_profile, reward, done, {}

    def render(self, mode='human', close=False):
        pass
