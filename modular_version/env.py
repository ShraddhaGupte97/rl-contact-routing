# env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class ContactRoutingEnv(gym.Env):
    def __init__(self, X, y, aht):
        super(ContactRoutingEnv, self).__init__()
        self.X = X
        self.y = y
        self.aht = aht
        self.num_actions = len(np.unique(y))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(X.shape[1],), dtype=np.float32)
        self.action_space = spaces.Discrete(self.num_actions)
        self.current_index = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_index = np.random.randint(0, len(self.X))
        obs = self.X[self.current_index].astype(np.float32)
        info = {}
        return obs, info

    def step(self, action):
        true_action = self.y[self.current_index]
        true_aht = self.aht[self.current_index]
        normalized_aht = true_aht / (np.max(self.aht) + 1e-6)

        if action == true_action:
            reward = 1.0 - normalized_aht
        else:
            reward = -1.0 - normalized_aht

        terminated = True
        truncated = False
        info = {"true_queue": int(true_action), "chosen_queue": int(action)}
        obs, _ = self.reset()
        return obs, reward, terminated, truncated, info
