import numpy as np
import pandas as pd
import gym
from gym import spaces
import yfinance as yf

class TradingEnv(gym.Env):
    def __init__(self, data, initial_balance=10000):
        super(TradingEnv, self).__init__()
        self.data = data.reset_index()  # Reset index to avoid issues with step indexing
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.total_profit = 0
        self.net_worth = initial_balance

        # Define action and observation space
        self.action_space = spaces.Discrete(3) #0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        return self._next_observation()
        
    def _next_observation(self):
        obs = self.data.iloc[self.current_step][['Open', 'High', 'Low', 'Close', 'Volume']].values
        return obs.astype(np.float32)
        
    def step(self, action):

        if self.current_step >= len(self.data) - 1:
            return self._next_observation(), 0, True, {}

        current_price = self.data.iloc[self.current_step]['Close'].item()
        prev_net_worth = self.net_worth
        reward = 0

        if action == 2 and self.shares_held > 0: # Sell
            self.balance += self.shares_held * current_price
            self.shares_held = 0
        elif action == 1 and self.balance >= current_price: # Buy
            num_shares = self.balance // current_price
            self.balance -= num_shares * current_price
            self.shares_held += num_shares

        self.net_worth = self.balance + (self.shares_held * current_price)
        reward = self.net_worth - prev_net_worth

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        return self._next_observation(), reward, done, {}
        
data = yf.download('AAPL', start='2023-01-01', end='2023-12-31', interval='1d')
env = TradingEnv(data)

obs = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    print(f"Observation: {obs}, Reward: {reward}, Done: {done}")

