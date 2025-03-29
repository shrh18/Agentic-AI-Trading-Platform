import gym
import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

class TradingEnv(gym.Env):
    def __init__(self, data, initial_balance=10000):
        super(TradingEnv, self).__init__()
        self.data = data.reset_index()
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance
        self.prev_net_worth = initial_balance
        self.buy_price = 0
        self.transaction_fee = 0.001  # 0.1% per trade

        self.action_space = gym.spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.buy_price = 0
        return self._next_observation()

    def _next_observation(self):
        obs = self.data.iloc[self.current_step][['Open', 'High', 'Low', 'Close', 'Volume']].values
        return obs.astype(np.float32)

    def step(self, action):
        if self.current_step >= len(self.data) - 1:
            return self._next_observation(), 0, True, {}

        current_price = self.data.iloc[self.current_step]['Close'].item()
        self.prev_net_worth = self.net_worth
        reward = 0

        if action == 1 and self.balance >= current_price:  # Buy
            num_shares = self.balance // current_price
            cost = num_shares * current_price
            fee = cost * self.transaction_fee
            if num_shares > 0:
                self.balance -= (cost + fee)
                self.shares_held += num_shares
                self.buy_price = ((self.buy_price * (self.shares_held - num_shares)) + (num_shares * current_price)) / self.shares_held
                reward -= fee  # Small penalty for trading

        elif action == 2 and self.shares_held > 0:  # Sell
            sale_value = self.shares_held * current_price
            fee = sale_value * self.transaction_fee
            profit = (current_price - self.buy_price) * self.shares_held
            self.balance += (sale_value - fee)
            self.shares_held = 0
            self.buy_price = 0
            reward += profit - fee

        self.net_worth = self.balance + (self.shares_held * current_price)
        reward += (self.net_worth - self.prev_net_worth) * 0.01

        if action == 0:
            reward -= 0.02  # Small penalty for inactivity

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        return self._next_observation(), reward, done, {}

    def render(self):
        print(f"Step: {self.current_step}, Balance: {self.balance}, Shares Held: {self.shares_held}, Net Worth: {self.net_worth}")

# Fetch AAPL stock data
data = yf.download('AAPL', start='2023-01-01', end='2023-12-31', interval='1d')
env = TradingEnv(data)

# Check if the environment is compatible with RL
check_env(env, warn=True)

# Train a DQN agent
model = DQN("MlpPolicy", env, verbose=1, learning_rate=0.001, buffer_size=10000, batch_size=32, gamma=0.99)
model.learn(total_timesteps=10000)

# Test the trained agent
obs = env.reset()
done = False

print("Initial net worth:", env.net_worth)
while not done:
    action, _states = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    env.render()

print("Final net worth:", env.net_worth)
print("Total profit/loss:", env.net_worth - env.initial_balance)
print("Total shares held:", env.shares_held)
# This code sets up a trading environment, trains a DQN agent, and evaluates its performance.
