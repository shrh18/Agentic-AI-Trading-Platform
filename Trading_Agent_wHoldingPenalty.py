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
        self.prev_net_worth = initial_balance
        self.buy_price = 0  # Track the price at which shares were bought

        # Define action and observation space
        self.action_space = spaces.Discrete(3) #0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.buy_price = 0  # Reset the buy price
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

        if action == 1 and self.balance >= current_price: # Buy
            num_shares = self.balance // current_price
            self.balance -= num_shares * current_price
            self.shares_held += num_shares
            self.buy_price = current_price  # Set the buy price to the current price
            reward -= 0.001*(num_shares * current_price)  # Penalty for buying (0.1% of the transaction)

        elif action == 2 and self.shares_held > 0: # Sell
            sale_value = self.shares_held * current_price
            self.balance += sale_value
            reward += sale_value - (self.shares_held * self.buy_price)
            self.shares_held = 0
            reward -= 0.001 * sale_value

        self.net_worth = self.balance + (self.shares_held * current_price)
        reward += self.net_worth - self.prev_net_worth  # Reward based on net worth change

        if action == 0: # Small penalty for holding
            reward -= 0.05

        price_window = self.data.iloc[max(0, self.current_step-5):self.current_step+1]['Close']
        volatility = np.std(price_window, axis=0)  # Calculate volatility over the last 5 prices
        reward -= 0.1 * volatility

        self.prev_net_worth = self.net_worth
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