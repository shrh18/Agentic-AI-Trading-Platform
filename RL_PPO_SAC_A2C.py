import gymnasium as gym
# from gym import spaces
from gymnasium import spaces
import numpy as np
import pandas as pd
import yfinance as yf
# from gym import spaces
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.env_checker import check_env

class TradingEnv(gym.Env):
    def __init__(self, data, initial_balance=10000, use_continuous=False):
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
        self.use_continuous = use_continuous

        # Define action space (Discrete for PPO/A2C, Continuous for SAC)
        if use_continuous:
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell

        # Observation space (Stock indicators: Open, High, Low, Close, Volume)
        self.observation_space = gym.spaces.Box(low=-1e6, high=1e6, shape=(5,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.buy_price = 0
        return self._next_observation(), {}

    def _next_observation(self):
        obs = self.data.iloc[self.current_step][['Open', 'High', 'Low', 'Close', 'Volume']].values.astype(np.float32)
        obs[:4] /= 1000
        obs[4] = np.clip(obs[4] / 1e6, -1e6, 1e6)  # Normalize volume to be within a reasonable range
        return obs

    def step(self, action):
        if self.current_step >= len(self.data) - 1:
            return self._next_observation(), 0, True, False, {}

        current_price = self.data.iloc[self.current_step]['Close'].item()
        self.prev_net_worth = self.net_worth
        reward = 0

        if self.use_continuous:
            action = np.clip(action, -1, 1)
            if action > 0.5 and self.balance >= current_price:  # Buy
                num_shares = self.balance // current_price
                cost = num_shares * current_price
                fee = cost * self.transaction_fee
                if num_shares > 0:
                    self.balance -= (cost + fee)
                    self.shares_held += num_shares
                    self.buy_price = ((self.buy_price * (self.shares_held - num_shares)) + (num_shares * current_price)) / self.shares_held
                    reward -= fee  # Small penalty for trading

            elif action < -0.5 and self.shares_held > 0:  # Sell
                sale_value = self.shares_held * current_price
                fee = sale_value * self.transaction_fee
                profit = (current_price - self.buy_price) * self.shares_held
                self.balance += (sale_value - fee)
                self.shares_held = 0
                self.buy_price = 0
                reward += profit - fee
        else:
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
        truncated = False
        return self._next_observation(), reward, done, truncated, {}

    def render(self):
        print(f"Step: {self.current_step}, Balance: {self.balance}, Shares Held: {self.shares_held}, Net Worth: {self.net_worth}")

# Fetch AAPL stock data
data = yf.download('AAPL', start='2023-01-01', end='2023-12-31', interval='1d')




# Train PPO
env = TradingEnv(data)
check_env(env, warn=True)
ppo_model = PPO("MlpPolicy", env, verbose=1)
ppo_model.learn(total_timesteps=10000)

# Train A2C
a2c_model = A2C("MlpPolicy", env, verbose=1)
a2c_model.learn(total_timesteps=10000)

# Train SAC (requires continuous action space)
env_continuous = TradingEnv(data, use_continuous=True)
sac_model = SAC("MlpPolicy", env_continuous, verbose=1)
sac_model.learn(total_timesteps=10000)




def evaluate_model(model, env, name):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        env.render()

    print(f"\n{name} Total Reward: {total_reward}\n")

# Evaluate PPO
evaluate_model(ppo_model, env, "PPO")

# Evaluate A2C
evaluate_model(a2c_model, env, "A2C")

# Evaluate SAC
evaluate_model(sac_model, env_continuous, "SAC")
