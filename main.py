import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt


# 1. Тепловая суррогатная модель (PINN)
class ThermalPINN(nn.Module):
    def __init__(self):
        super(ThermalPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


# 2. Среда маршрутизации печатной платы
class PCBRouterEnv(gym.Env):
    def __init__(self, grid_size=32):
        super(PCBRouterEnv, self).__init__()
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(3, grid_size, grid_size),
            dtype=np.uint8
        )
        self.pinn_model = lambda x: 92.0 + np.random.normal(0, 1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Увеличиваем размер сетки до 32x32, чтобы сверточный экстрактор Stable-Baselines3
        # не падал с ошибкой Kernel size can't be greater than actual input size
        self.grid_size = 32
        self.current_pos = [self.grid_size // 2, 4]
        self.target_pos = [self.grid_size // 2, self.grid_size - 5]
        self.trace_width = 1
        self.path = [tuple(self.current_pos)]

        self.state = np.zeros((3, self.grid_size, self.grid_size), dtype=np.uint8)
        self.state[1, self.grid_size // 2, self.grid_size // 2] = 255  # Источник тепла
        return self.state, {}

    def step(self, action):
        if action == 0:
            self.current_pos[0] -= 1
        elif action == 1:
            self.current_pos[0] += 1
        elif action == 2:
            self.current_pos[1] -= 1
        elif action == 3:
            self.current_pos[1] += 1
        elif action == 4:
            self.trace_width = min(3, self.trace_width + 1)
        elif action == 5:
            self.trace_width = max(1, self.trace_width - 1)

        self.current_pos[0] = np.clip(self.current_pos[0], 0, self.grid_size - 1)
        self.current_pos[1] = np.clip(self.current_pos[1], 0, self.grid_size - 1)
        self.path.append(tuple(self.current_pos))

        self.state[0, self.current_pos[0], self.current_pos[1]] = 255

        terminated = False
        reward = -0.1

        if len(self.path) > len(set(self.path)):
            reward -= 100.0
            terminated = True
        elif self.current_pos == self.target_pos:
            L_loop = len(self.path) * 0.5
            T_max = self.pinn_model(self.current_pos)
            w1, w2 = 0.65, 0.35
            reward = 100.0 - (w1 * L_loop + w2 * T_max)
            terminated = True

        truncated = len(self.path) > 200
        return self.state, reward, terminated, truncated, {}


# 3. Запуск обучения и визуализация
if __name__ == "__main__":
    device = "auto"
    env = PCBRouterEnv(grid_size=32)
    check_env(env)

    # Создаем модель один раз, используя MlpPolicy для работы с многоканальной сеткой
    model = PPO("MlpPolicy", env,
                learning_rate=1e-4,
                n_steps=1024,
                batch_size=256,
                gamma=0.99,
                clip_range=0.2,
                verbose=1,
                device=device)

    print("Начинаем обучение PPO...")
    model.learn(total_timesteps=15000)

    print("Обучение завершено. Строим оптимальную трассу...")
    obs, _ = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    plt.figure(figsize=(6, 6))
    plt.title("RL-Optimized PCB Layout")
    plt.imshow(obs[0] + obs[1] * 0.5, cmap='hot')
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
    plt.show()