import gymnasium as gym
from gymnasium import spaces
import numpy as np


class PCBRouterEnv(gym.Env):
    def __init__(self, grid_size=32):
        super(PCBRouterEnv, self).__init__()
        self.grid_size = grid_size

        # Действия: 0=Вверх, 1=Вниз, 2=Влево, 3=Вправо, 4=Шире, 5=Уже
        self.action_space = spaces.Discrete(6)

        # Состояние: 3 канала (Слой трассировки, Тепловые источники, Препятствия)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(3, grid_size, grid_size),
            dtype=np.uint8
        )

        # Заглушка для инференции суррогата (выдает температуру около 92°C)
        self.pinn_model = lambda x: 92.0 + np.random.normal(0, 1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_pos = [self.grid_size // 2, 1]
        self.target_pos = [self.grid_size // 2, self.grid_size - 2]
        self.trace_width = 1
        self.path = [tuple(self.current_pos)]

        self.state = np.zeros((3, self.grid_size, self.grid_size), dtype=np.uint8)
        self.state[1, self.grid_size // 2, self.grid_size // 2] = 255  # Источник тепла (MOSFET)
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
        reward = -0.1  # Штраф за каждый шаг (минимизация длины трассы)

        # Проверка DRC (самопересечение)
        if len(self.path) > len(set(self.path)):
            reward -= 100.0
            terminated = True

        # Успешное соединение терминалов
        elif self.current_pos == self.target_pos:
            L_loop = len(self.path) * 0.5  # PEEC Analytical estimator
            T_max = self.pinn_model(self.current_pos)  # Оценка PINN суррогата

            w1, w2 = 0.65, 0.35
            reward = 100.0 - (w1 * L_loop + w2 * T_max)  # Уравнение награды из статьи
            terminated = True

        truncated = len(self.path) > 200
        return self.state, reward, terminated, truncated, {}