import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import math


def compute_loop_inductance(path, trace_width_px, copper_thickness_um=105.0, cell_size_mm=0.5):
    # Rosa's formula per segment, summed over the path. No mutual inductance
    # between segments -- a simplified stand-in for a full PEEC solve, fast
    # enough to call thousands of times during training. See README.
    mu0 = 4 * math.pi * 1e-7
    w = max(trace_width_px * cell_size_mm * 1e-3, 1e-5)
    t = copper_thickness_um * 1e-6

    L_total = 0.0
    for i in range(1, len(path)):
        dx = (path[i][0] - path[i-1][0]) * cell_size_mm * 1e-3
        dy = (path[i][1] - path[i-1][1]) * cell_size_mm * 1e-3
        l = math.hypot(dx, dy)
        if l == 0:
            continue
        L_seg = (mu0 * l / (2 * math.pi)) * (math.log(2 * l / (w + t)) + 0.5 + (w + t) / (3 * l))
        L_total += L_seg

    return L_total * 1e9


class PCBRouterEnv(gym.Env):
    def __init__(self, grid_size=32, pinn_model=None):
        super(PCBRouterEnv, self).__init__()
        self.grid_size = grid_size

        # 0=up 1=down 2=left 3=right 4=widen 5=narrow
        self.action_space = spaces.Discrete(6)

        # NOTE: had this as uint8 0-255 originally, MLP wouldn't learn anything
        # (first layer just saturates). float32 0-1 fixed it.
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(3, grid_size, grid_size),
            dtype=np.float32
        )
        self.pinn_model = pinn_model

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_pos = [self.grid_size // 2, 1]
        self.target_pos = [self.grid_size // 2, self.grid_size - 2]
        self.trace_width = 1
        self.path = [tuple(self.current_pos)]
        self.T_history = []

        self.state = np.zeros((3, self.grid_size, self.grid_size), dtype=np.float32)
        self.state[1, self.grid_size // 2, self.grid_size // 2] = 1.0  # MOSFET marker
        return self.state, {}

    def _predict_temperature(self):
        x_norm = self.current_pos[0] / self.grid_size
        y_norm = self.current_pos[1] / self.grid_size
        w_norm = self.trace_width / 3.0
        xyw = torch.tensor([[x_norm, y_norm, w_norm]], dtype=torch.float32)
        with torch.no_grad():
            T_pred = self.pinn_model(xyw)
        return T_pred.item()

    def step(self, action):
        is_movement = action in (0, 1, 2, 3)
        prev_pos = tuple(self.current_pos)
        dist_before = abs(prev_pos[0] - self.target_pos[0]) + abs(prev_pos[1] - self.target_pos[1])

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
        new_pos = tuple(self.current_pos)
        self.state[0, self.current_pos[0], self.current_pos[1]] = 1.0

        terminated = False

        # standing still (width change or blocked move) has to cost more than
        # moving forward, otherwise PPO just learns to sit there and collect
        # -0.1 forever instead of risking the -10 self-intersection penalty
        if is_movement and new_pos != prev_pos:
            reward = -0.1
        else:
            reward = -1.0

        # dense shaping term on top of the sparse terminal reward -- needed for
        # PPO to actually find the target within a reasonable timestep budget.
        # not in Eq. (2) of the paper, added purely for training tractability
        dist_after = abs(new_pos[0] - self.target_pos[0]) + abs(new_pos[1] - self.target_pos[1])
        reward += 1.0 * (dist_before - dist_after)

        if is_movement and new_pos != prev_pos:
            was_visited = new_pos in set(self.path)
            self.path.append(new_pos)
            if was_visited:
                reward -= 10.0
                terminated = True

        if self.pinn_model is not None:
            self.T_history.append(self._predict_temperature())

        if not terminated and self.current_pos == self.target_pos:
            L_loop = compute_loop_inductance(self.path, self.trace_width)
            T_max = max(self.T_history) if self.T_history else 92.0

            w1, w2 = 0.65, 0.35  # alpha, beta from Table I
            reward = 100.0 - (w1 * L_loop + w2 * T_max)
            terminated = True

        truncated = len(self.path) > 200
        return self.state, reward, terminated, truncated, {}
