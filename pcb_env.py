import gymnasium as gym
from gymnasium import spaces
import numpy as np

import params as P
import physics as ph


class PCBRouterEnv(gym.Env):
    MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))

    """Single net on a 2D grid with keep-outs. Action = direction x width level (12);
    moves that break design rules or cross the trace are masked out.
    Terminal reward: Eq. (2) with both terms dimensionless, plus a completion bonus.
    Step terms: -0.02 per action plus potential-based shaping on Manhattan distance."""

    def __init__(self, geo, dtheta_table, base_tmax_rise, alpha=0.65, beta=0.35, max_steps=160):
        self.geo, self.tab, self.base = geo, dtheta_table, base_tmax_rise
        self.alpha, self.beta, self.max_steps = alpha, beta, max_steps
        self.action_space = spaces.Discrete(12)
        n = geo.n
        self.observation_space = spaces.Box(0.0, 1.0, (5, n, n), np.float32)
        self.legal = np.array([[[geo.legal((r, c), lv) for c in range(n)] for r in range(n)]
                               for lv in range(3)])
        hm = dtheta_table[1]
        self.heat = np.clip(hm / max(hm.max(), 1e-6), 0, 1).astype(np.float32)

    def _obs(self):
        o = np.zeros((5, self.geo.n, self.geo.n), np.float32)
        o[0] = self.mask
        o[1][self.pos] = (self.level + 1) / 3
        o[2][self.geo.target] = 1.0
        o[3] = self.geo.keepout
        o[4] = self.heat
        return o

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = self.geo.start
        self.level = 1
        self.path, self.levels = [self.pos], [self.level]
        self.mask = np.zeros((self.geo.n, self.geo.n), np.float32)
        self.mask[self.pos] = 1
        self.t = 0
        self.trise = 0.0
        return self._obs(), {}

    def _dist(self, p):
        return abs(p[0] - self.geo.target[0]) + abs(p[1] - self.geo.target[1])

    def action_masks(self):
        m = np.zeros(12, bool)
        n = self.geo.n
        for a in range(12):
            d, lv = self.MOVES[a % 4], a // 4
            p = (self.pos[0] + d[0], self.pos[1] + d[1])
            m[a] = 0 <= p[0] < n and 0 <= p[1] < n and self.mask[p] == 0 and self.legal[lv][p]
        return m

    def step(self, action):
        action = int(action)
        self.t += 1
        d, lv = self.MOVES[action % 4], action // 4
        before = self._dist(self.pos)
        self.pos = (self.pos[0] + d[0], self.pos[1] + d[1])
        self.level = lv
        self.path.append(self.pos)
        self.levels.append(lv)
        if len(self.levels) == 2:
            self.levels[0] = lv
        self.mask[self.pos] = (lv + 1) / 3
        self.trise = max(self.trise, float(self.tab[lv][self.pos]))
        # local thermal penalty: the max() in the terminal term alone gives no gradient
        reward = -0.02 + 0.5 * (before - self._dist(self.pos)) - self.beta * float(self.tab[lv][self.pos]) / P.T_REF_C
        term = False
        if self.pos == self.geo.target:
            L = ph.loop_inductance_nh(self.path, self.levels, self.geo)
            T = max(self.trise, self.base)
            reward += 40.0 - 40.0 * (self.alpha * L / P.L_REF_NH + self.beta * T / P.T_REF_C)
            term = True
        elif not self.action_masks().any():
            reward -= 10.0                 # dead end
            term = True
        trunc = self.t >= self.max_steps
        return self._obs(), reward, term, trunc, {}
