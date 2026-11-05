"""One training run; writes results/<tag>.json. Usage: train.py --tag x --seed 0 ..."""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
from sb3_contrib import MaskablePPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import physics as ph
from pinn_thermal import ThermalPINN, table
from pcb_env import PCBRouterEnv
from routers import evaluate


class GridCNN(BaseFeaturesExtractor):
    """Three conv layers (32, 64, 128) to a 256-dim embedding."""
    def __init__(self, space, features_dim=256):
        super().__init__(space, features_dim)
        c = space.shape[0]
        self.cnn = nn.Sequential(nn.Conv2d(c, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
                                 nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(), nn.Flatten())
        with torch.no_grad():
            n = self.cnn(torch.zeros(1, *space.shape)).shape[1]
        self.fc = nn.Sequential(nn.Linear(n, features_dim), nn.ReLU())

    def forward(self, x):
        return self.fc(self.cnn(x))


def run(a):
    torch.set_num_threads(1)
    geo = ph.Geometry(a.grid)
    pinn = ThermalPINN()
    pinn.load_state_dict(torch.load(a.pinn, weights_only=True))
    tab = table(pinn, geo)
    T0, ts0 = ph.solve_temperature([], [], geo)
    env = PCBRouterEnv(geo, tab, float(T0.max() - ts0), a.alpha, a.beta)
    model = MaskablePPO("CnnPolicy", env, learning_rate=a.lr, n_steps=1024, batch_size=256, n_epochs=4, gamma=0.99,
                clip_range=0.2, ent_coef=a.ent, seed=a.seed, device="cpu", verbose=0,
                policy_kwargs=dict(features_extractor_class=GridCNN, normalize_images=False,
                                   net_arch=dict(pi=[128, 128], vf=[128, 128])))
    t = time.time()
    model.learn(a.steps)
    train_s = time.time() - t
    obs, _ = env.reset()
    done = False
    t = time.time()
    while not done:
        act, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
        obs, r, term, trunc, _ = env.step(act)
        done = term or trunc
    infer_s = time.time() - t
    ok = env.pos == geo.target
    out = dict(tag=a.tag, seed=a.seed, alpha=a.alpha, beta=a.beta, grid=a.grid, steps=a.steps,
               train_s=train_s, infer_s=infer_s, routed=bool(ok),
               path=[list(p) for p in env.path], levels=env.levels)
    if ok:
        out.update(evaluate(env.path, env.levels, geo))
        out['Tpinn_C'] = max(env.trise, env.base) + ph.solve_temperature(env.path, env.levels, geo)[1]
    os.makedirs('results', exist_ok=True)
    json.dump(out, open(f'results/{a.tag}.json', 'w'))
    model.save(f'results/{a.tag}')
    print(a.tag, {k: out.get(k) for k in ('routed', 'L_nH', 'Tmax_C', 'Vpk_V', 'length_mm', 'train_s')}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--alpha', type=float, default=.65); p.add_argument('--beta', type=float, default=.35)
    p.add_argument('--grid', type=int, default=32); p.add_argument('--steps', type=int, default=60000)
    p.add_argument('--ent', type=float, default=0.02); p.add_argument('--lr', type=float, default=3e-4); p.add_argument('--pinn', default='results/pinn32.pt')
    run(p.parse_args())
