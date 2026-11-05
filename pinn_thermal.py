"""PINN surrogate. Input (x, y, w) in [0,1]; output theta = (T - T_sink) / 100 K.
Physics term: nondimensional fin equation. Data term: T_exact sampled from the
finite-difference solver (physics.solve_temperature) on routed layouts."""
import numpy as np
import torch
import torch.nn as nn

import params as P
import physics as ph

device = torch.device("cpu")
THETA_REF = 100.0


class ThermalPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 64), nn.Tanh(), nn.Linear(64, 128), nn.Tanh(),
                                 nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x)


def _sources(x, y, w):
    """Smoothed device mask, h(x,y) and q(x,y,w) for the physics residual."""
    m = torch.zeros_like(x)
    for r0, r1, c0, c1 in P.DEVICES:
        s = 0.01
        m = m + torch.sigmoid((x - r0) / s) * torch.sigmoid((r1 - x) / s) \
              * torch.sigmoid((y - c0) / s) * torch.sigmoid((c1 - y) / s)
    area = (P.DEVICES[0][1] - P.DEVICES[0][0]) * (P.DEVICES[0][3] - P.DEVICES[0][2]) * (P.BOARD_MM * 1e-3) ** 2
    h = P.H_BOARD + (P.H_DEVICE - P.H_BOARD) * m
    widths = torch.tensor(P.WIDTHS_MM, dtype=torch.float32) * 1e-3
    w_m = widths[0] + (widths[-1] - widths[0]) * w
    q_joule = P.I_LOAD ** 2 * P.RHO_CU / (w_m ** 2 * P.T_CU)
    return h, m * P.P_DEVICE_W / area + q_joule


def physics_loss(model, n):
    xyw = torch.rand(n, 3, requires_grad=True)
    x, y, w = xyw[:, 0:1], xyw[:, 1:2], xyw[:, 2:3]
    th = model(xyw)
    g = torch.autograd.grad(th.sum(), xyw, create_graph=True)[0]
    txx = torch.autograd.grad(g[:, 0].sum(), xyw, create_graph=True)[0][:, 0:1]
    tyy = torch.autograd.grad(g[:, 1].sum(), xyw, create_graph=True)[0][:, 1:2]
    kt = P.K_FR4 * P.T_FR4 + P.K_CU * P.T_PLANE + P.K_CU * P.T_CU
    L = P.BOARD_MM * 1e-3
    h, q = _sources(x, y, w)
    res = (kt / L ** 2 * (txx + tyy) - h * th + q / THETA_REF) / P.H_BOARD
    return (res ** 2).mean()


def bc_loss(model, n):
    """Adiabatic board edges: zero normal derivative."""
    xyw = torch.rand(n, 3)
    side = torch.randint(0, 2, (n,))
    xyw[torch.arange(n), side] = torch.round(xyw[torch.arange(n), side])
    xyw.requires_grad_(True)
    g = torch.autograd.grad(model(xyw).sum(), xyw, create_graph=True)[0]
    return (g[torch.arange(n), side] ** 2).mean()


def make_dataset(geo, n_layouts=60, seed=0):
    """Collocation data: T_exact at trace cells of randomised routed layouts."""
    from routers import dijkstra
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_layouts):
        noise = rng.random((geo.n, geo.n)) * rng.uniform(0.2, 3.0)
        lv = sorted(rng.choice(3, size=rng.integers(1, 4), replace=False).tolist())
        res = dijkstra(geo, lambda l, r, c: 1 + noise[r, c], lv)
        if res is None:
            continue
        path, levels = res
        T, ts = ph.solve_temperature(path, levels, geo)
        for (r, c), l in zip(path, levels):
            rows.append(((r + .5) / geo.n, (c + .5) / geo.n, l / 2, (T[r, c] - ts) / THETA_REF))
    a = np.array(rows, dtype=np.float32)
    return torch.tensor(a[:, :3]), torch.tensor(a[:, 3:4])


def train_pinn(model, geo, epochs=3000, lr=1e-3, n_p=512, n_b=128, lam_phy=0.01, seed=0, log=print):
    torch.manual_seed(seed)
    X, Y = make_dataset(geo, seed=seed)
    n_val = len(X) // 5
    perm = torch.randperm(len(X))
    Xv, Yv, Xt, Yt = X[perm[:n_val]], Y[perm[:n_val]], X[perm[n_val:]], Y[perm[n_val:]]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for e in range(epochs):
        opt.zero_grad()
        idx = torch.randint(0, len(Xt), (1024,))
        ld = ((model(Xt[idx]) - Yt[idx]) ** 2).mean()
        loss = ld + lam_phy * physics_loss(model, n_p) + bc_loss(model, n_b)
        loss.backward()
        opt.step()
        sched.step()
        if e % 500 == 0:
            log(f"pinn {e} data={ld.item():.5f}")
    with torch.no_grad():
        rmse = THETA_REF * ((model(Xv) - Yv) ** 2).mean().sqrt().item()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return dict(n_data=len(Xt), n_val=len(Xv), n_physics=n_p, n_boundary=n_b, epochs=epochs,
                lambda_phy=lam_phy, val_rmse_K=rmse)


def table(model, geo):
    """theta lookup [level, row, col] so the env does not call torch per step."""
    n = geo.n
    r, c = np.meshgrid((np.arange(n) + .5) / n, (np.arange(n) + .5) / n, indexing='ij')
    out = np.zeros((3, n, n), np.float32)
    for lv in range(3):
        x = torch.tensor(np.stack([r.ravel(), c.ravel(), np.full(n * n, lv / 2)], 1), dtype=torch.float32)
        with torch.no_grad():
            out[lv] = model(x).numpy().reshape(n, n)
    return out * THETA_REF
