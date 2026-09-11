import torch
import torch.nn as nn

# MPS is slower than CPU for a net this small — too much per-call overhead
device = torch.device("cpu")


class ThermalPINN(nn.Module):
    def __init__(self):
        super(ThermalPINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),   # x, y, trace width -- all normalized 0..1
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        ).to(device)

    def forward(self, x):
        return self.net(x)

    def compute_physics_loss(self, xyw, T_pred, q_gen, k_copper):
        # residual of the steady-state heat eq: k*(Txx+Tyy) + q = 0
        xyw = xyw.to(device)
        xyw.requires_grad_(True)
        T = self.forward(xyw)

        dT_dxyw = torch.autograd.grad(T, xyw, grad_outputs=torch.ones_like(T), create_graph=True)[0]
        dT_dx, dT_dy = dT_dxyw[:, 0], dT_dxyw[:, 1]

        d2T_dx2 = torch.autograd.grad(dT_dx, xyw, grad_outputs=torch.ones_like(dT_dx), create_graph=True)[0][:, 0]
        d2T_dy2 = torch.autograd.grad(dT_dy, xyw, grad_outputs=torch.ones_like(dT_dy), create_graph=True)[0][:, 1]

        physics_residual = k_copper * (d2T_dx2 + d2T_dy2) + q_gen
        return torch.mean(physics_residual ** 2)


def pretrain_pinn(model, q_base=50.0, k_copper=385.0, T_boundary=290.0, epochs=500, lr=1e-3,
                   source_xy=(0.5, 0.5), source_sigma=0.15):
    # q_gen depends on distance from the MOSFET (gaussian bump) and on 1/width --
    # narrower trace = higher current density = more heat. Without the spatial
    # part the net has no idea where the hot spot actually is.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_domain, n_boundary = 200, 50
    sx, sy = source_xy

    for epoch in range(epochs):
        optimizer.zero_grad()

        xyw_domain = torch.rand(n_domain, 3, device=device)
        x_d, y_d, w_d = xyw_domain[:, 0:1], xyw_domain[:, 1:2], xyw_domain[:, 2:3].clamp(min=0.1)

        dist_sq = (x_d - sx) ** 2 + (y_d - sy) ** 2
        spatial_factor = torch.exp(-dist_sq / (2 * source_sigma ** 2))
        q_field = (q_base / w_d) * spatial_factor

        loss_physics = model.compute_physics_loss(xyw_domain, None, q_field, k_copper)

        # board edges pinned to T_boundary
        xyw_bc = torch.rand(n_boundary, 3, device=device)
        xyw_bc[:, 0] = torch.round(xyw_bc[:, 0])
        T_pred_bc = model(xyw_bc)
        loss_bc = torch.mean((T_pred_bc - T_boundary) ** 2)

        loss = loss_physics + loss_bc
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"[PINN pretrain] epoch {epoch}, loss={loss.item():.4f}", flush=True)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
