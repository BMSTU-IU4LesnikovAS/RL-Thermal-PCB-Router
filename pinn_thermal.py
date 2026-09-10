import torch
import torch.nn as nn

# Автоматический выбор GPU-ускорителя (включая нативный Apple Silicon MPS)
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


class ThermalPINN(nn.Module):
    def __init__(self):
        super(ThermalPINN, self).__init__()
        # Легкий MLP суррогат для быстрой инференции внутри RL-цикла
        self.net = nn.Sequential(
            nn.Linear(3, 64),  # Вход: координаты (x, y) и ширина трассы (w)
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)  # Выход: предсказанная температура T
        ).to(device)

    def forward(self, x):
        return self.net(x)

    def compute_physics_loss(self, xyw, T_pred, q_gen, k_copper):
        """
        Реализация физических потерь (уравнение Пуассона для теплопроводности).
        Использует autograd для вычисления производных по пространству.
        """
        xyw = xyw.to(device)
        xyw.requires_grad_(True)
        T = self.forward(xyw)

        dT_dxyw = torch.autograd.grad(T, xyw, grad_outputs=torch.ones_like(T), create_graph=True)[0]
        dT_dx, dT_dy = dT_dxyw[:, 0], dT_dxyw[:, 1]

        d2T_dx2 = torch.autograd.grad(dT_dx, xyw, grad_outputs=torch.ones_like(dT_dx), create_graph=True)[0][:, 0]
        d2T_dy2 = torch.autograd.grad(dT_dy, xyw, grad_outputs=torch.ones_like(dT_dy), create_graph=True)[0][:, 1]

        physics_residual = k_copper * (d2T_dx2 + d2T_dy2) + q_gen
        loss_physics = torch.mean(physics_residual ** 2)

        return loss_physics