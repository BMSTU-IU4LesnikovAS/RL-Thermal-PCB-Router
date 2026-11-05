"""Reference physics: loop inductance, finite-difference heat solver, switching
transient. Plain numpy/scipy, so results do not depend on the neural nets."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

import params as P


class Geometry:
    def __init__(self, grid=32):
        self.n = grid
        self.cell_m = P.BOARD_MM * 1e-3 / grid
        self.keepout = np.zeros((grid, grid), bool)
        for r0, r1, c0, c1 in P.DEVICES:
            self.keepout[int(round(r0 * grid)):int(round(r1 * grid)),
                         int(round(c0 * grid)):int(round(c1 * grid))] = True
        self.start = (int(P.SOURCE[0] * grid), int(P.SOURCE[1] * grid))
        self.target = (int(P.TARGET[0] * grid), int(P.TARGET[1] * grid))

    def radius(self, level):
        """Footprint half-width (cells) of width level 0, 1, 2."""
        return int(round((P.WIDTHS_MM[level] / 2) / (self.cell_m * 1e3) - 0.5 + 1e-9))

    def footprint(self, cell, level):
        r = self.radius(level)
        m = np.zeros((self.n, self.n), bool)
        m[max(cell[0] - r, 0):cell[0] + r + 1, max(cell[1] - r, 0):cell[1] + r + 1] = True
        return m

    def legal(self, cell, level):
        return not (self.footprint(cell, level) & self.keepout).any()


def loop_inductance_nh(path, levels, geo, h=P.H_DIEL):
    """Trace over a solid return plane: L' = mu0*h/(w + 2h) per unit length, summed
    per segment using that segment's own width, plus the package term. Mutual
    coupling is carried by the plane return; bend effects are neglected."""
    L = P.L_PACKAGE_NH * 1e-9
    for i in range(1, len(path)):
        w = P.WIDTHS_MM[levels[i]] * 1e-3
        L += P.MU0 * h * geo.cell_m / (w + 2 * h)
    return L * 1e9


def layout_maps(path, levels, geo, t_cu=P.T_CU):
    """Sheet conductance kt (W/K), sink coefficient h (W/m2/K), source q (W/m2)."""
    n, d = geo.n, geo.cell_m
    kt = np.full((n, n), P.K_FR4 * P.T_FR4 + P.K_CU * P.T_PLANE)
    h = np.full((n, n), P.H_BOARD)
    q = np.zeros((n, n))
    for r0, r1, c0, c1 in P.DEVICES:
        m = np.zeros((n, n), bool)
        m[int(round(r0 * n)):int(round(r1 * n)), int(round(c0 * n)):int(round(c1 * n))] = True
        q[m] += P.P_DEVICE_W / (m.sum() * d * d)
        h[m] = P.H_DEVICE
    if path:
        trace = np.zeros((n, n), bool)
        for i, cell in enumerate(path):
            lv = levels[i]
            fp = geo.footprint(cell, lv)
            trace |= fp
            if i:
                w = P.WIDTHS_MM[lv] * 1e-3
                pw = P.I_LOAD ** 2 * P.RHO_CU * d / (w * t_cu)    # W in this segment
                q[fp] += pw / (fp.sum() * d * d)
        kt[trace] += P.K_CU * t_cu
    return kt, h, q


def sink_temperature(total_power_w, t_amb=P.T_AMB_C, r_sa=P.R_SA):
    return t_amb + r_sa * total_power_w


def solve_temperature(path, levels, geo, t_amb=P.T_AMB_C, t_cu=P.T_CU):
    """Steady 2D fin equation with adiabatic edges:
       div(kt grad T) - h (T - T_sink) + q = 0.  Returns T (deg C) and T_sink."""
    n, d = geo.n, geo.cell_m
    kt, h, q = layout_maps(path, levels, geo, t_cu)
    idx = np.arange(n * n).reshape(n, n)
    east = 2 * kt[:, :-1] * kt[:, 1:] / (kt[:, :-1] + kt[:, 1:]) / d ** 2
    south = 2 * kt[:-1, :] * kt[1:, :] / (kt[:-1, :] + kt[1:, :]) / d ** 2
    i = np.r_[idx[:, :-1].ravel(), idx[:, 1:].ravel(), idx[:-1, :].ravel(), idx[1:, :].ravel()]
    j = np.r_[idx[:, 1:].ravel(), idx[:, :-1].ravel(), idx[1:, :].ravel(), idx[:-1, :].ravel()]
    v = -np.r_[east.ravel(), east.ravel(), south.ravel(), south.ravel()]
    diag = np.zeros(n * n)
    np.add.at(diag, i, -v)
    A = sp.csr_matrix((v, (i, j)), shape=(n * n, n * n)) + sp.diags(diag + h.ravel())
    t_sink = sink_temperature(q.sum() * d * d, t_amb)
    theta = spl.spsolve(A.tocsc(), q.ravel())
    return theta.reshape(n, n) + t_sink, t_sink


def turn_off_transient(L_nh, t_end=120e-9, dt=2e-12):
    """Series R-L-C drain node with a linear channel-current fall at di/dt.
    Euler-Cromer time stepping; returns t (ns), v_ds (V), peak (V)."""
    L = L_nh * 1e-9
    tf = P.I_LOAD / P.DI_DT
    n = int(t_end / dt)
    t = np.arange(n) * dt
    il = P.I_LOAD
    v = P.V_DC - P.R_LOOP * il          # consistent initial condition
    out = np.empty(n)
    for k in range(n):
        ich = max(P.I_LOAD * (1 - t[k] / tf), 0.0)
        il += dt * (P.V_DC - v - P.R_LOOP * il) / L
        v += dt * (il - ich) / P.C_OSS
        out[k] = v
    return t * 1e9, out, out.max()
