# RL-Thermal-PCB-Router

Reference implementation of the Actor-Critic reinforcement learning framework for thermal-aware PCB layout optimization of SiC-based power modules, described in:

> A. S. Lesnikov and E. K. Shamshetdinov, "Reinforcement Learning-Based Thermal-Aware PCB Layout Optimization for Electric Vehicle Charging Station Power Modules," in Proc. 2026 IEEE 4th Int. Conf. Problems of Informatics, Electronics and Radio Engineering (PIERE), Novosibirsk, Russia, 2026.

## Overview

The repository trains a Maskable Proximal Policy Optimization (PPO) agent to route one high-current PCB net of a 50 kW, 800 V SiC half-bridge power module, jointly minimizing:

- **Parasitic loop inductance** ($L_{loop}$), computed analytically for a trace over a solid return plane, $L' = \mu_0 h / (w + 2h)$ per unit length, plus a fixed 5 nH package term;
- **Peak temperature rise** ($\Delta T_{max}$), estimated during training by a Physics-Informed Neural Network (PINN) surrogate of the 2D steady-state fin equation.

The terminal reward follows Eq. (2) of the paper: $R_{global} = 40\,[1 - (\alpha L_{loop}/L_{ref} + \beta \Delta T_{max}/T_{ref})]$, with $L_{ref} = 10$ nH, $T_{ref} = 100$ K, $\alpha = 0.65$, $\beta = 0.35$.

All reported temperatures are computed with a finite-difference (FD) solver of the same physics, not with the PINN. Every number in `results/summary.json` is produced by the code in this repository.

## Repository structure

- `params.py` — all physical and geometric parameters of the case study (assumed values, not measurements)
- `physics.py` — grid geometry, design-rule check, inductance model, FD thermal solver, R-L-C turn-off transient
- `pinn_thermal.py` — PINN model, physics-informed training loop, lookup table used by the environment
- `pcb_env.py` — Gymnasium environment: routing grid, action masking, reward
- `routers.py` — non-learning comparators: fixed-width shortest-path router and scalarized Dijkstra search
- `train_pinn.py` — trains the PINN for the 32×32 and 16×16 grids
- `train.py` — one RL training run (CNN encoder + MaskablePPO), writes `results/<tag>.json`
- `run_all.sh` — all 14 RL runs used in the paper (main configuration and ablation)
- `analyze.py` — baselines, Dijkstra sweep, Monte Carlo, `summary.json`, Fig. 2 and Fig. 3
- `main.py` — quick single run (PINN + one RL run)
- `results/` — trained PINN weights, all run results, `summary.json`, figures
- `requirements.txt` — Python dependencies
- `LICENSE` — MIT license

## Requirements

Python 3.10+. See `requirements.txt` for exact package versions.

Install:
```bash
pip install -r requirements.txt
```

Runs entirely on CPU; no GPU required.

## Usage

Quick single run:
```bash
python main.py
```

Full reproduction of the paper results:
```bash
python train_pinn.py          # PINN for 32x32 and 16x16 grids, a few minutes on CPU
PY=python ./run_all.sh        # 14 RL runs, 150,000 steps each, about 45 min on a 2-core CPU
python analyze.py             # summary.json, Monte Carlo, Fig. 2 and Fig. 3 (PNG 300 dpi, TIFF 600 dpi)
```

One RL run of 150,000 steps takes about 6.4 min on a 2-core CPU; inference of the trained policy takes about 0.03 s.

## Implementation notes and simplifications

This is a research/reference implementation, not a production EDA tool:

- **Routing** is limited to a single net between two fixed terminals on a single-layer grid (32×32 cells of 2.5 mm on an 80×80 mm board). Multi-net routing and vias are not implemented.
- **Actions**: 4 directions × 3 trace widths (2.5, 7.5, 12.5 mm). Moves that violate the device keep-out regions or cross the trace are masked, so the design-rule term $C(P)$ is zero by construction.
- **State encoding**: 5-channel grid image (routed path with width level, head, target, keep-outs, thermal map) processed by a CNN encoder (32, 64, 128 filters, 256-dim embedding).
- **Reward shaping**: in addition to the terminal reward of Eq. (2), each step adds a −0.02 cost, a potential-based progress term (0.5 per cell of Manhattan-distance reduction) and a local thermal penalty $\beta \Delta T / T_{ref}$; a dead end is penalized by 10 and an episode is truncated after 160 steps.
- **Inductance** model neglects bends, vias and mutual coupling; it is not a full PEEC solution.
- **Thermal model** is 2D and steady-state with adiabatic board edges; through-thickness spreading, edge effects and the thermal interface are not resolved. The heat sink follows $T_{sink} = T_{amb} + R_{sa} P_{total}$.
- **PINN** takes $(x, y, w)$ as inputs; its RMSE on held-out FD points is about 6.3 K, because the temperature depends on the whole layout, not only on these inputs.
- **Switching model** is a series R-L-C drain node with a linear current fall at 5 A/ns. R_LOOP carries the load current before the fall, so $v_{ds}$ starts at $V_{DC} - R_{loop} I_{load}$ (768.75 V) and settles at $V_{DC}$. Peak values depend on R_LOOP; the RL-vs-baseline difference (about 6 V) does not change materially.
- **16×16 grid**: the copper footprint is quantized to 5 mm cells (the 2.5 and 7.5 mm levels occupy one cell, the 12.5 mm level occupies 15 mm), so its FD temperature (67.6 °C) is optimistic; on the 32×32 grid the same layout reaches 68.8 °C.
- The reward scale, the masked action space and the local thermal penalty were set after pilot runs on the same case study.

## Reproducing paper results

`run_all.sh` reproduces all rows of Table II:

| Configuration | Command options |
|---|---|
| Proposed (α = 0.65, β = 0.35), 5 seeds | `--seed 0..4` |
| α = 1, β = 0, 3 seeds | `--alpha 1 --beta 0` |
| α = 0, β = 1, 3 seeds | `--alpha 0 --beta 1` |
| 16×16 grid, 3 seeds | `--grid 16 --pinn results/pinn16.pt` |

The fixed-width routes, the Dijkstra sweep and the Monte Carlo analysis (1000 samples) are computed by `analyze.py`.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{lesnikov2026rlpcb,
  author    = {Lesnikov, Alexey S. and Shamshetdinov, Emil K.},
  title     = {Reinforcement Learning-Based Thermal-Aware PCB Layout Optimization for Electric Vehicle Charging Station Power Modules},
  booktitle = {Proc. 2026 IEEE 4th Int. Conf. Problems of Informatics, Electronics and Radio Engineering (PIERE)},
  address   = {Novosibirsk, Russia},
  year      = {2026}
}
```

## License

Released under the MIT License — see [LICENSE](LICENSE) for details.
