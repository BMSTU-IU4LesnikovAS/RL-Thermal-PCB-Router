# RL-Thermal-PCB-Router

Reference implementation of the Actor-Critic reinforcement learning framework for thermal-aware PCB layout optimization of SiC-based power modules, described in:

> A. S. Lesnikov and E. K. Shamshetdinov, "Reinforcement Learning-Based Thermal-Aware PCB Layout Optimization for Electric Vehicle Charging Station Power Modules," in Proc. 2026 IEEE 4th Int. Conf. Problems of Informatics, Electronics and Radio Engineering (PIERE), Novosibirsk, Russia, 2026.

## Overview

The repository trains a Proximal Policy Optimization (PPO) agent to route high-current PCB traces for a SiC half-bridge power module, jointly minimizing:

- **Parasitic loop inductance** ($L_{loop}$), estimated via a geometric self-inductance calculation along the routed path;
- **Peak temperature** ($\Delta T_{max}$), predicted by a Physics-Informed Neural Network (PINN) surrogate trained on the 2D steady-state heat conduction equation;

The terminal reward follows Eq. (2) of the paper: $R_{global} = -(\alpha \cdot L_{loop} + \beta \cdot \Delta T_{max})$, with $\alpha = 0.65$, $\beta = 0.35$.

## Repository structure

- `main.py` — entry point: pretrains PINN, trains PPO agent, visualizes result
- `pcb_env.py` — Gymnasium environment: routing grid, reward, inductance estimator
- `pinn_thermal.py` — PINN model and physics-informed pretraining loop
- `requirements.txt` — Python dependencies
- `LICENSE` — MIT license

## Requirements

Python 3.10+. See `requirements.txt` for exact package versions.

Install:
```bash
pip install -r requirements.txt
```

Runs entirely on CPU; no GPU required for this reference-scale environment.

## Usage

```bash
python main.py
```

This will pretrain the PINN thermal surrogate on the governing heat-conduction equation (~500 epochs), train the PPO agent for 50,000 timesteps on the PCB routing environment, then roll out the trained policy, print a short diagnostic summary, and save the resulting optimized layout to `optimized_layout.png`.

A full run (pretraining + training + rollout) takes well under a minute on a standard laptop CPU. Training on the reference hardware described in the paper (NVIDIA RTX 4090, full CNN-based state encoder, 15,000 episodes) takes approximately 4.5 hours; this repository provides a lighter-weight, MLP-based reference implementation intended to demonstrate the core reward-driven architecture rather than to reproduce the exact training wall-clock time reported in the paper.

## Implementation notes and simplifications

This is a research/reference implementation intended to demonstrate the core RL–dual-surrogate architecture, not a production EDA tool. Relative to the full methodology described in the paper:

- **Inductance estimation** uses Rosa's formula for straight conductor segments, summed along the path. Mutual inductance between segments is **not** modeled, unlike a full geometric PEEC solver.
- **Routing** is limited to a single net between two fixed terminals on a single-layer 32×32 grid; multi-net routing, via transitions, and full DRC clearance checking beyond simple self-intersection avoidance are not implemented in this version.
- **State encoding** uses a compact `MlpPolicy` on a flattened, normalized 3-channel grid, rather than the CNN-based spatial encoder described in Section III.A of the paper. Observation values are normalized to `[0, 1]` (float32); an earlier version using raw `0/255` pixel values prevented the policy network from learning anything at all, since it saturated the first linear layer.
- **Reward shaping**: in addition to the terminal reward defined in Eq. (2), this implementation adds a dense, potential-based shaping term (progress toward the target at every step, plus a mild penalty for non-productive actions such as a width change or a move blocked by the grid boundary). This is a standard RL engineering technique required to make PPO training tractable within a limited timestep budget on an otherwise sparse-reward gridworld task; it does not alter the terminal objective being optimized, only the density of the training signal.
- The PINN is pretrained once before RL training begins (frozen surrogate), consistent with the "frozen surrogate" description in Section III.B of the paper. Heat generation in the pretraining PDE is modeled as spatially localized near the fixed MOSFET position (Gaussian profile) and inversely proportional to trace width.

These simplifications do not affect the validity of the core reward-driven optimization mechanism reported in the paper but should be considered before using this code for quantitative benchmarking against commercial EDA tools.

## Reproducing paper results

Default configuration in `main.py` (α=0.65, β=0.35, 32×32 grid) corresponds to the "Proposed Optimum" row in Table II of the paper. To reproduce other ablation rows (e.g., α=0, β=1 or α=1, β=0), edit the `w1, w2` values in `pcb_env.py`'s `step()` method accordingly.

## AI usage disclosure

Portions of this codebase were developed with the assistance of an AI coding tool and reviewed and validated by the authors, including debugging based on real training logs (reward curves, rollout diagnostics) to identify and fix issues such as observation normalization and reward balance.

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
