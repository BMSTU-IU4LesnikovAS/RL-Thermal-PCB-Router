# RL-Thermal-PCB-Router (v2)

RL-based routing of one high-current net on a 2-D grid for a 50 kW, 800 V SiC half-bridge case study.
Every number in `results/summary.json` is produced by the code in this folder.

## Reproduce
```bash
pip install -r requirements.txt
python train_pinn.py          # PINN for 32x32 and 16x16 grids, a few minutes on CPU
PY=python ./run_all.sh        # 14 RL runs, 150,000 steps each, about 45 min on 2 cores
python analyze.py             # summary.json, Monte Carlo, Fig. 2 and Fig. 3 (PNG 300 dpi, TIFF 600 dpi)
```

## Model
- Assumptions: all in `params.py`. They are assumed inputs, not measurements.
- Inductance: trace over a solid return plane, L' = mu0 h / (w + 2h), plus a fixed package term. Not PEEC.
- Temperature (reference): steady 2-D fin equation solved by finite differences; adiabatic edges;
  heat leaves to a sink at T_sink = T_amb + R_sa * P_total, so the sink follows the ambient temperature.
- Switching: series R-L-C drain node with a linear channel-current fall at di/dt.
- PINN: inputs (x, y, width); data term from the finite-difference solver, physics and adiabatic-edge terms.
  It evaluates a local trace temperature during training. All reported temperatures come from the
  finite-difference solver, not the PINN.
- RL: MaskablePPO, CNN encoder (32, 64, 128 filters, 256-dim embedding). Action = direction x 3 width levels;
  design-rule violations and self-crossing are masked.
- Comparators: fixed-width shortest-path router and a scalarised Dijkstra search (`routers.py`).

## Limits
Single layer and a single net; no vias; steady state; no measurements. The PINN has an RMSE of about 6 K on held-out
points because the temperature depends on the whole layout, not only on (x, y, w).
The reward scale, the masked action space and the local thermal penalty were changed after pilot runs in which PPO
did not learn a usable width policy. This reward-design step used the same case study as the reported runs.
On the 16x16 grid (5 mm cells) the copper footprint is quantised: the 2.5 and 7.5 mm levels occupy one 5 mm cell and the
12.5 mm level occupies 15 mm, so its FD temperature (67.6 C) is optimistic; on the 32x32 grid the same layout reaches 68.8 C.
In the switching model R_LOOP carries the load current before the fall, so v_ds starts at V_DC - R_LOOP*I_LOAD (768.75 V)
and settles at V_DC. Peak values depend on R_LOOP; the RL-vs-baseline difference (about 6 V) does not change materially.

## Citation
See the paper. MIT License.
