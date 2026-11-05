#!/bin/bash
# Reproduces all RL runs. Two parallel jobs; about 45 min on a 2-core CPU.
PY=${PY:-python}
{
for s in 0 1 2 3 4; do echo "--tag proposed_s$s --seed $s"; done
for s in 0 1 2; do
  echo "--tag ind_s$s --seed $s --alpha 1 --beta 0"
  echo "--tag thermal_s$s --seed $s --alpha 0 --beta 1"
  echo "--tag grid16_s$s --seed $s --grid 16 --pinn results/pinn16.pt"
done
} | xargs -P 2 -I{} sh -c "$PY train.py {} --steps 150000"
