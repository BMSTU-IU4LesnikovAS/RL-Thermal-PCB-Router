import json, os, sys, torch
import physics as ph
from pinn_thermal import ThermalPINN, train_pinn
torch.set_num_threads(2)
os.makedirs('results', exist_ok=True)
for grid in (32, 16):
    geo = ph.Geometry(grid); m = ThermalPINN()
    info = train_pinn(m, geo, epochs=int(sys.argv[1]) if len(sys.argv) > 1 else 6000)
    torch.save(m.state_dict(), f'results/pinn{grid}.pt')
    json.dump(info, open(f'results/pinn{grid}.json', 'w')); print(grid, info, flush=True)
