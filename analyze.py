"""Builds results/summary.json and the paper's figures from results/*.json."""
import glob, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import params as P, physics as ph
from routers import wirelength_router, dijkstra, evaluate

plt.rcParams.update({'font.family': 'serif', 'font.size': 8})
g = ph.Geometry(32)
S = {}
# baselines
base = {}
for lv, w in enumerate(P.WIDTHS_MM):
    base[f'wirelength_{w}mm'] = (*wirelength_router(g, lv),)
    S[f'wirelength_{w}mm'] = evaluate(*base[f'wirelength_{w}mm'], g)
T0, ts0 = ph.solve_temperature([], [], g)
S['no_trace'] = dict(Tmax_C=float(T0.max()), Tsink_C=float(ts0))
# scalarised Dijkstra sweep (classical comparator)
d = g.cell_m
sweep = []
for mu in (0, .01, .03, .1, .3, 1, 3, 10):
    def cost(lv, r, c):
        return P.MU0 * P.H_DIEL * d / (P.WIDTHS_MM[lv] * 1e-3 + 2 * P.H_DIEL) * 1e9 \
            + mu * max(T0[r, c] - ts0, 0) / 100
    res = dijkstra(g, cost, [0, 1, 2], width_change_cost=1e-6)
    e = evaluate(*res, g); e['mu'] = mu; sweep.append((e, res))
S['dijkstra_sweep'] = [e for e, _ in sweep]
# RL runs
groups = {}
for f in sorted(glob.glob('results/*_s?.json')):
    r = json.load(open(f)); groups.setdefault(r['tag'].rsplit('_', 1)[0], []).append(r)
for k, rs in groups.items():
    ok = [r for r in rs if r['routed']]
    S[k] = dict(n=len(rs), routed=len(ok), **{m + '_mean': float(np.mean([r[m] for r in ok])) for m in
               ('L_nH', 'Tmax_C', 'Vpk_V', 'length_mm', 'train_s', 'infer_s', 'Tpinn_C')},
               **{m + '_std': float(np.std([r[m] for r in ok], ddof=1)) if len(ok) > 1 else 0.0
                  for m in ('L_nH', 'Tmax_C', 'Vpk_V')})
best = min(groups['proposed'], key=lambda r: r['seed'])
rl = ([tuple(p) for p in best['path']], best['levels'])
wl = base['wirelength_7.5mm']
# Monte Carlo
rng = np.random.default_rng(0)
tamb = rng.normal(25, 2.5, 1000); tcu = rng.normal(105e-6, 5e-6, 1000)
for name, (p, l) in dict(rl=rl, wirelength=wl).items():
    v = np.array([ph.solve_temperature(p, l, g, t_amb=a, t_cu=c)[0].max() for a, c in zip(tamb, tcu)])
    S['mc_' + name] = dict(mean=float(v.mean()), std=float(v.std(ddof=1)), p99=float(np.percentile(v, 99)), max=float(v.max()))
json.dump(S, open('results/summary.json', 'w'), indent=1)
# Fig. 2
fig, ax = plt.subplots(figsize=(3.4, 2.197))
for (lab, L, c, ls) in (('7.5 mm fixed-width route', S['wirelength_7.5mm']['L_nH'], 'tab:red', '--'),
                        ('RL layout', evaluate(*rl, g)['L_nH'], 'tab:blue', '-')):
    t, v, pk = ph.turn_off_transient(L)
    ax.plot(t, v, c, ls=ls, lw=1, label=f'{lab} ($L_{{loop}}$ = {L:.1f} nH)')
ax.axhline(800, color='k', lw=.5, ls=':'); ax.set(xlim=(5, 40), ylim=(720, 880), xlabel='Time (ns)', ylabel='$V_{ds}$ (V)')
ax.legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig('results/fig2_transient.tif', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
fig.savefig('results/fig2_transient.png', dpi=300)
# Fig. 3
Ta = ph.solve_temperature(*wl, g)[0]; Tb = ph.solve_temperature(*rl, g)[0]
vmin, vmax = min(Ta.min(), Tb.min()), max(Ta.max(), Tb.max())
fig, ax = plt.subplots(1, 2, figsize=(3.4, 1.8), constrained_layout=True)
for a, T, t in zip(ax, (Ta, Tb), ('(a)', '(b)')):
    im = a.imshow(T, cmap='inferno', vmin=vmin, vmax=vmax, extent=(0, 80, 80, 0))
    a.set_title(f'{t} $T_{{max}}$ = {T.max():.1f} °C', fontsize=8); a.set_xlabel('x (mm)')
ax[0].set_ylabel('y (mm)'); fig.colorbar(im, ax=ax, label='T (°C)', shrink=.9)
fig.savefig('results/fig3_fields.tif', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
fig.savefig('results/fig3_fields.png', dpi=300)
print(json.dumps({k: v for k, v in S.items() if k != 'dijkstra_sweep'}, indent=0)[:2500])
for e in S['dijkstra_sweep']: print({k: round(v, 2) for k, v in e.items()})
