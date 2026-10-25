#!/usr/bin/env python
"""2026-10-03: print the full knee-point sweep (same formula as figure_src/33_fig09_orig_print.py)
for the ISA criterion on all parallel observations (zeros included) and on the far-ISA > 0 subset,
plus the building-height sweep on all parallel, so the knee choice can be inspected, not just argmax."""
import numpy as np, pandas as pd
PROJ = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results"
df = pd.read_parquet(f"{PROJ}/swot_urban_nearfar_v6.5_results.parquet",
                     columns=["flow_angle", "far_building", "far_isa", "error_node"], engine="pyarrow")
e = df.error_node.abs().values; ok = ~np.isnan(e)
fa = df.flow_angle.values % 360; ang = np.minimum.reduce([fa, np.abs(fa - 180), 360 - fa])
bld = np.nan_to_num(df.far_building.values); isa = df.far_isa.values
isa = np.nan_to_num(isa * 100 if np.nanmax(isa) <= 1 else isa)
par = (ang <= 13) & ok
p68 = lambda m: float(np.quantile(e[m], 0.68)) if m.sum() else np.nan
def sweep(name, pop, var, ths, unit):
    n = int(pop.sum()); base = p68(pop); rows = []
    for t in ths:
        f = pop & (var > t); rows.append((t, (base - p68(pop & ~f)) / base * 100, f[pop].mean() * 100, int(f.sum())))
    red = np.array([r[1] for r in rows]); rm = np.array([r[2] for r in rows]); k = red / np.nanmax(red) - rm / np.nanmax(rm)
    best = int(np.nanargmax(k))
    print(f"\n[{name}] n {n:,} p68 {base:.3f} m  (knee = red/max(red) - rm/max(rm); max red {np.nanmax(red):.1f}% at {ths[int(np.nanargmax(red))]:g}{unit}, max rm {np.nanmax(rm):.1f}%)")
    print(f"  {'thr':>5} {'red%':>6} {'rm%':>6} {'n_rm':>7} {'knee':>7}")
    for j, (t, r, m, c) in enumerate(rows):
        print(f"  {t:5g} {r:6.2f} {m:6.2f} {c:7,} {k[j]:7.3f}{'  <== best' if j == best else ''}")
sweep("ISA sweep, all parallel (zeros included)", par, isa, np.arange(63, 4, -2), "%")
sweep("ISA sweep, parallel & far ISA > 0", par & (isa > 0), isa, np.arange(63, 4, -2), "%")
sweep("height sweep, all parallel", par, bld, np.arange(20, 4.5, -0.5), " m")
