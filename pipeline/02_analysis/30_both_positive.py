#!/usr/bin/env python
"""2026-10-02: evaluate single criteria on parallel observations with BOTH
far-nadir building height > 0 AND far-nadir ISA > 0. Reports population sizes, single-criterion
and full-mask effects, and knee-point sweeps (same formula as figure_src/33_fig09_orig_print.py)
on: all parallel (zeros included), bldg > 0, isa > 0, both > 0."""
import numpy as np, pandas as pd
PROJ = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results"
df = pd.read_parquet(f"{PROJ}/swot_urban_nearfar_v6.5_results.parquet",
                     columns=["reach_id", "flow_angle", "far_building", "far_isa", "error_node"], engine="pyarrow")
e = df.error_node.abs().values; ok = ~np.isnan(e)
fa = df.flow_angle.values % 360; ang = np.minimum.reduce([fa, np.abs(fa - 180), 360 - fa])
bld = np.nan_to_num(df.far_building.values); isa = df.far_isa.values
isa = np.nan_to_num(isa * 100 if np.nanmax(isa) <= 1 else isa)
reach = df.reach_id.values
par = (ang <= 13) & ok
pops = {"all parallel": par, "parallel & bldg > 0": par & (bld > 0), "parallel & isa > 0": par & (isa > 0),
        "parallel & bldg > 0 & isa > 0": par & (bld > 0) & (isa > 0)}
p68 = lambda m: float(np.quantile(e[m], 0.68)) if m.sum() else np.nan
def eff(pop, flag):
    b = p68(pop); a = p68(pop & ~flag); return (b - a) / b * 100, flag[pop].mean() * 100
def knee(pop, var, ths):
    base = p68(pop); red, rm = [], []
    for t in ths:
        f = pop & (var > t); red.append((base - p68(pop & ~f)) / base * 100); rm.append(f[pop].mean() * 100)
    red, rm = np.array(red), np.array(rm); k = red / np.nanmax(red) - rm / np.nanmax(rm); i = int(np.nanargmax(k))
    return ths[i], red[i], rm[i]
bt = np.arange(20, 4.5, -0.5); it = np.arange(63, 4, -2)
for name, pop in pops.items():
    n = int(pop.sum()); nr = len(np.unique(reach[pop]))
    h = eff(pop, bld > 8); s = eff(pop, isa > 19); f = eff(pop, (bld > 8) | (isa > 19))
    kb = knee(pop, bld, bt); kc = knee(pop, isa, it)
    print(f"[{name}] n {n:,} ({100*n/par.sum():.1f}% of parallel) reaches {nr} p68 {p68(pop):.3f} m")
    print(f"   height>8 only: red {h[0]:5.1f}% flagged {h[1]:5.1f}% | ISA>19 only: red {s[0]:5.1f}% flagged {s[1]:5.1f}% | full: red {f[0]:5.1f}% flagged {f[1]:5.1f}%")
    print(f"   knee: height {kb[0]:g} m (red {kb[1]:.1f}%, rm {kb[2]:.1f}%) | ISA {kc[0]:g}% (red {kc[1]:.1f}%, rm {kc[2]:.1f}%)")
