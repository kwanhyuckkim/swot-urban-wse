#!/usr/bin/env python
"""2026-10-02: replicate Figure 9 panels (b) and (c) knee-point selection exactly as in
figure_src/33_fig09_orig_print.py, then repeat each sweep on the other population:
 (b)  as published: parallel (flow_par < 13) & far_building >= 0   [zeros included]
 (b') variant:      parallel & far_building > 0                    [zeros excluded]
 (c)  as published: parallel & far_isa > 0                         [zeros excluded]
 (c') variant:      parallel & far_isa >= 0                        [zeros included]"""
import numpy as np, pandas as pd
PROJ = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results"
OPT_A = 13
df = pd.read_parquet(f"{PROJ}/swot_urban_nearfar_v6.5_results.parquet",
                     columns=["flow_angle", "far_building", "far_isa", "error_node"], engine="pyarrow")
abs_err = df.error_node.abs().values
fp = df.flow_angle % 180; flow_par = np.minimum(fp, 180 - fp).values
far_bldg = df.far_building.values
far_isa = (df.far_isa * 100 if df.far_isa.max() <= 1.0 else df.far_isa).values
print(f"rows {len(df):,} | NaN error {np.isnan(abs_err).sum():,} | parallel <13: {(flow_par < OPT_A).sum():,} | <=13: {(flow_par <= OPT_A).sum():,}")
ok = ~np.isnan(abs_err)
def sweep(name, pop, var, thresholds, unit):
    pop = pop & ok
    n = int(pop.sum()); base = np.quantile(abs_err[pop], 0.68)
    red, rm = [], []
    for t in thresholds:
        remove = pop & (var > t); ret = pop & ~remove
        red.append((base - np.quantile(abs_err[ret], 0.68)) / base * 100 if ret.sum() else np.nan)
        rm.append(remove.sum() / n * 100)
    red, rm = np.array(red), np.array(rm)
    knee = red / np.nanmax(red) - rm / rm.max(); k = int(np.nanargmax(knee))
    print(f"{name:40s} n {n:7,} p68 {base:.4f} | knee {thresholds[k]:g}{unit} (red {red[k]:.1f}%, rm {rm[k]:.1f}%) | "
          f"top-3 knee: " + ", ".join(f"{thresholds[j]:g}{unit}:{knee[j]:.3f}" for j in np.argsort(-knee)[:3]))
par = flow_par < OPT_A
bt = np.arange(20, 4.5, -0.5); it = np.arange(63, 4, -2)
sweep("(b) published: parallel & bldg >= 0", par & (far_bldg >= 0), far_bldg, bt, " m")
sweep("(b') variant: parallel & bldg > 0", par & (far_bldg > 0), far_bldg, bt, " m")
sweep("(c) published: parallel & isa > 0", par & (far_isa > 0), far_isa, it, "%")
sweep("(c') variant: parallel & isa >= 0", par & (far_isa >= 0), far_isa, it, "%")
