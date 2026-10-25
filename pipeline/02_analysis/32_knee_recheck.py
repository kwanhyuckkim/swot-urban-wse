#!/usr/bin/env python
"""2026-10-03: faithful re-implementation of figure_src/33_fig09_orig_print.py panels (b) and (c)
(same load, same flow_angle_parallel formula, strict < OPT_A, far_bldg >= 0 for (b), far_isa > 0 for (c),
same grids, np.quantile 0.68, knee = red/nanmax(red) - rm/rm.max(), np.argmax), then the same sweeps on the
other population for each criterion. Prints full sweeps and NaN diagnostics."""
import numpy as np, pandas as pd
PROJ = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results"
OPT_A = 13.0
cols = ["flow_angle", "far_building", "far_isa", "error_node", "near_isa", "near_ratio", "far_ratio"]
df = pd.read_parquet(f"{PROJ}/swot_urban_nearfar_v6.5_results.parquet", columns=cols, engine="pyarrow")
df["abs_error"] = df.error_node.abs()
df["flow_angle_parallel"] = df.flow_angle.apply(lambda a: min(a % 180, 180 - a % 180))
df["far_isa_pct"] = df.far_isa * 100 if df.far_isa.max() <= 1.0 else df.far_isa
abs_err = df.abs_error.values; flow_par = df.flow_angle_parallel.values
far_bldg = df.far_building.values; far_isa = df.far_isa_pct.values
print(f"rows {len(df):,} | NaN: error {np.isnan(abs_err).sum():,}, building {np.isnan(far_bldg).sum():,}, isa {np.isnan(far_isa).sum():,}")
print(f"parallel: flow_par < 13 → {(flow_par < OPT_A).sum():,} | <= 13 → {(flow_par <= OPT_A).sum():,} | exactly 13.0 → {(flow_par == OPT_A).sum():,}")
def sweep(label, pop, var, ths, unit, show=True):
    n_sub = int(pop.sum()); p68_sub = np.quantile(abs_err[pop], 0.68); res = []
    for t in ths:
        remove = pop & (var > t); ret = pop & ~remove
        p68s = np.quantile(abs_err[ret], 0.68) if ret.sum() else np.nan
        res.append((t, (p68_sub - p68s) / p68_sub * 100, remove.sum() / n_sub * 100, int(remove.sum())))
    red = np.array([r[1] for r in res]); rm = np.array([r[2] for r in res])
    knee = red / np.nanmax(red) - rm / rm.max(); oi = int(np.argmax(knee))
    order = np.argsort(-knee)[:4]
    print(f"\n{label}: n={n_sub:,} p68={p68_sub:.4f}  θ*={ths[oi]:g}{unit}  red={res[oi][1]:.2f}% rm={res[oi][2]:.2f}% (n_rm {res[oi][3]:,})")
    print("   top-4 knee scores: " + ", ".join(f"{ths[j]:g}{unit}={knee[j]:.4f}" for j in order))
    if show:
        print(f"   {'thr':>5} {'red%':>7} {'rm%':>7} {'n_rm':>8} {'knee':>8}")
        for j, (t, r, m, c) in enumerate(res):
            print(f"   {t:5g} {r:7.2f} {m:7.2f} {c:8,} {knee[j]:8.4f}{'  <==' if j == oi else ''}")
bt = np.arange(20, 4.5, -0.5); it = np.arange(63, 4, -2); par = flow_par < OPT_A
sweep("(b) PUBLISHED  height, parallel & far_bldg >= 0", par & (far_bldg >= 0), far_bldg, bt, " m")
sweep("(b') VARIANT   height, parallel & far_bldg > 0", par & (far_bldg > 0), far_bldg, bt, " m")
sweep("(c) PUBLISHED  ISA, parallel & far_isa > 0", par & (far_isa > 0), far_isa, it, "%")
sweep("(c') VARIANT   ISA, parallel & far_isa >= 0 (all parallel)", par & (far_isa >= 0), far_isa, it, "%")
