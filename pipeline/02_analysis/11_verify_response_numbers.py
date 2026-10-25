#!/usr/bin/env python
"""
Independent re-verification of every checkable number quoted in the TGRS-2026-02661
response letter and the revised manuscript, recomputed from the primary analysis
files rather than from the earlier session's printouts.

Each block prints CLAIMED (what the letter or manuscript says) next to COMPUTED, so
a mismatch is visible at a glance.
"""
import glob
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from scipy import stats as st

PROJ = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results"
RC = f"{PROJ}/reviewer_checks"
WORK = "/work/pi_kandread_umass_edu/swot-urban"
NODES = f"{PROJ}/swot_urban_nearfar_v6.5_results.parquet"
OBS = f"{WORK}/2_urban_reaches/all_final_shape.parquet"
T = 0.1832


def sec(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def line(claim, computed):
    print(f"  CLAIMED  {claim}\n  COMPUTED {computed}")


# ------------------------------------------------ A. network / obstructions
sec("A. R3-G1 obstruction network numbers")
d = pd.read_parquet(f"{RC}/obstruction_network_distance.parquet")
up = d[d.up_found.astype(bool)]
dn = d[d.dn_found.astype(bool)]
near = d[d.nearest_km.notna()]
line("upstream found for 331 reaches, median 39.6 km, 4 reaches crossed",
     f"up {len(up)}, median {up.up_km.median():.1f} km, hops {up.up_hops.median():.0f}")
line("downstream found for 250 reaches, median 38.4 km, 5 reaches crossed",
     f"dn {len(dn)}, median {dn.dn_km.median():.1f} km, hops {dn.dn_hops.median():.0f}")
line("358 of 409 within 200 km, median 26.9 km; 102 within 1 km",
     f"near {len(near)}, median {near.nearest_km.median():.1f} km; "
     f"<=1 km {(near.nearest_km <= 1).sum()}")
r, p = st.spearmanr(near.nearest_km, near.median_abs_error_m)
line("error vs distance Spearman +0.019 (p = 0.73)", f"{r:+.3f} (p = {p:.2f}, n = {len(near)})")
a = near[near.nearest_km <= 10].median_abs_error_m
b = near[near.nearest_km > 10].median_abs_error_m
mw = st.mannwhitneyu(a, b)
line("<=10 km median 0.208 vs 0.200 beyond, Mann-Whitney p = 0.84",
     f"{a.median():.3f} vs {b.median():.3f}, p = {mw.pvalue:.2f}")
r2, p2 = st.spearmanr(near.nearest_km, near.isa)
line("proximity vs ISA Spearman -0.30 (p < 0.001)", f"{r2:+.2f} (p = {p2:.1e})")

# ------------------------------------------------ B. pixel-level 166 passes
sec("B. R4 pixel-level numbers over 166 passes")
c = pd.read_parquet(f"{RC}/cross_level_pixc_vs_riversp.parquet")
lo = c[c.isa <= 2].bright_land_fire_pct.mean()
hi = c[c.isa >= 45].bright_land_fire_pct.mean()
r, p = st.spearmanr(c.bright_land_fire_pct, c.isa)
line("bright-land fires 1.7% at 1% ISA and 59% at 50% ISA, Spearman 0.91",
     f"{lo:.1f}% at <=2% ISA, {hi:.1f}% at >=45% ISA, Spearman {r:.2f} (n = {len(c)})")
s = pd.read_parquet(f"{RC}/pixc_layover_sweep.parquet")
bias = s.layover_pixel_bias_m.dropna()
line("flagged pixels sit 0.305 m higher in 164 of 166 passes",
     f"median bias {bias.median():.3f} m, positive in {(bias > 0).sum()} of {len(bias)}")
err_all = c.abs_error_m.abs()
shift = c.h_no_layover - c.h_all
err_excl = (c.error_m + shift).abs()
ok = err_excl.notna() & err_all.notna()
line("excluding layover-flagged pixels: mean |error| 0.398 -> 0.665 m, improves 51/166",
     f"mean {err_all[ok].mean():.3f} -> {err_excl[ok].mean():.3f} m, "
     f"improves {(err_excl[ok] < err_all[ok]).sum()}/{ok.sum()}")
pc = pd.read_parquet(f"{RC}/pixc_partial_correlation.parquet")
top = pc.loc[pc.partial_rho.abs().idxmax()]
line("after ISA control no pixel indicator significant; strongest 0.147 (p = 0.059)",
     f"strongest {top.partial_rho:+.3f} (p = {top.partial_p:.3f}, {top.indicator}); "
     f"min p = {pc.partial_p.min():.3f}")

# ------------------------------------------------ C. layovr_val three tiers
sec("C. layovr_val node and reach tiers")
n = pd.read_parquet(f"{RC}/layovr_val_node.parquet",
                    columns=["reach_id", "node_id", "time_str", "wse_u", "wse_r_u",
                             "layovr_val"])
lv = n.layovr_val.dropna()
sysu = np.sqrt(np.maximum(n.wse_u**2 - n.wse_r_u**2, 0)).dropna()
line("node layovr_val median 0.080 m over 2,739,848 obs",
     f"median {lv.median():.3f} m over {len(n):,} obs, {n.node_id.nunique():,} nodes")
line("systematic uncertainty median 0.0900 m; Spearman with layovr_val +0.000",
     f"median {sysu.median():.4f} m; Spearman "
     f"{st.spearmanr(n.layovr_val, np.sqrt(np.maximum(n.wse_u**2-n.wse_r_u**2,0)), nan_policy='omit').statistic:+.3f}")

rf = sorted(glob.glob(f"{RC}/layovr_val_reach*.parquet"))
if not rf:
    print("  reach-level layovr_val parquet MISSING under reviewer_checks -- re-pull needed")
else:
    rr = pd.read_parquet(rf[0])
    rl = rr.layovr_val.dropna()
    rsys = np.sqrt(np.maximum(rr.wse_u**2 - rr.wse_r_u**2, 0))
    ratio = (rr.layovr_val / rsys.replace(0, np.nan)).dropna()
    line("reach layovr_val median 4.57 m over 52,240 obs; ~52x the systematic uncertainty",
         f"median {rl.median():.3f} m over {len(rr):,} obs, {rr.reach_id.nunique()} reaches; "
         f"ratio of medians {rl.median()/rsys.dropna().median():.1f}, "
         f"median per-obs ratio {ratio.median():.1f}")
    line("uncorrelated with systematic uncertainty at reach scale (0.003)",
         f"Spearman {st.spearmanr(rr.layovr_val, rsys, nan_policy='omit').statistic:+.3f}")
    agg = n.groupby(["reach_id", "time_str"]).layovr_val.max().rename("node_max").reset_index()
    j = rr.merge(agg, on=["reach_id", "time_str"])
    j = j.dropna(subset=["layovr_val", "node_max"])
    line("reach value tracks largest node value: Spearman 0.88, median ratio 1.8, 44,175 pairs",
         f"Spearman {st.spearmanr(j.layovr_val, j.node_max).statistic:.2f}, "
         f"median ratio {(j.layovr_val/j.node_max.replace(0,np.nan)).median():.1f}, "
         f"{len(j):,} pairs")
    ob = pq.read_table(OBS, columns=["reach_id", "wse_relative_m",
                                     "wse_usgs_relative_m"]).to_pandas()
    ob["reach_id"] = ob.reach_id.astype("int64")
    ob["e"] = (ob.wse_relative_m - ob.wse_usgs_relative_m).abs()
    er = ob.groupby("reach_id").e.median().rename("err").reset_index()
    lvr = rr.groupby("reach_id").layovr_val.median().rename("lv").reset_index()
    lvr["reach_id"] = lvr.reach_id.astype("int64")
    jj = er.merge(lvr, on="reach_id")
    line("against gauges: reach correlation 0.06",
         f"reach-median Spearman {st.spearmanr(jj.err, jj.lv, nan_policy='omit').statistic:+.2f} "
         f"(n = {len(jj)})")

# ------------------------------------------------ D. displacement h/tan(theta)
sec("D. R3-G4 layover displacement")
d = pq.read_table(NODES, columns=["flow_angle", "far_building", "far_isa", "error_node",
                                  "width_node", "xtrk_dist_node", "sat_altitude",
                                  "geometry_category"]).to_pandas()
d["e"] = d.error_node.abs()
fa = d.flow_angle % 360
d["angle_dev"] = np.minimum.reduce([fa, (fa - 180).abs(), 360 - fa])
par = d[(d.angle_dev <= 13) & (d.far_building > 0) & d.e.notna()
        & d.xtrk_dist_node.notna() & d.sat_altitude.notna()].copy()
par["tan_th"] = par.xtrk_dist_node.abs() / par.sat_altitude
line("median tan(theta) = 0.037", f"median {par.tan_th.median():.3f} (n = {len(par):,})")
par["disp"] = par.far_building / par.tan_th
hi = par[par.e >= T]
q = hi.disp.quantile([.25, .5, .75])
line("high-error displacement median 219 m, IQR 169-355 m",
     f"median {q[.5]:.0f} m, IQR {q[.25]:.0f}-{q[.75]:.0f} m (n = {len(hi):,})")
w = hi.dropna(subset=["width_node"])
line(">= full width for 71%, >= half width for 92%",
     f"full {100*(w.disp >= w.width_node).mean():.0f}%, "
     f"half {100*(w.disp >= 0.5*w.width_node).mean():.0f}%")
loe = par[par.e < T].dropna(subset=["width_node"])
print(f"  (low-error group: median {loe.disp.median():.0f} m, "
      f">= full width {100*(loe.disp >= loe.width_node).mean():.0f}%)")

# ------------------------------------------------ E. published dataset anchors
sec("E. dataset anchors")
ob = pq.read_table(OBS, columns=["reach_id", "wse_relative_m",
                                 "wse_usgs_relative_m", "reach_q_b"]).to_pandas()
ob["e"] = (ob.wse_relative_m - ob.wse_usgs_relative_m).abs()
line("12,145 observations, 409 reaches, base rate 51.1%, bright_land bit always zero",
     f"{len(ob):,} obs, {ob.reach_id.nunique()} reaches, "
     f"base {(ob.e >= T).mean()*100:.1f}%, "
     f"bright bit set {(ob.reach_q_b.astype('int64') & 128).astype(bool).sum()}")
print()
