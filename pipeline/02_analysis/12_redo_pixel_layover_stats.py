#!/usr/bin/env python
"""
Second verification pass for the pixel and layovr_val numbers, with explicit
sentinel masking and the exact per-pass formulas of 06_reviewer_product_checks.py.

Resolves the four items the first pass left open: the bright-land fire-rate
endpoints, the origin of the 0.305 m / 164-of-166 sentence, the bright-exclusion
error shift, and the displacement statistics under candidate definitions.
"""
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from scipy import stats as st

PROJ = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results"
RC = f"{PROJ}/reviewer_checks"
WORK = "/work/pi_kandread_umass_edu/swot-urban"
NODES = f"{PROJ}/swot_urban_nearfar_v6.5_results.parquet"
OBS = f"{WORK}/2_urban_reaches/all_final_shape.parquet"
T = 0.1832
FILL = -1e11


def mask(s):
    return s.where(s > FILL)


def sec(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# ---------------------------------------------------------------- pixel passes
sec("1. bright-land fire rate endpoints (fractional ISA bins)")
c = pd.read_parquet(f"{RC}/cross_level_pixc_vs_riversp.parquet")
print("   columns:", [x for x in c.columns if "bright" in x or x.startswith("h_")
                      or "delta" in x or "bias" in x])
isa = c.isa * 100 if c.isa.max() <= 1.5 else c.isa
for lo, hi in [(0, 2), (2, 10), (10, 30), (30, 45), (45, 100)]:
    g = c[(isa >= lo) & (isa < hi)]
    if len(g):
        print(f"   ISA {lo:>2}-{hi:<3}%  n {len(g):3d}  bright fire "
              f"{g.bright_land_fire_pct.mean():5.1f}%  (median {g.bright_land_fire_pct.median():5.1f}%)")
r, p = st.spearmanr(c.bright_land_fire_pct, isa, nan_policy="omit")
print(f"   Spearman(fire, ISA) {r:.3f}")

sec("2. per-pass height shifts: what does 0.305 / 164-of-166 belong to?")
for col in ["layover_pixel_bias_m", "delta_layover_m", "delta_bright_m"]:
    if col in c.columns:
        v = c[col].dropna()
        print(f"   {col:22s} n {len(v):3d}  median {v.median():+.3f} m  "
              f"pos {(v > 0).sum():3d}  neg {(v < 0).sum():3d}")
if "pct_bright_flagged" in c.columns:
    print(f"   passes where bright flag fires at all: "
          f"{(c.pct_bright_flagged > 0).sum()} of {len(c)}")
if "h_no_bright" in c.columns:
    db = (c.h_no_bright - c.h_all).dropna()
    print(f"   h_no_bright - h_all       median {db.median():+.4f} m  neg {(db < 0).sum()}/{len(db)}"
          f"   (negative means marked pixels sit high)")

sec("3. flag-exclusion error shifts (formula err_excl = err + (h_excl - h_all))")
for name, hcol in [("layover", "h_no_layover"), ("bright", "h_no_bright")]:
    if hcol not in c.columns:
        print(f"   {name}: {hcol} not stored")
        continue
    shift = c[hcol] - c.h_all
    err_excl = (c.error_m + shift).abs()
    ok = err_excl.notna() & c.abs_error_m.notna()
    print(f"   {name:8s} mean |err| {c.abs_error_m[ok].mean():.3f} -> {err_excl[ok].mean():.3f} m, "
          f"improves {(err_excl[ok] < c.abs_error_m[ok]).sum()}/{int(ok.sum())}")

# ---------------------------------------------------------------- layovr_val
sec("4. layovr_val tiers with sentinel masking")
n = pd.read_parquet(f"{RC}/layovr_val_node.parquet",
                    columns=["reach_id", "time_str", "wse_u", "wse_r_u", "layovr_val"])
for col in ["layovr_val", "wse_u", "wse_r_u"]:
    n[col] = mask(n[col])
lv = n.layovr_val.dropna()
sysu = np.sqrt(np.maximum(n.wse_u**2 - n.wse_r_u**2, 0))
ok = n.layovr_val.notna() & sysu.notna()
print(f"   node: valid {len(lv):,} of {len(n):,}  median {lv.median():.3f} m  "
      f"p75 {lv.quantile(.75):.3f}")
print(f"   node systematic unc median {sysu.dropna().median():.4f} m  "
      f"Spearman {st.spearmanr(n.layovr_val[ok], sysu[ok]).statistic:+.3f} (n={ok.sum():,})")

r = pd.read_parquet(f"{RC}/layovr_val_reach.parquet")
for col in ["layovr_val", "wse_u", "wse_r_u"]:
    r[col] = mask(r[col])
rl = r.layovr_val.dropna()
rsys = np.sqrt(np.maximum(r.wse_u**2 - r.wse_r_u**2, 0))
rok = r.layovr_val.notna() & rsys.notna()
ratio = (r.layovr_val / rsys.replace(0, np.nan))[rok].dropna()
print(f"   reach: valid {len(rl):,} of {len(r):,}  median {rl.median():.3f} m")
print(f"   reach Spearman vs sys unc {st.spearmanr(r.layovr_val[rok], rsys[rok]).statistic:+.3f}  "
      f"per-obs ratio median {ratio.median():.1f}")
nmax = (n.dropna(subset=["layovr_val"])
        .groupby(["reach_id", "time_str"]).layovr_val.max().rename("node_max").reset_index())
j = r.dropna(subset=["layovr_val"]).merge(nmax, on=["reach_id", "time_str"])
print(f"   reach vs max node: Spearman "
      f"{st.spearmanr(j.layovr_val, j.node_max).statistic:.2f}, median ratio "
      f"{(j.layovr_val / j.node_max.replace(0, np.nan)).median():.1f}, pairs {len(j):,}")

ob = pq.read_table(OBS, columns=["reach_id", "wse_relative_m", "wse_usgs_relative_m",
                                 "nlcd_21_mean", "nlcd_22_mean", "nlcd_23_mean",
                                 "nlcd_24_mean", "fctimp_mean"]).to_pandas()
ob["reach_id"] = ob.reach_id.astype("int64")
ob["e"] = (ob.wse_relative_m - ob.wse_usgs_relative_m).abs()
er = ob.groupby("reach_id").agg(err=("e", "median"), fct=("fctimp_mean", "first")).reset_index()
lvr = r.dropna(subset=["layovr_val"]).groupby("reach_id").layovr_val.median().rename("lv").reset_index()
lvr["reach_id"] = lvr.reach_id.astype("int64")
jj = er.merge(lvr, on="reach_id")
rho_all = st.spearmanr(jj.err, jj.lv).statistic
hi = jj[jj.fct >= 30]
print(f"   reach-median layovr vs gauge error: rho {rho_all:+.3f} (n={len(jj)}); "
      f"fctimp>=30: rho {st.spearmanr(hi.err, hi.lv).statistic if len(hi)>5 else np.nan:+.3f} (n={len(hi)})")

# ---------------------------------------------------------------- displacement
sec("5. displacement definitions (target: median 219, IQR 169-355, 71%/92%)")
d = pq.read_table(NODES, columns=["flow_angle", "far_building", "far_isa", "far_ratio",
                                  "far_ratio_building", "dw_far_building", "error_node",
                                  "width_node", "xtrk_dist_node", "sat_altitude",
                                  "geometry_category"]).to_pandas()
d["e"] = d.error_node.abs()
fa = d.flow_angle % 360
d["angle_dev"] = np.minimum.reduce([fa, (fa - 180).abs(), 360 - fa])
d["tan_th"] = d.xtrk_dist_node.abs() / d.sat_altitude

defs = {
    "A angle<=13, far_building>0, e>=T": (d.angle_dev <= 13) & (d.far_building > 0),
    "B + far ISA present": (d.angle_dev <= 13) & (d.far_building > 0) & (d.far_isa.fillna(0) > 0),
    "C + coverage>=0.5 (building)": (d.angle_dev <= 13) & (d.far_building > 0)
        & (d.far_ratio_building.fillna(0) >= 0.5),
    "D angle<=13, dw_far_building>0": (d.angle_dev <= 13) & (d.dw_far_building.fillna(0) > 0),
    "E cat far/both, building>0": (d.angle_dev <= 13) & (d.far_building > 0)
        & d.geometry_category.isin(["far", "both", "Far", "Both"]),
}
for name, m in defs.items():
    g = d[m & d.e.notna() & d.tan_th.notna()].copy()
    if not len(g):
        print(f"   {name:34s} n=0")
        continue
    g["disp"] = (g.dw_far_building if name.startswith("D") else g.far_building) / g.tan_th
    hi = g[g.e >= T].dropna(subset=["width_node"])
    if not len(hi):
        continue
    q = hi.disp.quantile([.25, .5, .75])
    print(f"   {name:34s} n_hi {len(hi):6,}  median {q[.5]:4.0f}  IQR {q[.25]:.0f}-{q[.75]:.0f}  "
          f"full {100*(hi.disp >= hi.width_node).mean():.0f}%  half {100*(hi.disp >= .5*hi.width_node).mean():.0f}%")
print()
