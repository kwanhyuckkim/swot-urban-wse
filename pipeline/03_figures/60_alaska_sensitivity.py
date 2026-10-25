"""2026-10-05: how much do the headline numbers move if the 3 Alaska reaches are dropped? (cloud review: choose between
(a) drop Alaska and recompute with 406 reaches, (b) keep 409 and add a sensitivity sentence). The Alaska reaches lie outside
the CONUS NLCD and the GEDI building-height coverage, so their ISA and height were filled with 0
(03_usgs_datum_to_egm2008.ipynb c18; 13_region_geometry.py 45-46; 26b fillna(0)). Read-only; prints ALL vs NO_AK.
Definitions copied from revision1/cv_slope_20261004/26b_cv_slope.py (stage-1 summed slope, knee) and
figure_src/41_condition_grid_digits.py (condition grid)."""
import numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy.stats import spearmanr

WORK = "/work/pi_kandread_umass_edu/swot-urban"
PROJ = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results"
AK = {78380100051, 81130400021, 81130400041}

d = pq.read_table(f"{PROJ}/swot_urban_nearfar_v6.5_results.parquet",
                  columns=["reach_id", "flow_angle", "far_building", "far_isa", "error_node", "near_isa", "near_ratio",
                           "far_ratio", "sample_geometry_category"]).to_pandas()
d = d.dropna(subset=["error_node"]); d["reach_id"] = d.reach_id.astype("int64")
fa = d.flow_angle % 180; d["ang"] = np.minimum(fa, 180 - fa); d["e"] = d.error_node.abs()
d["isa"] = (d.far_isa * 100 if d.far_isa.max() <= 1 else d.far_isa).fillna(0); d["bld"] = d.far_building.fillna(0)
has_far = (d.far_isa.fillna(0) > 0) & (d.far_ratio.fillna(0) >= 0.5)
has_near = (d.near_isa.fillna(0) > 0) & (d.near_ratio.fillna(0) >= 0.5)
d["All"] = True; d["Far"] = has_far & ~has_near; d["Near"] = has_near & ~has_far; d["Both"] = has_far & has_near; d["Neither"] = ~has_far & ~has_near
d["zone"] = d.sample_geometry_category.astype(str).replace({"NA": "Neither", "Near/Far": "Both"})
CAT = ["All", "Far", "Near", "Both", "Neither"]; GRID = np.arange(1, 32, 2)
ak = d.reach_id.isin(AK)
print(f"node rows {len(d):,}; Alaska rows {int(ak.sum()):,} from reaches {sorted(set(d.reach_id[ak]))}; Alaska parallel(<13) {int((ak & (d.ang < 13)).sum()):,}")
print("Alaska rows: isa>0", int((ak & (d.isa > 0)).sum()), "bld>0", int((ak & (d.bld > 0)).sum()), "zones", d.zone[ak].value_counts().to_dict())


def p68(x):
    return float(np.percentile(x, 68))


def knee(sub, col, grid):
    base = p68(sub.e); red, rmv, ts = [], [], []
    for t in grid:
        rm = sub[col] > t; ret = sub[~rm]
        red.append((base - p68(ret.e)) / base * 100 if len(ret) else np.nan); rmv.append(rm.mean() * 100); ts.append(t)
    red = np.array(red); rmv = np.array(rmv)
    return ts[int(np.nanargmax(red / np.nanmax(red) - rmv / np.nanmax(rmv)))]


def theta_slope(x):
    e = x.e.values; fp = x.ang.values; res = {}
    for n in CAT:
        m = x[n].values.astype(bool); base = np.quantile(e[m], 0.68)
        res[n] = [(base - np.quantile(e[m & ~(fp < at)], 0.68)) / base * 100 for at in GRID]
    s = np.array([sum((res[n][k + 1] - res[n][k]) / 2.0 for n in CAT) for k in range(len(GRID) - 1)])
    k = int(np.argmax(s)); return int(GRID[k + 1]), float(s[k])


def metrics(x):
    o = {"nodes": len(x), "reaches": x.reach_id.nunique(), "p68_all": p68(x.e)}
    a, smax = theta_slope(x); par0 = x[x.ang < a]
    o.update({"theta*": a, "max_slope_sum": smax, "h*": knee(par0, "bld", np.arange(20, 4.5, -0.5)),
              "i*": knee(par0, "isa", np.arange(63, 4, -2))})
    par = x[x.ang < 13]; base = p68(par.e); o.update({"par_n": len(par), "par_p68": base})
    for nm, fl in (("full", (par.bld > 8) | (par.isa > 19)), ("bld", par.bld > 8), ("isa", par.isa > 19)):
        o[f"{nm}_red%"] = (base - p68(par.e[~fl])) / base * 100; o[f"{nm}_flag%"] = fl.mean() * 100
    p13, perp, hi, fb = x.ang <= 13, x.ang >= 80, x.isa > 19, x.zone.isin(["Far", "Both"])
    for k, m in {"grid_worst": p13 & fb & hi, "grid_perp_FB_hi": perp & fb & hi, "grid_near_far0": p13 & (x.zone == "Near") & (x.isa <= 0),
                 "grid_FB_lo": p13 & fb & ~hi, "grid_Neither_par": p13 & (x.zone == "Neither"), "grid_Neither_perp": perp & (x.zone == "Neither")}.items():
        o[k] = p68(x.e[m])
    for pl, pm in (("le10", x.ang <= 10), ("le13", x.ang <= 13)):
        o[f"isa0-5_{pl}"] = p68(x.e[pm & (x.isa <= 5)]); o[f"isa80-100_{pl}"] = p68(x.e[pm & (x.isa > 80)])
    for z in ["Far", "Both", "Near", "Neither"]:
        o[f"ratio_{z}"] = p68(x.e[(x.ang <= 10) & (x.zone == z)]) / p68(x.e[(x.ang >= 80) & (x.zone == z)])
    return o


A, B = metrics(d), metrics(d[~ak])
print(f"\n{'metric':<22s}{'ALL':>14s}{'NO_AK':>14s}")
for k in A:
    print(f"{k:<22s}{A[k]:>14.4f}{B[k]:>14.4f}" if isinstance(A[k], float) else f"{k:<22s}{A[k]:>14}{B[k]:>14}")

# reach-observation level: Spearman(reach ISA, |reach error|) as in IV.A.1 (rho 0.004, p 0.663)
obs = pq.read_table(f"{WORK}/2_urban_reaches/all_final_shape.parquet",
                    columns=["reach_id", "site_no", "wse_relative_m", "wse_usgs_relative_m"]).to_pandas()
obs["e"] = (obs.wse_relative_m - obs.wse_usgs_relative_m).abs(); obs["reach_id"] = obs.reach_id.astype("int64")
cols = [c for c in pq.read_schema(f"{WORK}/2_urban_reaches/all_reaches_vD.parquet").names if c in ("reach_id", "fctimp_mean", "fctimp_count")]
ar = pq.read_table(f"{WORK}/2_urban_reaches/all_reaches_vD.parquet", columns=cols).to_pandas().drop_duplicates("reach_id")
ar["reach_id"] = ar.reach_id.astype("int64"); o2 = obs.merge(ar, on="reach_id", how="left")
print(f"\nreach observations {len(o2):,} (reaches {o2.reach_id.nunique()}), fctimp missing {int(o2.fctimp_mean.isna().sum())}")
for lab, x in (("ALL", o2.dropna(subset=["fctimp_mean", "e"])), ("NO_AK", o2[~o2.reach_id.isin(AK)].dropna(subset=["fctimp_mean", "e"]))):
    r = spearmanr(x.fctimp_mean, x.e); print(f"  {lab}: Spearman(reach ISA, |error|) {r.statistic:+.4f} p={r.pvalue:.3f} n={len(x):,}")
print("\nreach ISA (fctimp_mean) check:", ar[ar.reach_id.isin(AK | {77235000141})].to_string(index=False))
con = ar[~ar.reach_id.isin(AK)]
print(f"CONUS reaches {len(con)}: fctimp_mean min {con.fctimp_mean.min()!r} max {con.fctimp_mean.max()!r}; zero-valued {int((con.fctimp_mean <= 0).sum())}")
if "fctimp_count" in con: print("  zero-valued with count>0:", int(((con.fctimp_mean <= 0) & (con.fctimp_count > 0)).sum()))
