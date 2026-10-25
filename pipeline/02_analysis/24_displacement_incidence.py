#!/usr/bin/env python
"""2026-09-28: layover displacement with the incidence angle instead of the look angle.

The published numbers used tan(theta) = |xtrk_dist_node| / sat_altitude (flat-Earth look
angle; 11_verify_response_numbers.py:135, 12_redo_pixel_layover_stats.py:124,
figure_src/36_figS5_displacement_print.py:31). Ground displacement needs the incidence
angle on a sphere, theta_i = look + central angle (tan_inc below, as in
width_meeting_0924/width_v2.py). Every block prints LOOK (should reproduce the paper) next
to INC (the correction), for:
  A. Section V.B / letter R3-G4: definition C of 12_redo (angle <= 13, far_building > 0,
     far_ratio_building >= 0.5), high-error subset: median tan, median/IQR displacement,
     shares >= full and half width (paper: 0.039, 222 m, 168-356 m, 73 %, 94 %).
  B. Section V.E: share with displacement >= width by Table IV width bin
     (paper: 99.9 % below 100 m, 0.4 % at 800-1200 m).
  C. Supplementary Fig. S5 population (parallel <= 10 deg, far_building >= 5 m, n = 17,669).
  D. Section V.A / letter R4-15: candidate populations for "incidence median 2.1 deg,
     90th percentile 3.5 deg; median 8.8 m near-bank building; shadow 0.34 m (0.60 m)".
"""
import json
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJ = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results"
NODES = f"{PROJ}/swot_urban_nearfar_v6.5_results.parquet"
OUT = sys.argv[1] if len(sys.argv) > 1 else "displacement_incidence.json"
T = 0.1832
R = 6378.137e3


def tan_look(x, h):
    return np.abs(x) / h


def tan_inc(x, h):
    g = np.abs(x) / R
    rho = np.sqrt((R + h) ** 2 + R ** 2 - 2 * R * (R + h) * np.cos(g))
    return np.tan(np.arcsin(R * np.sin(g) / rho) + g)


ANGLES = {"look": tan_look, "inc": tan_inc}
res = {}

cols = ["reach_id", "flow_angle", "far_building", "far_ratio_building", "far_isa",
        "near_building", "near_ratio_building", "near_isa", "error_node", "width_node",
        "xtrk_dist_node", "sat_altitude", "geometry_category"]
d = pq.read_table(NODES, columns=cols).to_pandas()
print(f"rows {len(d):,}; geometry_category {d.geometry_category.value_counts(dropna=False).to_dict()}")
d["e"] = d.error_node.abs()
fa = d.flow_angle % 360
d["angle_dev"] = np.minimum.reduce([fa, (fa - 180).abs(), 360 - fa])
d = d[d.e.notna() & d.xtrk_dist_node.notna() & d.sat_altitude.notna()].copy()
for k, f in ANGLES.items():
    d[f"tan_{k}"] = f(d.xtrk_dist_node.values, d.sat_altitude.values)
print(f"usable rows {len(d):,}; tan_inc/tan_look median {np.nanmedian(d.tan_inc / d.tan_look):.4f}")


def q(s, p):
    return float(np.nanpercentile(s, p)) if len(s) else float("nan")


# ------------------------------------------------ A. V.B displacement (definition C)
print("\nA. V.B displacement, definition C, high-error subset")
C = d[(d.angle_dev <= 13) & (d.far_building > 0) & (d.far_ratio_building.fillna(0) >= 0.5)].copy()
res["A"] = {"n_C": int(len(C))}
for k in ANGLES:
    C[f"disp_{k}"] = C.far_building / C[f"tan_{k}"]
    hi = C[C.e >= T]
    hw = hi.dropna(subset=["width_node"])
    r = {"median_tan_C": float(C[f"tan_{k}"].median()), "median_tan_hi": float(hi[f"tan_{k}"].median()),
         "n_hi": int(len(hi)), "n_hi_width": int(len(hw)),
         "disp_hi_p25_p50_p75": [q(hi[f"disp_{k}"], p) for p in (25, 50, 75)],
         "disp_hiw_p25_p50_p75": [q(hw[f"disp_{k}"], p) for p in (25, 50, 75)],
         "share_full": float((hw[f"disp_{k}"] >= hw.width_node).mean()),
         "share_half": float((hw[f"disp_{k}"] >= 0.5 * hw.width_node).mean())}
    res["A"][k] = r
    print(f"  {k:4s} tan med C {r['median_tan_C']:.4f} hi {r['median_tan_hi']:.4f} | n_hi {r['n_hi']:,} "
          f"| disp hi {r['disp_hi_p25_p50_p75'][1]:.0f} m (IQR {r['disp_hi_p25_p50_p75'][0]:.0f}-"
          f"{r['disp_hi_p25_p50_p75'][2]:.0f}) | hi&width {r['disp_hiw_p25_p50_p75'][1]:.0f} m "
          f"(IQR {r['disp_hiw_p25_p50_p75'][0]:.0f}-{r['disp_hiw_p25_p50_p75'][2]:.0f}) "
          f"| >= w {100*r['share_full']:.1f}% | >= w/2 {100*r['share_half']:.1f}%")

# ------------------------------------------------ B. V.E width bins (Table IV bins, width_node)
print("\nB. V.E share with displacement >= width, by width bin (definition C)")
bins = [0, 100, 200, 300, 400, 600, 800, 1200, 1e9]
lab = ["<100", "100-200", "200-300", "300-400", "400-600", "600-800", "800-1200", ">1200"]
Cw = C.dropna(subset=["width_node"]).copy()
Cw["wbin"] = pd.cut(Cw.width_node, bins, labels=lab)
res["B"] = {}
for scope, S in (("all", Cw), ("high_error", Cw[Cw.e >= T])):
    res["B"][scope] = {}
    for b, g in S.groupby("wbin", observed=True):
        row = {"n": int(len(g))}
        for k in ANGLES:
            row[f"share_full_{k}"] = float((g[f"disp_{k}"] >= g.width_node).mean())
        res["B"][scope][str(b)] = row
        print(f"  {scope:10s} {str(b):9s} n {len(g):6,}  look {100*row['share_full_look']:5.1f}%  "
              f"inc {100*row['share_full_inc']:5.1f}%")

# ------------------------------------------------ C. Fig. S5 population
print("\nC. Supplementary Fig. S5 population")
fa2 = d.flow_angle % 180
d["ang"] = np.minimum(fa2, 180 - fa2)
S5 = d[(d.ang <= 10) & (d.far_building >= 5)].copy()
res["C"] = {"n": int(len(S5))}
edges = [5, 6, 7, 8, 9, 10, 11, 12, 13, 1e9]
labs = ["5-6", "6-7", "7-8", "8-9", "9-10", "10-11", "11-12", "12-13", "13+"]
for k in ANGLES:
    S5[f"dr_{k}"] = S5.far_building / S5[f"tan_{k}"]
    S5[f"ratio_{k}"] = S5[f"dr_{k}"] / S5.width_node
    r = {"median_dr": float(S5[f"dr_{k}"].median()),
         "share_ratio_ge1": float((S5[f"ratio_{k}"] >= 1).mean()),
         "share_ratio_ge_half": float((S5[f"ratio_{k}"] >= 0.5).mean()),
         "median_ratio_by_bin": {}}
    for lo, hi_, lb in zip(edges[:-1], edges[1:], labs):
        g = S5[(S5.far_building >= lo) & (S5.far_building < hi_)]
        r["median_ratio_by_bin"][lb] = float(g[f"ratio_{k}"].median())
    res["C"][k] = r
    print(f"  {k:4s} n {len(S5):,} | median dr {r['median_dr']:.0f} m | ratio>=1 {100*r['share_ratio_ge1']:.1f}% "
          f"| ratio>=0.5 {100*r['share_ratio_ge_half']:.1f}% | median ratio 5-6 "
          f"{r['median_ratio_by_bin']['5-6']:.2f}, 13+ {r['median_ratio_by_bin']['13+']:.2f}")

# ------------------------------------------------ D. V.A incidence and shadow candidates
print("\nD. V.A candidates (paper: 2.1 deg / 3.5 deg; building 8.8 m; shadow 0.34 m / 0.60 m)")
cat = d.geometry_category.astype(str)
par = d.angle_dev <= 13
nb = d.near_building.fillna(0) > 0
nq = nb & (d.near_ratio_building.fillna(0) >= 0.5)
cands = {
    "all": pd.Series(True, index=d.index),
    "parallel13": par,
    "near_bld>0": nb,
    "parallel13 & near_bld>0": par & nb,
    "near_bld>0 & cov>=0.5": nq,
    "parallel13 & near_bld>0 & cov>=0.5": par & nq,
    "parallel13 & cat Near": par & cat.isin(["Near", "near"]),
    "parallel13 & cat Near|Both": par & cat.isin(["Near", "near", "Near/Far", "Both", "both"]),
    "cat Near|Both": cat.isin(["Near", "near", "Near/Far", "Both", "both"]),
}
res["D"] = {}
for name, m in cands.items():
    g = d[m]
    if not len(g):
        continue
    row = {"n": int(len(g)), "near_bld_median": float(g.near_building.median())}
    for k in ANGLES:
        ang = np.degrees(np.arctan(g[f"tan_{k}"]))
        sh = g.near_building * g[f"tan_{k}"]
        row[k] = {"deg_p50": q(ang, 50), "deg_p90": q(ang, 90),
                  "shadow_p50": q(sh, 50), "shadow_p90": q(sh, 90)}
    res["D"][name] = row
    lk, ic = row["look"], row["inc"]
    print(f"  {name:36s} n {len(g):7,} bld {row['near_bld_median']:5.2f} | look {lk['deg_p50']:.2f}/"
          f"{lk['deg_p90']:.2f} deg sh {lk['shadow_p50']:.2f}/{lk['shadow_p90']:.2f} m | inc "
          f"{ic['deg_p50']:.2f}/{ic['deg_p90']:.2f} deg sh {ic['shadow_p50']:.2f}/{ic['shadow_p90']:.2f} m")

with open(OUT, "w") as fh:
    json.dump(res, fh, indent=1)
print(f"\nsaved {OUT}")
