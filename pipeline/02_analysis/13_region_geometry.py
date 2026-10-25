#!/usr/bin/env python
"""
Decompose the regional transfer of the geometric mask (TGRS-2026-02661, R3-G2 follow-up).

Claim under test: the mask is a geometric control, so regional differences in its
performance should reduce to the regional prevalence of the geometric preconditions
(parallel alignment, far-bank ISA and building height), not to any regional physics.

Decomposition per HUC2 region:
  exposure  = share of parallel observations the mask flags (preconditions present)
  response  = how much worse flagged observations are than unflagged ones (p68 ratio)
  reduction = what removing the flagged share achieves (Table 3 number)
If the control is geometric, exposure should explain the regional spread while the
response stays comparatively uniform wherever enough flagged data exist.
"""
import os
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from scipy import stats as st

WORK = "/work/pi_kandread_umass_edu/swot-urban"
PROJ = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results"
OUT = f"{PROJ}/round2"
NODES = f"{PROJ}/swot_urban_nearfar_v6.5_results.parquet"
OBS = f"{WORK}/2_urban_reaches/all_final_shape.parquet"
ANGLE, HEIGHT, ISA = 13.0, 8.0, 19.0
HUCNAME = {"01": "New England", "02": "Mid-Atlantic", "03": "South Atlantic-Gulf",
           "04": "Great Lakes", "05": "Ohio", "06": "Tennessee",
           "07": "Upper Mississippi", "08": "Lower Mississippi", "10": "Missouri",
           "11": "Arkansas-White-Red", "12": "Texas-Gulf", "14": "Upper Colorado",
           "15": "Lower Colorado", "17": "Pacific Northwest", "18": "California",
           "19": "Alaska"}


def p68(x):
    return float(np.nanpercentile(x, 68)) if len(x) else np.nan


d = pq.read_table(NODES, columns=["reach_id", "flow_angle", "far_isa", "far_building",
                                  "error_node", "sample_geometry_category"]).to_pandas()
d["reach_id"] = d.reach_id.astype("int64")
d["e"] = d.error_node.abs()
fa = d.flow_angle % 360
d["angle_dev"] = np.minimum.reduce([fa, (fa - 180).abs(), 360 - fa])
d["isa_pct"] = (d.far_isa * 100 if d.far_isa.max() <= 1 else d.far_isa).fillna(0)
d["bld"] = d.far_building.fillna(0)
d = d.dropna(subset=["e"])
zone = d.sample_geometry_category.astype(str).str.lower()

link = pq.read_table(OBS, columns=["reach_id", "site_no"]).to_pandas()
link["reach_id"] = link.reach_id.astype("int64")
hu = pd.read_csv(f"{OUT}/site_huc.csv", dtype=str)
hu["huc2"] = hu.huc_cd.str[:2]
link = (link.drop_duplicates().merge(hu[["site_no", "huc2"]].drop_duplicates("site_no"),
                                     on="site_no", how="left")
        [["reach_id", "huc2"]].drop_duplicates("reach_id"))
d = d.merge(link, on="reach_id", how="left")
zone = d.sample_geometry_category.astype(str).str.lower()

rows = []
for h, g in d.groupby("huc2"):
    par = g[g.angle_dev <= ANGLE]
    if len(par) < 300 or pd.isna(h):
        continue
    z = zone.loc[g.index]
    flag = (par.bld > HEIGHT) | (par.isa_pct > ISA)
    f, u = par[flag], par[~flag]
    resp = p68(f.e) / p68(u.e) if len(f) >= 100 and len(u) >= 100 else np.nan
    base = p68(par.e)
    red = (base - p68(u.e)) / base * 100
    rows.append({
        "huc2": h, "region": HUCNAME.get(h, h), "n_obs": len(g),
        "parallel_pct": 100 * (g.angle_dev <= ANGLE).mean(),
        "zone_far_pct": 100 * (z == "far").mean(),
        "zone_near_pct": 100 * (z == "near").mean(),
        "zone_both_pct": 100 * (z == "near/far").mean(),
        "med_far_isa_par": float(par.isa_pct.median()),
        "far_isa_gt19_pct": 100 * (par.isa_pct > ISA).mean(),
        "med_far_bld_par": float(par.bld.median()),
        "far_bld_gt8_pct": 100 * (par.bld > HEIGHT).mean(),
        "exposure_flagged_pct": 100 * flag.mean(),
        "n_flagged": int(flag.sum()),
        "response_p68_ratio": resp,
        "reduction_pct": red,
    })
out = pd.DataFrame(rows).sort_values("n_obs", ascending=False)
os.makedirs(OUT, exist_ok=True)
out.to_csv(f"{OUT}/region_geometry_distribution.csv", index=False)
print(out.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

ok = out.dropna(subset=["reduction_pct"])
r1, p1 = st.spearmanr(ok.exposure_flagged_pct, ok.reduction_pct)
r2, p2 = st.spearmanr(ok.parallel_pct, ok.reduction_pct)
r3, p3 = st.spearmanr(ok.med_far_isa_par, ok.reduction_pct)
print(f"\nexposure (flagged share) vs reduction: rho {r1:+.2f} (p={p1:.4f}, n={len(ok)})")
print(f"parallel share vs reduction:           rho {r2:+.2f} (p={p2:.3f})")
print(f"median far ISA vs reduction:           rho {r3:+.2f} (p={p3:.3f})")
rr = out.dropna(subset=["response_p68_ratio"])
print(f"\nresponse ratio where computable (n_flag>=100): {len(rr)} regions, "
      f"p68(flagged)/p68(unflagged) = "
      f"{rr.response_p68_ratio.min():.2f}-{rr.response_p68_ratio.max():.2f} "
      f"(median {rr.response_p68_ratio.median():.2f})")
print(f"all >1: {(rr.response_p68_ratio > 1).all()}")
cv_expo = out.exposure_flagged_pct.std() / out.exposure_flagged_pct.mean()
cv_resp = rr.response_p68_ratio.std() / rr.response_p68_ratio.mean()
print(f"dispersion: exposure CV {cv_expo:.2f} vs response CV {cv_resp:.2f}")
print(f"\nsaved {OUT}/region_geometry_distribution.csv")
