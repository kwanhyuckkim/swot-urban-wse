"""2026-10-04: more digits for the Conclusions condition grid (exp_64522941.out A3 printed 3 decimals).
Question: is the perpendicular-crossing p68 0.485 m below or above 0.4850 (conclusion writes 0.48 m)?
Definitions copied from /work/kwanhyuckkim_umass_edu/swot_exp_20260916/exp2.py (A3; run_exp2.sbatch -> exp_64522941.out)."""
import numpy as np, pandas as pd, pyarrow.parquet as pq
NODES = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results/swot_urban_nearfar_v6.5_results.parquet"
d = pq.read_table(NODES, columns=["reach_id", "flow_angle", "far_isa", "error_node", "sample_geometry_category"]).to_pandas()
d["e"] = d.error_node.abs()
fa = d.flow_angle % 360
d["angle_dev"] = np.minimum.reduce([fa, (fa - 180).abs(), 360 - fa])
d["far_isa_pct"] = d.far_isa * 100 if d.far_isa.max() <= 1 else d.far_isa
d = d.dropna(subset=["e"])
zone = d.sample_geometry_category.astype(str).replace({"NA": "Neither", "Near/Far": "Both"})
print("zones:", zone.value_counts().to_dict())
p68 = lambda x: float(np.nanpercentile(x, 68))
par, perp, hi = d.angle_dev <= 13, d.angle_dev >= 80, d.far_isa_pct > 19
fb = zone.isin(["Far", "Both"])
rows = {"WORST parallel & Far|Both & far_isa>19": par & fb & hi,
        "flip ANGLE perpendicular & Far|Both & far_isa>19": perp & fb & hi,
        "flip SIDE parallel & Near": par & (zone == "Near"),
        "flip SIDE parallel & Near & far_isa==0": par & (zone == "Near") & (d.far_isa_pct <= 0),
        "flip URBAN parallel & Far|Both & far_isa<=19": par & fb & ~hi,
        "BASELINE Neither parallel": par & (zone == "Neither"),
        "BASELINE Neither perpendicular": perp & (zone == "Neither")}
for k, m in rows.items():
    print(f"{k:<52s} n={int(m.sum()):>7,}  p68={p68(d.e[m]):.6f}")
