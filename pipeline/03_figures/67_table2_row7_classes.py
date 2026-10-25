#!/usr/bin/env python
"""2026-10-05: class composition of Table II row 7 ("parallel and far side qualifies (coverage >= 0.5)"), defined in
swot-urban/pipeline/02_analysis/07_round2_analyses.py lines 75-76 as angle_dev <= 13 & far_isa > 0 & far_ratio >= 0.5.
Checks whether the row holds both the Far and the Both classes (user question)."""
import numpy as np, pyarrow.parquet as pq
PROJ = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results"
d = pq.read_table(f"{PROJ}/swot_urban_nearfar_v6.5_results.parquet",
                  columns=["flow_angle", "far_isa", "far_ratio", "near_isa", "near_ratio", "error_node",
                           "sample_geometry_category"]).to_pandas().dropna(subset=["error_node"])
fa = d.flow_angle % 360
d["ang"] = np.minimum.reduce([fa, (fa - 180).abs(), 360 - fa])
par = d[d.ang <= 13]
row7 = par[(par.far_isa.fillna(0) > 0) & (par.far_ratio.fillna(0) >= 0.5)]
print(f"parallel (<=13) n {len(par):,}; row 7 n {len(row7):,}")
print("row 7 by class:", row7.sample_geometry_category.astype(str).value_counts().to_dict())
cls = par.sample_geometry_category.astype(str).str.lower()
print("parallel by class:", cls.value_counts().to_dict())
print(f"parallel Far+Both n {int(cls.isin(['far', 'near/far']).sum()):,}")
