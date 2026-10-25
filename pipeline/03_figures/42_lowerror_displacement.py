"""2026-10-04: V.B last sentence check ("Displacements of similar size occur in the low-error group").
Same data, definition C and incidence as 24_displacement_incidence.py (lines 28-75); adds the low-error
subset (|e| < T) and, for the S5 caption decision, the S5 population at 13 deg instead of 10 deg."""
import numpy as np, pandas as pd, pyarrow.parquet as pq
NODES = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results/swot_urban_nearfar_v6.5_results.parquet"
T, R = 0.1832, 6378.137e3


def tan_inc(x, h):
    g = np.abs(x) / R
    rho = np.sqrt((R + h) ** 2 + R ** 2 - 2 * R * (R + h) * np.cos(g))
    return np.tan(np.arcsin(R * np.sin(g) / rho) + g)


cols = ["flow_angle", "far_building", "far_ratio_building", "error_node", "width_node", "xtrk_dist_node", "sat_altitude"]
d = pq.read_table(NODES, columns=cols).to_pandas()
d["e"] = d.error_node.abs()
fa = d.flow_angle % 360
d["angle_dev"] = np.minimum.reduce([fa, (fa - 180).abs(), 360 - fa])
d = d[d.e.notna() & d.xtrk_dist_node.notna() & d.sat_altitude.notna()].copy()
d["tan_inc"] = tan_inc(d.xtrk_dist_node.values, d.sat_altitude.values)
C = d[(d.angle_dev <= 13) & (d.far_building > 0) & (d.far_ratio_building.fillna(0) >= 0.5)].copy()
C["disp"] = C.far_building / C.tan_inc
print(f"definition C n={len(C):,}")
for name, g in (("all", C), ("high |e|>=T", C[C.e >= T]), ("low |e|<T", C[C.e < T])):
    w = g.dropna(subset=["width_node"])
    p25, p50, p75 = np.nanpercentile(g.disp, [25, 50, 75])
    print(f"  {name:12s} n={len(g):6,} (with width {len(w):6,})  tan med {g.tan_inc.median():.4f}  "
          f"disp median {p50:.1f} m (IQR {p25:.1f}-{p75:.1f})  >=w {100*(w.disp >= w.width_node).mean():.1f}%  "
          f">=w/2 {100*(w.disp >= 0.5*w.width_node).mean():.1f}%  bldg med {g.far_building.median():.2f} m  "
          f"width med {w.width_node.median():.0f} m")
fa2 = d.flow_angle % 180
d["ang"] = np.minimum(fa2, 180 - fa2)
for lim in (10, 13):
    S = d[(d.ang <= lim) & (d.far_building >= 5)].copy()
    dr = S.far_building / S.tan_inc; ratio = dr / S.width_node
    print(f"S5 population ang<={lim}: n={len(S):,}  median dr {dr.median():.0f} m  ratio>=1 {100*(ratio >= 1).mean():.1f}%  "
          f"ratio>=0.5 {100*(ratio >= 0.5).mean():.1f}%")
