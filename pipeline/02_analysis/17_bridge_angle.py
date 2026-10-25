#!/usr/bin/env python
"""
Bridges as layover sources, rotated 90 degrees from riverside buildings.

Riverside buildings line the bank, so their mass runs along the channel and their
layover reaches the water when the channel runs parallel to the ground track. A bridge
runs across the channel, so its mass is rotated by about 90 degrees, and the same
physics predicts that a bridge lays over onto the water when the channel runs
perpendicular to the track, which is the configuration that is safest for buildings.

Test: split reaches by gauge-to-bridge distance from the National Bridge Inventory,
then compare the error-versus-flow-angle curve between the groups. If the rotation
argument holds, the near-bridge group should carry relatively more error toward the
perpendicular end, where the bridge axis aligns with the track.
"""
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from scipy import stats as st

PROJ = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results"
WORK = "/work/pi_kandread_umass_edu/swot-urban"
OUT = f"{PROJ}/round2"
NODES = f"{PROJ}/swot_urban_nearfar_v6.5_results.parquet"
BR = f"{OUT}/bridge_check.csv"
NEAR_M = 100.0


def p68(x):
    return float(np.nanpercentile(x, 68)) if len(x) else np.nan


d = pq.read_table(NODES, columns=["reach_id", "flow_angle", "far_isa", "far_building",
                                  "error_node", "width_node",
                                  "sample_geometry_category"]).to_pandas()
d["reach_id"] = d.reach_id.astype("int64")
d["e"] = d.error_node.abs()
fa = d.flow_angle % 360
d["ang"] = np.minimum.reduce([fa, (fa - 180).abs(), 360 - fa])
d["isa_pct"] = (d.far_isa * 100 if d.far_isa.max() <= 1 else d.far_isa).fillna(0)
d["bld"] = d.far_building.fillna(0)
d = d.dropna(subset=["e", "ang"])

br = pd.read_csv(BR, dtype={"site_no": str}).groupby("reach_id").bridge_dist_m.min().reset_index()
br["reach_id"] = br.reach_id.astype("int64")
d = d.merge(br, on="reach_id", how="left")
d["bridge_dist_m"] = d.bridge_dist_m.fillna(1e9)
d["grp"] = np.where(d.bridge_dist_m <= NEAR_M, f"bridge <= {NEAR_M:.0f} m", "bridge farther")
# The bridge axis is roughly perpendicular to the channel, so its deviation from the
# track is the complement of the channel's.
d["bridge_ang"] = 90.0 - d.ang

print(f"nodes {len(d):,} | reaches {d.reach_id.nunique()} | "
      f"near-bridge reaches {d[d.grp.str.startswith('bridge <=')].reach_id.nunique()}")

print("\n[1] error against channel-to-track angle, by bridge proximity")
bins = [0, 10, 20, 30, 45, 60, 75, 90]
rows = []
for g, sub in d.groupby("grp"):
    for lo, hi in zip(bins[:-1], bins[1:]):
        s = sub[(sub.ang >= lo) & (sub.ang < hi)]
        if len(s) < 300:
            continue
        rows.append({"group": g, "ang_lo": lo, "ang_hi": hi, "n": len(s),
                     "p68_m": p68(s.e), "median_m": float(s.e.median()),
                     "isa_med": float(s.isa_pct.median()),
                     "bld_med": float(s.bld.median())})
t = pd.DataFrame(rows)
piv = t.pivot(index=["ang_lo", "ang_hi"], columns="group", values="p68_m")
cnt = t.pivot(index=["ang_lo", "ang_hi"], columns="group", values="n")
print(piv.round(3).to_string())
print("\n  n per cell:")
print(cnt.to_string())

print("\n[2] parallel-to-perpendicular ratio (buildings predict a large ratio)")
for g, sub in d.groupby("grp"):
    par = sub[sub.ang <= 13]
    perp = sub[sub.ang >= 77]
    if len(par) < 300 or len(perp) < 300:
        continue
    print(f"  {g:18s} parallel p68 {p68(par.e):.3f} | perpendicular p68 {p68(perp.e):.3f} "
          f"| ratio {p68(par.e)/p68(perp.e):.2f}  (n {len(par):,} / {len(perp):,})")

print("\n[3] does error track the bridge axis where bridges are close")
for g, sub in d.groupby("grp"):
    r1, p1 = st.spearmanr(sub.ang, sub.e)
    r2, p2 = st.spearmanr(sub.bridge_ang.abs(), sub.e)
    print(f"  {g:18s} vs channel angle {r1:+.3f} (p={p1:.1e}) | "
          f"vs bridge axis angle {r2:+.3f} (p={p2:.1e})")

print("\n[4] perpendicular zone only: does bridge proximity raise the error there")
perp = d[d.ang >= 77]
a = perp[perp.grp.str.startswith("bridge <=")].e
b = perp[~perp.grp.str.startswith("bridge <=")].e
if len(a) > 100 and len(b) > 100:
    mw = st.mannwhitneyu(a, b)
    print(f"  near-bridge p68 {p68(a):.3f} (n={len(a):,}) vs farther {p68(b):.3f} "
          f"(n={len(b):,})  MW p={mw.pvalue:.2e}")
par = d[d.ang <= 13]
a2 = par[par.grp.str.startswith("bridge <=")].e
b2 = par[~par.grp.str.startswith("bridge <=")].e
if len(a2) > 100 and len(b2) > 100:
    mw2 = st.mannwhitneyu(a2, b2)
    print(f"  same split in the parallel zone: {p68(a2):.3f} vs {p68(b2):.3f}  "
          f"MW p={mw2.pvalue:.2e}")

print("\n[5] control: is the near-bridge group simply more urban")
for g, sub in d.groupby("grp"):
    print(f"  {g:18s} far ISA median {sub.isa_pct.median():5.1f}% | "
          f"building median {sub.bld.median():4.1f} m | width median {sub.width_node.median():.0f} m")

t.to_csv(f"{OUT}/bridge_angle.csv", index=False)
print(f"\nsaved {OUT}/bridge_angle.csv")
