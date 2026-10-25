#!/usr/bin/env python
"""2026-10-02: how many parallel-zone node observations have far-nadir ISA = 0, and why."""
import numpy as np, pandas as pd, pyarrow.parquet as pq
PROJ = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results"
NODES = f"{PROJ}/swot_urban_nearfar_v6.5_results.parquet"
cols = ["reach_id", "node_id", "flow_angle", "far_isa", "near_isa", "far_ratio", "near_ratio",
        "far_building", "error_node", "geometry_category"]
d = pq.read_table(NODES, columns=cols).to_pandas()
d["e"] = d.error_node.abs(); d = d.dropna(subset=["e"])
fa = d.flow_angle % 360; d["angle_dev"] = np.minimum.reduce([fa, (fa - 180).abs(), 360 - fa])
d["isa_pct"] = d.far_isa * 100 if d.far_isa.max() <= 1 else d.far_isa
p68 = lambda x: float(np.nanpercentile(x, 68))
def summ(name, g):
    z = g.isa_pct == 0; nn = g.isa_pct.isna(); pos = g.isa_pct > 0
    print(f"[{name}] n {len(g):,} | far ISA NaN {nn.sum():,} | == 0 {z.sum():,} ({100*z.mean():.1f}%) | > 0 {pos.sum():,}")
    bins = [0, 1, 5, 19, 100.01]; lab = ["0-1", "1-5", "5-19", ">19"]
    gp = g[pos]; c = pd.cut(gp.isa_pct, bins, labels=lab, right=True, include_lowest=False).value_counts().reindex(lab)
    print("    far ISA > 0 by bin (%): " + ", ".join(f"{l}: {v:,}" for l, v in c.items()))
summ("all", d); par = d[d.angle_dev <= 13]; summ("parallel <= 13", par)
z = par[par.isa_pct == 0]
print(f"\nparallel & far ISA == 0: n {len(z):,} | reaches {z.reach_id.nunique()} of {par.reach_id.nunique()} | nodes {z.node_id.nunique():,}")
print(f"  far_ratio == 0 among them: {(z.far_ratio.fillna(0) == 0).mean()*100:.1f}% | far_building == 0: {(z.far_building.fillna(0) == 0).mean()*100:.1f}%")
print(f"  near ISA also 0: {(z.near_isa.fillna(0) == 0).sum():,} ({(z.near_isa.fillna(0) == 0).mean()*100:.1f}%)")
print(f"  geometry_category: {z.geometry_category.value_counts().to_dict()}")
print(f"  p68 error: far ISA == 0 {p68(z.e):.3f} m | far ISA > 0 {p68(par[par.isa_pct > 0].e):.3f} m | all parallel {p68(par.e):.3f} m")
# reach-level: are the zeros whole rural reaches or rural stretches of urban reaches?
r = par.groupby("reach_id").agg(n=("e", "size"), zero_share=("isa_pct", lambda s: (s == 0).mean()), reach_far_isa_mean=("isa_pct", "mean"))
print(f"\nreaches with parallel obs: {len(r)} | all-zero reaches: {(r.zero_share == 1).sum()} (obs {int(r.n[r.zero_share == 1].sum()):,}) | "
      f"no-zero reaches: {(r.zero_share == 0).sum()} | mixed: {((r.zero_share > 0) & (r.zero_share < 1)).sum()} (zero obs in mixed {int((r.n * r.zero_share)[(r.zero_share > 0) & (r.zero_share < 1)].sum()):,})")
top = r.sort_values("n", ascending=False).head(8)
print("largest parallel reaches (n, zero share, mean far ISA %):")
for rid, row in top.iterrows(): print(f"  {rid}: n {int(row.n):,}, zero {100*row.zero_share:.0f}%, mean far ISA {row.reach_far_isa_mean:.1f}%")
