#!/usr/bin/env python
"""2026-10-02: single-criterion mask effects on ONE population (parallel zone, angle_dev <= 13),
so the abstract can quote height-only, ISA-only and full-mask results with one denominator.
Checks: height-only 8.9 % / 22.7 % (parallel 73,190); ISA-only 17.9 % / 21.5 % on the far-ISA-bearing
subset (Table II row 3, n 41,091); full mask 12.3 % / 27.3 % (Table II row 2). New: ISA-only on all parallel."""
import numpy as np, pandas as pd, pyarrow.parquet as pq
PROJ = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results"
NODES = f"{PROJ}/swot_urban_nearfar_v6.5_results.parquet"
ANGLE, HEIGHT, ISA = 13.0, 8.0, 19.0
d = pq.read_table(NODES, columns=["reach_id", "flow_angle", "far_isa", "far_building", "error_node"]).to_pandas()
d["e"] = d.error_node.abs()
fa = d.flow_angle % 360
d["angle_dev"] = np.minimum.reduce([fa, (fa - 180).abs(), 360 - fa])
d["isa_pct"] = d.far_isa * 100 if d.far_isa.max() <= 1 else d.far_isa
d = d.dropna(subset=["e"])
par = d[d.angle_dev <= ANGLE]
p68 = lambda x: float(np.nanpercentile(x, 68))
def row(name, pop, flag):
    b, a = p68(pop.e), p68(pop.e[~flag])
    print(f"{name:52s} n {len(pop):7,} reaches {pop.reach_id.nunique():4d} p68 {b:.3f} -> {a:.3f} "
          f"reduction {100*(b-a)/b:5.2f}% flagged {100*flag.mean():5.2f}%")
h = par.far_building.fillna(0) > HEIGHT
i = par.isa_pct.fillna(0) > ISA
row("parallel: height only", par, h)
row("parallel: ISA only", par, i)
row("parallel: full mask (height or ISA)", par, h | i)
sub = par[par.isa_pct.fillna(0) > 0]
row("parallel & far ISA > 0: ISA only (Table II row 3)", sub, sub.isa_pct > ISA)
