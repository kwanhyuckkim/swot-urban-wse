#!/usr/bin/env python
"""2026-10-03: re-check the Section II reach-width sentence against SWORD v17b.

Manuscript, Figure 2 paragraph: "Reach widths in SWORD range from 80 m on narrow urban
streams to about 1330 m on the Missouri River." This reads the SWORD v17b NA reach
shapefiles directly (width = reach mean width, max_width) for the study reaches taken
from two sources: all_reaches_vD.parquet (REACH_TABLE of 06_reviewer_product_checks.py)
and the node-observation results parquet (the 409 reaches of the error analysis).
Writes sword_width_check.json and one per-reach CSV per source into OUT.
"""
import glob
import json
import sys

import pandas as pd
import pyarrow.parquet as pq

WORK = "/work/pi_kandread_umass_edu/swot-urban"
PROJ = "/project/pi_kandread_umass_edu/swot-urban"
SWORD_SHP = "/nas/cee-ice/data/SWORD/SWORD_v17b/shp/NA"
REACH_TABLE = f"{WORK}/2_urban_reaches/all_reaches_vD.parquet"
RESULTS = f"{PROJ}/3_analysis_results/swot_urban_nearfar_v6.5_results.parquet"
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
WANT = ["reach_id", "river_name", "width", "max_width", "reach_len", "n_nodes", "type",
        "lakeflag", "obstr_type", "facc", "strm_order"]


def fields(path):
    try:
        import pyogrio
        return list(pyogrio.read_info(path)["fields"])
    except Exception:
        import geopandas as gpd
        return list(gpd.read_file(path, rows=1).columns)


def read_attrs(path, cols):
    try:
        import pyogrio
        return pyogrio.read_dataframe(path, columns=cols, read_geometry=False)
    except Exception:
        import geopandas as gpd
        return pd.DataFrame(gpd.read_file(path).drop(columns="geometry"))[cols]


files = sorted(glob.glob(f"{SWORD_SHP}/*reaches*.shp"))
have = fields(files[0])
cols = [c for c in WANT if c in have]
print("SWORD reach fields:", have)
print("using:", cols)
sword = pd.concat([read_attrs(f, cols) for f in files], ignore_index=True)
sword["reach_id"] = sword.reach_id.astype("int64")
print(f"SWORD v17b NA reaches: {len(sword):,}")

rt_cols = pq.read_schema(REACH_TABLE).names
print("REACH_TABLE columns:", rt_cols)
rt = pq.read_table(REACH_TABLE, columns=[c for c in ["reach_id", "p_width", "river_name"] if c in rt_cols]
                   ).to_pandas().drop_duplicates("reach_id")
rt["reach_id"] = rt.reach_id.astype("int64")
rs = pq.read_table(RESULTS, columns=["reach_id"]).to_pandas().drop_duplicates()
rs["reach_id"] = rs.reach_id.astype("int64")
print(f"ids: all_reaches_vD {len(rt)}, results {len(rs)}, "
      f"in results not in table {len(set(rs.reach_id) - set(rt.reach_id))}, "
      f"in table not in results {len(set(rt.reach_id) - set(rs.reach_id))}")

out = {"sword_na_reaches": int(len(sword))}
for name, s in (("all_reaches_vD", rt), ("results_v6.5", rs)):
    ids = set(s.reach_id)
    m = sword[sword.reach_id.isin(ids)].copy()
    d = {"n_ids": len(ids), "n_matched": int(len(m))}
    for col in [c for c in ("width", "max_width") if c in m]:
        v = m[col].astype(float)
        lo, hi = v.idxmin(), v.idxmax()
        top = m.nlargest(5, col)[["reach_id", "river_name", col]]
        bot = m.nsmallest(5, col)[["reach_id", "river_name", col]]
        d[col] = {"min": float(v.min()), "max": float(v.max()), "p5": float(v.quantile(.05)),
                  "p50": float(v.median()), "p95": float(v.quantile(.95)),
                  "min_reach": {"reach_id": int(m.loc[lo, "reach_id"]), "river_name": str(m.loc[lo, "river_name"])},
                  "max_reach": {"reach_id": int(m.loc[hi, "reach_id"]), "river_name": str(m.loc[hi, "river_name"])},
                  "top5": top.to_dict("records"), "bottom5": bot.to_dict("records")}
        print(f"\n[{name}] {col}: n {len(m)}, min {v.min():.1f} m ({m.loc[lo, 'river_name']}), "
              f"max {v.max():.1f} m ({m.loc[hi, 'river_name']}), median {v.median():.1f} m")
        print("  widest 5:\n" + top.to_string(index=False))
        print("  narrowest 5:\n" + bot.to_string(index=False))
    mo = m[m.river_name.astype(str).str.contains("Missouri", case=False)]
    d["missouri"] = {"n": int(len(mo)), "widths": sorted(mo["width"].astype(float).round(1).tolist())}
    print(f"  Missouri-named reaches: {len(mo)}; widths {d['missouri']['widths']}")
    if "p_width" in s:
        pw = s.set_index("reach_id")["p_width"].astype(float)
        d["p_width_riversp"] = {"min": float(pw.min()), "max": float(pw.max()), "n": int(pw.notna().sum())}
        print(f"  RiverSP p_width: min {pw.min():.1f} m, max {pw.max():.1f} m (n {pw.notna().sum()})")
    out[name] = d
    m.sort_values("width").to_csv(f"{OUT}/sword_width_{name}.csv", index=False)

with open(f"{OUT}/sword_width_check.json", "w") as fh:
    json.dump(out, fh, indent=1, default=str)
print("\nsaved", f"{OUT}/sword_width_check.json")
