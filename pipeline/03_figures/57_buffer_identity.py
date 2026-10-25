"""2026-10-05: is the non-urban control's 'corridor' (buffered_polygons_with_nlcd.parquet, used by 08_control_reaches_pull.py)
the same reach-buffer polygon as all_reaches_vD.parquet (Figure 2 'Reach Buffer Zones')?"""
import geopandas as gpd, pandas as pd
W = "/work/pi_kandread_umass_edu/swot-urban"
b = gpd.read_parquet(f"{W}/1_data_processing/buffered_polygons_with_nlcd.parquet")
a = gpd.read_parquet(f"{W}/2_urban_reaches/all_reaches_vD.parquet").drop_duplicates("reach_id")
print("buffered cols:", [c for c in b.columns][:15], len(b), b.crs)
b["reach_id"] = b.reach_id.astype("int64"); a["reach_id"] = a.reach_id.astype("int64")
b5 = b.to_crs(5070); a5 = a.to_crs(5070)
m = a5[["reach_id", "geometry", "fctimp_mean"]].merge(b5[["reach_id", "geometry", "fctimp_mean"]], on="reach_id", how="inner", suffixes=("_a", "_b"))
m["area_a"] = m.geometry_a.area; m["area_b"] = gpd.GeoSeries(m.geometry_b, crs=5070).area
m["ov"] = [ga.intersection(gb).area / ga.area for ga, gb in zip(m.geometry_a, m.geometry_b)]
fa = m.fctimp_mean_a * (100 if m.fctimp_mean_a.max() <= 1 else 1)
print(f"study reaches matched {len(m)}/409; area ratio median {(m.area_b/m.area_a).median():.4f}; overlap median {m.ov.median():.4f} min {m.ov.min():.4f}")
print(f"fctimp equal (pct scale): {(abs(fa - m.fctimp_mean_b) < 0.01).mean():.3f}; scales a max {m.fctimp_mean_a.max():.3f} b max {m.fctimp_mean_b.max():.3f}")
