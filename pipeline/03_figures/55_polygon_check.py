"""2026-10-05: is the 12-site polygon (selected_12_vD.shp, used for the exclusion experiment) the same as the reach
buffer polygon in all_reaches_vD.parquet (Figure 2 'Reach Buffer Zones'; 2.5 widths each side)?"""
import geopandas as gpd, numpy as np
W = "/work/pi_kandread_umass_edu/swot-urban"
s = gpd.read_file(f"{W}/2_urban_reaches/selected_12_vD.shp")
a = gpd.read_parquet(f"{W}/2_urban_reaches/all_reaches_vD.parquet")
print("selected_12 cols:", list(s.columns)[:20], s.geom_type.unique(), s.crs)
print("all_reaches cols:", [c for c in a.columns if 'geom' in c or 'buf' in c.lower() or 'width' in c.lower()][:10], a.geom_type.unique()[:3], a.crs)
s = s.to_crs(5070); a = a.to_crs(5070); a = a.drop_duplicates("reach_id")
s["reach_id"] = s.reach_id.astype("int64"); a["reach_id"] = a.reach_id.astype("int64")
m = s.merge(a[["reach_id", "geometry", "width"]].rename(columns={"geometry": "g2", "width": "w2"}), on="reach_id", how="left")
for _, r in m.iterrows():
    inter = r.geometry.intersection(r.g2).area if r.g2 is not None else np.nan
    print(f"  {r.reach_id}: area sel {r.geometry.area/1e6:.3f} km2 | all_reaches {r.g2.area/1e6 if r.g2 is not None else float('nan'):.3f} km2 | overlap/sel {inter/r.geometry.area:.3f} | width {r.w2}")
