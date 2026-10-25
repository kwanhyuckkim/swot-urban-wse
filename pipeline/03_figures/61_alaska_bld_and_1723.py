"""2026-10-05: (i) are the Alaska building heights missing (NaN) or real zeros in the node results? (ii) what does the
GBH2020 raster hold at the three Alaska gauges? (iii) does the gauge-reach matchup file have 1,723 reaches? Read-only."""
import numpy as np, pandas as pd, pyarrow.parquet as pq, rasterio
from rasterio.windows import Window
W = "/work/pi_kandread_umass_edu/swot-urban"; PROJ = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results"
AK = {78380100051, 81130400021, 81130400041}
d = pq.read_table(f"{PROJ}/swot_urban_nearfar_v6.5_results.parquet",
                  columns=["reach_id", "far_building", "near_building", "far_isa", "near_isa", "far_ratio_building"]).to_pandas()
d["reach_id"] = d.reach_id.astype("int64"); a = d[d.reach_id.isin(AK)]
for c in ["far_building", "near_building", "far_isa", "near_isa", "far_ratio_building"]:
    print(f"Alaska {c}: n={len(a)} NaN={int(a[c].isna().sum())} zero={int((a[c] == 0).sum())} >0={int((a[c] > 0).sum())}")
gauges = {"15024800": (56.707924, -132.131971), "15266300": (60.4775, -151.079444), "15266110": (60.466076, -150.601071)}
with rasterio.open(f"{W}/0_data_collection/3_150m_building/GBH2020_150m_GEDI_4326.tif") as src:
    print("GBH bounds", src.bounds, "nodata", src.nodata, "dtype", src.dtypes[0], "res", src.res)
    for s, (lat, lon) in gauges.items():
        r, c = src.index(lon, lat); win = src.read(1, window=Window(c - 20, r - 20, 41, 41))
        nd = src.nodata
        vals = win[win != nd] if nd is not None else win
        print(f"  gauge {s} ({lat:.2f},{lon:.2f}): 41x41 window -> nodata {int((win == nd).sum()) if nd is not None else 'n/a'}, "
              f"zero {int((win == 0).sum())}, >0 {int((win > 0).sum())}, max {float(win.max()):.2f}, unique(first) {np.unique(win)[:6]}")
m = pq.read_table(f"{W}/1_data_processing/usgs_sword_matchups_merritt_v2.parquet").to_pandas()
print("matchup rows", len(m), "unique reaches", m.reach_id.nunique(), "unique sites", m.site_no.nunique() if "site_no" in m else "n/a")
ar = pq.read_table(f"{W}/2_urban_reaches/all_reaches_vD.parquet", columns=["reach_id"]).to_pandas()
print("409 reaches inside matchup:", int(ar.reach_id.astype('int64').isin(m.reach_id.astype('int64')).sum()), "of", ar.reach_id.nunique())
