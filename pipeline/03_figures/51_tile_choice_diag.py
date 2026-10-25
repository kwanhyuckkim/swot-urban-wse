"""2026-10-05: does the 12-site sweep drop observations because it takes the FIRST PIXC file within 300 s
(06_reviewer_product_checks.run_cross_level_check) rather than the tile that covers the reach?
For Des Plaines (0.61) and one kept site (Platte, 0.06): every time-matched file, its order, dt, pixels in polygon."""
import os, sys, glob, importlib
import numpy as np, pandas as pd, geopandas as gpd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
rpc = importlib.import_module("06_reviewer_product_checks")
sites = gpd.read_file(rpc.SITES_SHP); col = "fctimp_mea" if "fctimp_mea" in sites.columns else "fctimp_mean"
for isa in ("0.61", "0.06"):
    poly = sites[np.isclose(sites[col], float(isa), atol=0.01)].to_crs("EPSG:4326")
    obs = pd.read_parquet(f"{rpc.WORK}/1_data_processing/fctimp_mean_{isa}_sample.parquet")
    obs["t"] = pd.to_datetime(obs.time_str).dt.tz_localize(None)
    pixc = rpc._best_versions(glob.glob(f"{rpc.PROJ}/0_data_collection/SWOT_PIXC/fctimp_{isa}/SWOT_L2_HR_PIXC_*.nc"))
    vec_dir = f"{rpc.PROJ}/0_data_collection/SWOT_PIXCVec/fctimp_{isa}"
    print(f"\n=== fctimp_{isa}: {len(obs)} obs, {len(pixc)} PIXC files")
    first_ok = best_ok = 0
    for _, o in obs.iterrows():
        hits = []
        for i, f in enumerate(pixc):
            st = pd.to_datetime(os.path.basename(f).split("_")[7], format="%Y%m%dT%H%M%S")
            dt = abs((st - o.t).total_seconds())
            if dt <= 300:
                vec = os.path.join(vec_dir, os.path.basename(f).replace("PIXC_", "PIXCVec_"))
                df = rpc.extract_pixc_layover_fields(f, vec, poly); n = 0 if df is None else len(df)
                tile = "_".join(os.path.basename(f).split("_")[4:7])
                hits.append((i, int(dt), tile, n))
        first = hits[0][3] if hits else 0; best = max((h[3] for h in hits), default=0)
        first_ok += first >= 50; best_ok += best >= 50
        print(f"  {o.time_str}: " + "; ".join(f"#{i} dt{dt}s {tile} px{n}" for i, dt, tile, n in hits))
    print(f"  kept with FIRST match (original rule): {first_ok}/{len(obs)} | with best-covering tile: {best_ok}/{len(obs)}")
