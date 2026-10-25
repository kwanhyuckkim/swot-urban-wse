"""2026-10-04: why Des Plaines (site 12, fctimp 0.61, reach 74282700231) yields no pass in the 166-pass sweep.
Replays the matching steps of run_pixc_sweep (06_reviewer_product_checks.py 338-400) for that site only."""
import glob, os, sys, importlib.util
import numpy as np, pandas as pd, geopandas as gpd
spec = importlib.util.spec_from_file_location("rpc", os.path.join(os.path.dirname(os.path.abspath(__file__)), "06_reviewer_product_checks.py"))
rpc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rpc)
isa = "0.61"; d = f"{rpc.PROJ}/0_data_collection/SWOT_PIXC/fctimp_{isa}"
sites = gpd.read_file(rpc.SITES_SHP); col = "fctimp_mea" if "fctimp_mea" in sites.columns else "fctimp_mean"
print("site fctimp values:", sorted(np.round(sites[col].astype(float), 4).tolist()))
poly = sites[np.isclose(sites[col], float(isa), atol=0.01)].to_crs("EPSG:4326")
print("polygons matched for 0.61:", len(poly))
obs = pd.read_parquet(f"{rpc.WORK}/1_data_processing/fctimp_mean_{isa}_sample.parquet")
obs["time_dt"] = pd.to_datetime(obs["time_str"]).dt.tz_localize(None)
pixc = rpc._best_versions(glob.glob(f"{d}/SWOT_L2_HR_PIXC_*.nc"))
starts = []
for f in pixc:
    try: starts.append((pd.to_datetime(os.path.basename(f).split("_")[7], format="%Y%m%dT%H%M%S"), f))
    except Exception: pass
print(f"PIXC files {len(glob.glob(f'{d}/SWOT_L2_HR_PIXC_*.nc'))}, best versions {len(pixc)}, parsed starts {len(starts)}")
vec_dir = f"{rpc.PROJ}/0_data_collection/SWOT_PIXCVec/fctimp_{isa}"
for _, o in obs.iterrows():
    dts = [abs((s - o.time_dt).total_seconds()) for s, _ in starts]
    k = int(np.argmin(dts)) if dts else -1
    line = f"  obs {o.time_str} |e|={abs(o.wse_relative_m - o.wse_usgs_relative_m):.3f}  nearest PIXC dt {dts[k]:.0f} s" if k >= 0 else "  no PIXC"
    if k >= 0 and dts[k] <= 300:
        f = starts[k][1]; vec = os.path.join(vec_dir, os.path.basename(f).replace("PIXC_", "PIXCVec_"))
        try:
            df = rpc.extract_pixc_layover_fields(f, vec, poly); n = 0 if df is None else len(df)
        except Exception as ex:
            n = f"error {ex!r}"[:120]
        line += f" | vec exists {os.path.exists(vec)} | water pixels in polygon {n}"
    print(line)
