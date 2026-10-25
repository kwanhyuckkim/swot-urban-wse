"""2026-10-05: Des Plaines only (site 12, fctimp 0.61): build its cross-level rows with the same function as the
166-pass table (06_reviewer_product_checks._cross_level_one), append to the stored 08-05 table, and recompute
the V.C 'B' numbers and the A1 partial correlation (output in reviewer_checks/desplaines_20261005/)."""
import os, sys, glob, importlib
import numpy as np, pandas as pd, geopandas as gpd, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
rpc = importlib.import_module("06_reviewer_product_checks")
OLD = rpc.OUT_DIR; isa = "0.61"
sites = gpd.read_file(rpc.SITES_SHP); col = "fctimp_mea" if "fctimp_mea" in sites.columns else "fctimp_mean"
poly = sites[np.isclose(sites[col], float(isa), atol=0.01)].to_crs("EPSG:4326")
riversp = pq.read_table(rpc.ERROR_TABLE, columns=["reach_id", "time_str", "reach_q_b", "wse_relative_m", "wse_usgs_relative_m"]).to_pandas()
riversp["reach_id"] = riversp.reach_id.astype("int64"); riversp["t"] = pd.to_datetime(riversp.time_str).dt.tz_localize(None)
obs = pd.read_parquet(f"{rpc.WORK}/1_data_processing/fctimp_mean_{isa}_sample.parquet")
obs["t"] = pd.to_datetime(obs.time_str).dt.tz_localize(None)
obs["error"] = obs.wse_relative_m - obs.wse_usgs_relative_m; obs["abs_error"] = obs.error.abs()
pixc = rpc._best_versions(glob.glob(f"{rpc.PROJ}/0_data_collection/SWOT_PIXC/fctimp_{isa}/SWOT_L2_HR_PIXC_*.nc"))
vec_dir = f"{rpc.PROJ}/0_data_collection/SWOT_PIXCVec/fctimp_{isa}"
rows = []
for _, o in obs.sort_values("abs_error", ascending=False).iterrows():
    match = next((f for f in pixc if abs((pd.to_datetime(os.path.basename(f).split("_")[7], format="%Y%m%dT%H%M%S") - o.t).total_seconds()) <= 300), None)
    if match is None: continue
    vec = os.path.join(vec_dir, os.path.basename(match).replace("PIXC_", "PIXCVec_"))
    r = rpc._cross_level_one(((match, vec, poly, float(isa), int(o.reach_id), str(o.time_str), float(o.error), float(o.abs_error)), riversp, 0.1832))
    if r: rows.append(r)
dp = pd.DataFrame(rows); print(f"Des Plaines rows kept: {len(dp)} of {len(obs)} sample observations")
old = pd.read_parquet(f"{OLD}/cross_level_pixc_vs_riversp.parquet")
allr = pd.concat([old, dp], ignore_index=True)
OUT = f"{OLD}/desplaines_20261005"; os.makedirs(OUT, exist_ok=True)
dp.to_parquet(f"{OUT}/desplaines_rows.parquet", index=False); allr.to_parquet(f"{OUT}/cross_level_with_desplaines.parquet", index=False)


def bstats(o, lab):
    print(f"\n[{lab}] passes {len(o)}, sites {o.isa.nunique()}")
    for tag, dcol in (("layover", "delta_layover_m"), ("bright", "delta_bright_m")):
        s = o[["error_m", "abs_error_m", dcol]].dropna(); cor = (s.error_m + s[dcol]).abs()
        print(f"  exclude {tag:7s}: improves {int((cor < s.abs_error_m).sum())}/{len(s)}; mean |e| {s.abs_error_m.mean():.3f} -> {cor.mean():.3f} m")
    sb = o.delta_bright_m.dropna(); sl = o.layover_pixel_bias_m.dropna()
    print(f"  bright removal lowers pass median in {int((sb < 0).sum())}/{len(sb)}, median shift {sb.median():+.3f} m")
    print(f"  layover-pixel bias positive in {int((sl > 0).sum())}/{len(sl)}")


bstats(old, "stored 08-05 table (manuscript)"); bstats(dp, "Des Plaines only"); bstats(allr, "12 sites = stored + Des Plaines")
sweep_cols = [c for c in allr.columns if c.startswith(("pct_", "n_", "layover_", "bright_", "prior_", "sig0_", "inc_", "height_"))] + ["isa", "reach_id", "time_str", "abs_error_m", "high_error"]
rpc.OUT_DIR = OUT
allr[[c for c in dict.fromkeys(sweep_cols) if c in allr.columns]].to_parquet(f"{OUT}/pixc_layover_sweep.parquet", index=False)
print("\n--- partial correlation, 12 sites")
rpc.run_partial_check(verbose=True)
