"""2026-10-05: 12-site cross-level (exclusion) experiment with the COVERING-tile rule. Same extraction and
statistics as 06_reviewer_product_checks.run_cross_level_check (max_per_site=None), except the PIXC file:
instead of the first file within 300 s, try every file within 300 s in order of |dt| and keep the first that
yields >= 50 water pixels in the site polygon (_cross_level_one's own threshold). Output: reviewer_checks/covering_20261005/."""
import os, sys, glob, importlib
import numpy as np, pandas as pd, geopandas as gpd, pyarrow.parquet as pq
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from scipy import stats as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
rpc = importlib.import_module("06_reviewer_product_checks")
OUT = f"{rpc.OUT_DIR}/covering_20261005"; os.makedirs(OUT, exist_ok=True)
sites = gpd.read_file(rpc.SITES_SHP); col = "fctimp_mea" if "fctimp_mea" in sites.columns else "fctimp_mean"
RIVERSP = pq.read_table(rpc.ERROR_TABLE, columns=["reach_id", "time_str", "reach_q_b", "wse_relative_m", "wse_usgs_relative_m"]).to_pandas()
RIVERSP["reach_id"] = RIVERSP.reach_id.astype("int64"); RIVERSP["t"] = pd.to_datetime(RIVERSP.time_str).dt.tz_localize(None)
tasks = []
for d in sorted(glob.glob(f"{rpc.PROJ}/0_data_collection/SWOT_PIXC/fctimp_*")):
    isa = os.path.basename(d).replace("fctimp_", "")
    sample = f"{rpc.WORK}/1_data_processing/fctimp_mean_{isa}_sample.parquet"
    if not os.path.exists(sample): continue
    poly = sites[np.isclose(sites[col], float(isa), atol=0.01)].to_crs("EPSG:4326")
    if poly.empty: continue
    obs = pd.read_parquet(sample); obs["t"] = pd.to_datetime(obs.time_str).dt.tz_localize(None)
    obs["error"] = obs.wse_relative_m - obs.wse_usgs_relative_m; obs["abs_error"] = obs.error.abs()
    pixc = rpc._best_versions(glob.glob(f"{d}/SWOT_L2_HR_PIXC_*.nc")); vec_dir = f"{rpc.PROJ}/0_data_collection/SWOT_PIXCVec/fctimp_{isa}"
    starts = [(pd.to_datetime(os.path.basename(f).split("_")[7], format="%Y%m%dT%H%M%S"), f) for f in pixc]
    for _, o in obs.iterrows():
        cands = sorted(((abs((s - o.t).total_seconds()), f) for s, f in starts if abs((s - o.t).total_seconds()) <= 300), key=lambda z: (z[0], z[1]))
        if not cands: continue
        files = [(f, os.path.join(vec_dir, os.path.basename(f).replace("PIXC_", "PIXCVec_"))) for _, f in cands]
        tasks.append((files, poly, float(isa), int(o.reach_id), str(o.time_str), float(o.error), float(o.abs_error)))
print(f"matched observations: {len(tasks)}", flush=True)


def work(task):
    files, poly, isa, rid, ts, err, aerr = task
    for k, (f, v) in enumerate(files):
        r = rpc._cross_level_one(((f, v, poly, isa, rid, ts, err, aerr), RIVERSP, 0.1832))
        if r:
            r["pixc_file"] = os.path.basename(f); r["candidate_rank"] = k; r["n_candidates"] = len(files)
            return r
    return None


mp.set_start_method("fork", force=True)
with ProcessPoolExecutor(max_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))) as ex:
    rows = [r for r in ex.map(work, tasks, chunksize=1) if r]
out = pd.DataFrame(rows); out.to_parquet(f"{OUT}/cross_level_covering.parquet", index=False)
print(f"kept passes {len(out)} of {len(tasks)}; by site isa: {out.groupby('isa').size().to_dict()}")
print(f"chosen file was first candidate in {int((out.candidate_rank == 0).sum())}, later candidate in {int((out.candidate_rank > 0).sum())}")
old = pd.read_parquet(f"{rpc.OUT_DIR}/cross_level_pixc_vs_riversp.parquet")
j = old.merge(out, on=["reach_id", "time_str"], how="outer", suffixes=("_old", "_new"), indicator=True)
print("vs stored 166:", j._merge.value_counts().to_dict())
b = j[j._merge == "both"]
for c in ("h_all", "delta_layover_m", "delta_bright_m"):
    print(f"  {c}: max |old-new| on shared passes {(b[c+'_old'] - b[c+'_new']).abs().max():.3g}")


def bstats(o, lab):
    print(f"\n[{lab}] passes {len(o)}, sites {o.isa.nunique() if len(o) else 0}")
    if not len(o): return
    for tag, dcol in (("layover", "delta_layover_m"), ("bright", "delta_bright_m")):
        s = o[["error_m", "abs_error_m", dcol]].dropna(); cor = (s.error_m + s[dcol]).abs()
        print(f"  exclude {tag:7s}: improves {int((cor < s.abs_error_m).sum())}/{len(s)}; mean |e| {s.abs_error_m.mean():.3f} -> {cor.mean():.3f} m; median |e| {s.abs_error_m.median():.3f} -> {cor.median():.3f}")
    sb = o.delta_bright_m.dropna(); sl = o.layover_pixel_bias_m.dropna()
    print(f"  bright removal lowers pass median in {int((sb < 0).sum())}/{len(sb)}, median shift {sb.median():+.3f} m")
    print(f"  layover-pixel bias positive in {int((sl > 0).sum())}/{len(sl)}")


bstats(out, "covering-tile rule, all sites")
bstats(out[~np.isclose(out.isa, 0.61)], "covering-tile rule, without Des Plaines")
bstats(out[np.isclose(out.isa, 0.61)], "Des Plaines only")
bstats(old, "stored 08-05 (first-file rule)")
