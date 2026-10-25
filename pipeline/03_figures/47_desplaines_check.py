"""2026-10-04: why is Des Plaines (site 12, reach 74282700231, ISA 60.5%) absent from the 166-pass 12-site tables,
and is it in the V.A prior-land pixel statistics (priorland_check, 648 tiles)?"""
import glob, os
import pandas as pd, pyarrow.parquet as pq
W = "/work/pi_kandread_umass_edu/swot-urban"; P = "/project/pi_kandread_umass_edu/swot-urban"; R = f"{P}/3_analysis_results"
RID = 74282700231
for d in sorted(glob.glob(f"{P}/0_data_collection/SWOT_PIXC/fctimp_*")):
    isa = os.path.basename(d).replace("fctimp_", "")
    n_pixc = len(glob.glob(f"{d}/SWOT_L2_HR_PIXC_*.nc"))
    vec = f"{P}/0_data_collection/SWOT_PIXCVec/fctimp_{isa}"
    n_vec = len(glob.glob(f"{vec}/*.nc")) if os.path.isdir(vec) else -1
    samp = f"{W}/1_data_processing/fctimp_mean_{isa}_sample.parquet"
    ns = len(pd.read_parquet(samp)) if os.path.exists(samp) else -1
    print(f"fctimp_{isa}: PIXC {n_pixc}, PIXCVec {n_vec}, sample rows {ns}")
samp = f"{W}/1_data_processing/fctimp_mean_0.605_sample.parquet"
for cand in sorted(glob.glob(f"{W}/1_data_processing/fctimp_mean_0.6*_sample.parquet")):
    s = pd.read_parquet(cand); print(cand, len(s), "reach ids", s.reach_id.astype("int64").unique()[:5] if "reach_id" in s else "-")
ob = pq.read_table(f"{W}/2_urban_reaches/all_final_shape.parquet", columns=["reach_id", "time_str"]).to_pandas()
print("all_final_shape rows for Des Plaines:", int((ob.reach_id.astype("int64") == RID).sum()))
nd = pq.read_table(f"{R}/swot_urban_nearfar_v6.5_results.parquet", columns=["reach_id"]).to_pandas()
print("node table rows for Des Plaines:", int((nd.reach_id.astype("int64") == RID).sum()))
pl = f"{R}/round2/priorland_nodes_12site.parquet"
if os.path.exists(pl):
    m = pd.read_parquet(pl); print("priorland_nodes_12site rows", len(m), "| per reach:", m.reach_id.astype("int64").value_counts().to_dict())
for sh in sorted(glob.glob(f"{R}/round2/priorland*/tiles*.parquet"))[:3] + sorted(glob.glob(f"{R}/round2/priorland*/shard*.parquet"))[:2]:
    t = pd.read_parquet(sh); print(sh, len(t), list(t.columns)[:12])
