"""2026-10-04: signed vs absolute layovr_val against |error| on the manuscript's own populations (V.C).
Joins copied from 19_node_level_layover.py (node table x layovr_val_node on node_id + hour) and
23_number_checks.py [4] (reach obs x layovr_val_reach on reach_id + hour).
Reports n, share of negative layovr_val, Spearman(signed, |e|), Spearman(|lv|, |e|) with reach-cluster
bootstrap 95% intervals (B = 200), per stratum."""
import numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy import stats as st

PROJ = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results"
RC = f"{PROJ}/reviewer_checks"
W = "/work/pi_kandread_umass_edu/swot-urban"
NODES = f"{PROJ}/swot_urban_nearfar_v6.5_results.parquet"
FILL, B = -1e11, 200
rng = np.random.default_rng(0)


def boot(g, xcol, ycol="e"):
    g = g[[xcol, ycol, "reach_id"]].dropna()
    r0 = st.spearmanr(g[xcol], g[ycol]).statistic
    idx = g.groupby("reach_id").indices; keys = np.array(list(idx))
    bs = []
    for _ in range(B):
        pick = rng.choice(keys, len(keys), replace=True)
        ii = np.concatenate([idx[k] for k in pick])
        bs.append(st.spearmanr(g[xcol].values[ii], g[ycol].values[ii]).statistic)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return r0, lo, hi, len(g)


def report(lab, g, ycol="e"):
    g = g.copy(); g["absl"] = g.layovr_val.abs()
    v = g.layovr_val.dropna()
    s = boot(g, "layovr_val", ycol); a = boot(g, "absl", ycol)
    print(f"  {lab:46s} n={a[3]:>7,} reaches={g.reach_id.nunique():>3}  neg {100*(v<0).mean():5.1f}%  "
          f"signed {s[0]:+.4f} [{s[1]:+.3f},{s[2]:+.3f}]  |lv| {a[0]:+.4f} [{a[1]:+.3f},{a[2]:+.3f}]")


# ---------------- node scale (manuscript node table)
d = pq.read_table(NODES, columns=["reach_id", "node_id", "time_str", "flow_angle", "far_isa", "error_node"]).to_pandas()
d["e"] = d.error_node.abs()
fa = d.flow_angle % 360
d["ang"] = np.minimum.reduce([fa, (fa - 180).abs(), 360 - fa])
d["far_isa_pct"] = d.far_isa * 100 if d.far_isa.max() <= 1 else d.far_isa
d["t"] = pd.to_datetime(d.time_str, errors="coerce", utc=True).dt.floor("h")
d["node_id"] = d.node_id.astype("int64"); d["reach_id"] = d.reach_id.astype("int64")
print(f"node table rows {len(d):,}; with e and t {d.dropna(subset=['e','t']).shape[0]:,}")
d = d.dropna(subset=["e", "t"])
n = pd.read_parquet(f"{RC}/layovr_val_node.parquet", columns=["node_id", "time_str", "layovr_val"])
n["node_id"] = n.node_id.astype("int64"); n["layovr_val"] = n.layovr_val.where(n.layovr_val > FILL)
n["t"] = pd.to_datetime(n.time_str, errors="coerce", utc=True).dt.floor("h")
n = n.dropna(subset=["t"]).drop_duplicates(["node_id", "t"])[["node_id", "t", "layovr_val"]]
m = d.merge(n, on=["node_id", "t"], how="left")
print(f"left join rows {len(m):,}; layovr_val valid {m.layovr_val.notna().sum():,}")
rv = pq.read_table(f"{W}/2_urban_reaches/all_reaches_vD.parquet", columns=["reach_id", "fctimp_mean"]).to_pandas()
rv["reach_id"] = rv.reach_id.astype("int64"); rv = rv.drop_duplicates("reach_id")
m = m.merge(rv, on="reach_id", how="left")
print("\n[node layovr_val vs node |error|]")
report("all node observations", m)
report("parallel (angle <= 13)", m[m.ang <= 13])
report("parallel & far-nadir ISA >= 30%", m[(m.ang <= 13) & (m.far_isa_pct >= 30)])
report("parallel & far-nadir ISA >= 60%", m[(m.ang <= 13) & (m.far_isa_pct >= 60)])
report("reach buffer ISA > 30% (all angles)", m[m.fctimp_mean > 0.30])

# ---------------- reach scale
ob = pq.read_table(f"{W}/2_urban_reaches/all_final_shape.parquet",
                   columns=["reach_id", "time_str", "wse_relative_m", "wse_usgs_relative_m"]).to_pandas()
ob["reach_id"] = ob.reach_id.astype("int64")
ob["e"] = (ob.wse_relative_m - ob.wse_usgs_relative_m).abs()
ob["t"] = pd.to_datetime(ob.time_str, errors="coerce", utc=True).dt.floor("h")
r = pd.read_parquet(f"{RC}/layovr_val_reach.parquet", columns=["reach_id", "time_str", "layovr_val"])
r["reach_id"] = r.reach_id.astype("int64"); r["layovr_val"] = r.layovr_val.where(r.layovr_val > FILL)
r["t"] = pd.to_datetime(r.time_str, errors="coerce", utc=True).dt.floor("h")
jr = r.dropna(subset=["t", "layovr_val"]).merge(ob[["reach_id", "t", "e"]].dropna(), on=["reach_id", "t"], how="inner")
jr = jr.merge(rv, on="reach_id", how="left")
print("\n[reach layovr_val vs reach |error|] (23_number_checks [4] join)")
report("all reach observations", jr)
report("reach buffer ISA > 30%", jr[jr.fctimp_mean > 0.30])

# node estimate vs concurrent reach error (manuscript 'node scale 0.03' definition)
nv = pd.read_parquet(f"{RC}/layovr_val_node.parquet", columns=["reach_id", "node_id", "time_str", "layovr_val"])
nv["reach_id"] = nv.reach_id.astype("int64"); nv["layovr_val"] = nv.layovr_val.where(nv.layovr_val > FILL)
nv["t"] = pd.to_datetime(nv.time_str, errors="coerce", utc=True).dt.floor("h")
jn = nv.dropna(subset=["t", "layovr_val"]).merge(ob[["reach_id", "t", "e"]].dropna(), on=["reach_id", "t"], how="inner")
jn = jn.merge(rv, on="reach_id", how="left")
print("\n[node layovr_val vs concurrent reach |error|] (23_number_checks [4] definition)")
report("all", jn)
report("reach buffer ISA > 30%", jn[jn.fctimp_mean > 0.30])
