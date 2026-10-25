"""2026-10-04: follow-up checks for the V.C rewrite (cloud session questions).
A  node table (543,036 with layovr_val; join as 43/19): base rate of |e| >= T, top-decile high-error share by
   signed layovr_val, by |layovr_val|, and by out-of-fold boosted predictions (signed and |.| input; as 23 l.40-47).
B  node_coherence_all + layovr_val ('all rows' block of 25_pixc_population_check.py): boosted out-of-fold
   Spearman with signed vs absolute input, for layovr_val and layover_impact_median (manuscript D: +0.056/+0.038).
C  reach observations above 30% buffer ISA: 767 or 769? counts with and without (reach_id, hour) dedup."""
import numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy import stats as st
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

PROJ = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results"
RC = f"{PROJ}/reviewer_checks"; W = "/work/pi_kandread_umass_edu/swot-urban"
NODES = f"{PROJ}/swot_urban_nearfar_v6.5_results.parquet"; FILL, T = -1e11, 0.1832


def hour(s):
    return pd.to_datetime(s, errors="coerce", utc=True).dt.floor("h")


def boosted(mm, cols):
    y = mm.e.values; yr = pd.Series(y).rank().values; g = mm.reach_id.values
    pg = np.full(len(mm), np.nan)
    for tr, te in GroupKFold(n_splits=5).split(mm[cols], y, g):
        pg[te] = HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.08, random_state=0
                                               ).fit(mm[cols].values[tr], yr[tr]).predict(mm[cols].values[te])
    return pg


def top10(score, e):
    top = score >= np.nanpercentile(score, 90)
    return 100 * (e[top] >= T).mean(), int(top.sum())


lvn = pd.read_parquet(f"{RC}/layovr_val_node.parquet", columns=["node_id", "time_str", "layovr_val"])
lvn["node_id"] = lvn.node_id.astype("int64"); lvn["layovr_val"] = lvn.layovr_val.where(lvn.layovr_val > FILL)
lvn["t"] = hour(lvn.time_str); lvn = lvn.dropna(subset=["t"]).drop_duplicates(["node_id", "t"])[["node_id", "t", "layovr_val"]]

# ---------------- A
d = pq.read_table(NODES, columns=["reach_id", "node_id", "time_str", "error_node"]).to_pandas()
d["e"] = d.error_node.abs(); d["t"] = hour(d.time_str)
d["node_id"] = d.node_id.astype("int64"); d["reach_id"] = d.reach_id.astype("int64")
m = d.dropna(subset=["e", "t"]).merge(lvn, on=["node_id", "t"], how="left")
mm = m.dropna(subset=["layovr_val"]).reset_index(drop=True); mm["absl"] = mm.layovr_val.abs()
e = mm.e.values
print(f"[A] node table rows {len(m):,}; with layovr_val {len(mm):,}; base |e|>=T {100*(m.e>=T).mean():.2f}% (all rows), "
      f"{100*(e>=T).mean():.2f}% (with layovr_val)")
for lab, sc in (("signed layovr_val", mm.layovr_val.values), ("|layovr_val|", mm.absl.values)):
    s, k = top10(sc, e); print(f"  top decile by {lab:22s} high-error {s:.2f}% (n={k:,})")
for lab, cols in (("boosted, signed input", ["layovr_val"]), ("boosted, |.| input", ["absl"])):
    pg = boosted(mm, cols); s, k = top10(pg, e)
    print(f"  {lab:30s} rho {st.spearmanr(pg, e).statistic:+.4f} | top decile high-error {s:.2f}% (n={k:,})")

# ---------------- B
c = pd.read_parquet(f"{RC}/node_coherence_all.parquet")
c2 = c.merge(lvn, on=["node_id", "t"], how="left")
print(f"\n[B] node_coherence_all rows {len(c2):,}")
for col in ("layovr_val", "layover_impact_median"):
    sub = c2.dropna(subset=[col, "e"]).reset_index(drop=True); sub["absx"] = sub[col].abs()
    r_s = st.spearmanr(boosted(sub, [col]), sub.e).statistic
    r_a = st.spearmanr(boosted(sub, ["absx"]), sub.e).statistic
    print(f"  {col:22s} n={len(sub):,} boosted signed {r_s:+.4f} | boosted |.| {r_a:+.4f} | raw signed "
          f"{st.spearmanr(sub[col], sub.e).statistic:+.4f} | raw |.| {st.spearmanr(sub.absx, sub.e).statistic:+.4f}")

# ---------------- C
ob = pq.read_table(f"{W}/2_urban_reaches/all_final_shape.parquet",
                   columns=["reach_id", "time_str", "wse_relative_m", "wse_usgs_relative_m"]).to_pandas()
ob["reach_id"] = ob.reach_id.astype("int64"); ob["e"] = (ob.wse_relative_m - ob.wse_usgs_relative_m).abs()
ob["t"] = hour(ob.time_str)
rv = pq.read_table(f"{W}/2_urban_reaches/all_reaches_vD.parquet", columns=["reach_id", "fctimp_mean"]).to_pandas()
rv["reach_id"] = rv.reach_id.astype("int64"); rv = rv.drop_duplicates("reach_id")
ob = ob.merge(rv, on="reach_id", how="left")
hi = ob[ob.fctimp_mean > 0.30]
print(f"\n[C] reach obs rows {len(ob):,}; ISA>30% {len(hi):,} ({hi.reach_id.nunique()} reaches); "
      f"unique (reach_id, hour) {ob.drop_duplicates(['reach_id','t']).shape[0]:,} / ISA>30% {hi.drop_duplicates(['reach_id','t']).shape[0]:,}")
r = pd.read_parquet(f"{RC}/layovr_val_reach.parquet", columns=["reach_id", "time_str", "layovr_val"])
r["reach_id"] = r.reach_id.astype("int64"); r["layovr_val"] = r.layovr_val.where(r.layovr_val > FILL); r["t"] = hour(r.time_str)
r = r.dropna(subset=["t", "layovr_val"])
print(f"  layovr_val_reach rows {len(r):,}; duplicated (reach_id, hour) {r.duplicated(['reach_id','t']).sum():,}")
for lab, rr in (("as 23 [4] (no dedup)", r), ("dedup (reach_id, hour)", r.drop_duplicates(["reach_id", "t"]))):
    j = rr.merge(hi[["reach_id", "t", "e"]].dropna(), on=["reach_id", "t"], how="inner")
    ja = rr.merge(ob[["reach_id", "t", "e"]].dropna(), on=["reach_id", "t"], how="inner")
    print(f"  {lab:24s} all n={len(ja):,} rho {st.spearmanr(ja.layovr_val, ja.e).statistic:+.4f} | "
          f"ISA>30% n={len(j):,} rho {st.spearmanr(j.layovr_val, j.e).statistic:+.4f}")
