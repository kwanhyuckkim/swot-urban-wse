"""2026-10-04: recompute every Section V.C 'D' predictor score in ONE run with FIXED reach folds, twice,
to remove the run-to-run drift seen between 09-11/09-28 (+0.0563/+0.0380) and 10-04 (+0.0490/+0.0329).
Folds: sorted unique reach_id, permuted with default_rng(2026), fold = position % 5 (no GroupKFold argsort).
Population: node_coherence_all.parquet (520,958 rows) + layovr_val join as 25_pixc_population_check.py.
Also: original GroupKFold run of the same sets (for comparison), bridge-subset magnitudes, node_q_b bit 7."""
import numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy import stats as st
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

PROJ = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results"
RC = f"{PROJ}/reviewer_checks"; NODES = f"{PROJ}/swot_urban_nearfar_v6.5_results.parquet"; FILL, T = -1e11, 0.1832


def hour(s):
    return pd.to_datetime(s, errors="coerce", utc=True).dt.floor("h")


m = pd.read_parquet(f"{RC}/node_coherence_all.parquet")
print("columns:", [c for c in m.columns if c in ("ang", "isa", "bld", "e", "reach_id", "layover_impact_median", "bdist_m")])
n = pd.read_parquet(f"{RC}/layovr_val_node.parquet", columns=["node_id", "time_str", "layovr_val", "node_q_b"])
n["node_id"] = n.node_id.astype("int64"); n["layovr_val"] = n.layovr_val.where(n.layovr_val > FILL)
n["t"] = hour(n.time_str); n = n.dropna(subset=["t"]).drop_duplicates(["node_id", "t"])
m = m.merge(n[["node_id", "t", "layovr_val"]], on=["node_id", "t"], how="left")
m["ang_par"] = -m.ang
print(f"rows {len(m):,}")

reaches = np.sort(m.reach_id.astype("int64").unique())
perm = np.random.default_rng(2026).permutation(len(reaches))
fold_of = dict(zip(reaches[perm], np.arange(len(reaches)) % 5))
SETS = {"angle + ISA + building (ours)": ["ang_par", "isa", "bld"], "angle + ISA": ["ang_par", "isa"],
        "angle alone": ["ang_par"], "ISA alone": ["isa"], "building alone": ["bld"],
        "layover_impact alone": ["layover_impact_median"], "layovr_val alone": ["layovr_val"]}


def score(sub, cols, fixed):
    y = sub.e.values; yr = pd.Series(y).rank().values; pg = np.full(len(sub), np.nan)
    if fixed:
        f = sub.reach_id.astype("int64").map(fold_of).values
        splits = [(np.where(f != k)[0], np.where(f == k)[0]) for k in range(5)]
    else:
        splits = GroupKFold(n_splits=5).split(sub[cols], y, sub.reach_id.values)
    for tr, te in splits:
        pg[te] = HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.08, random_state=0
                                               ).fit(sub[cols].values[tr], yr[tr]).predict(sub[cols].values[te])
    return st.spearmanr(pg, y).statistic


for name, cols in SETS.items():
    sub = m.dropna(subset=cols + ["e"]).reset_index(drop=True)
    a, b = score(sub, cols, True), score(sub, cols, True)
    g = score(sub, cols, False)
    print(f"  {name:32s} n={len(sub):,}  fixed-fold run1 {a:+.4f} run2 {b:+.4f}  | GroupKFold {g:+.4f}")

br = m[m.bdist_m <= 100]
for col in ("layover_impact_median", "layovr_val"):
    k = br[[col, "e"]].dropna()
    print(f"  bridge<=100 m {col:22s} n={len(k):,} signed {st.spearmanr(k[col], k.e).statistic:+.4f} "
          f"|.| {st.spearmanr(k[col].abs(), k.e).statistic:+.4f}")

d = pq.read_table(NODES, columns=["node_id", "time_str", "error_node"]).to_pandas()
d["node_id"] = d.node_id.astype("int64"); d["t"] = hour(d.time_str)
j = d.merge(n[["node_id", "t", "node_q_b"]], on=["node_id", "t"], how="left")
q = j.node_q_b
print(f"\nnode table {len(j):,}; node_q_b present {q.notna().sum():,}; bit 7 (128) set "
      f"{int(((q.fillna(0).astype('int64') & 128) > 0).sum()):,}")
