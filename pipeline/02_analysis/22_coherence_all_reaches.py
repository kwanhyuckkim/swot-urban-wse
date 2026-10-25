#!/usr/bin/env python
"""
The coherence evaluation over all 409 reaches, stratified the way Brent asked.

The earlier version covered the twelve sites whose pixel clouds we hold, 49 reaches.
This one uses the streamed extraction, joins it to the node error table, and repeats
every test on the full record: coherence against node error, the combination models,
and the same structure and bridge strata used for the shipped layover estimate.
"""
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from scipy import stats as st
from scipy.spatial import cKDTree
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from statsmodels.stats.proportion import proportion_confint

PROJ = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results"
RC = f"{PROJ}/reviewer_checks"
OUT = f"{PROJ}/round2"
NODES = f"{PROJ}/swot_urban_nearfar_v6.5_results.parquet"
COH = f"{RC}/coh_all/coherence_all_reaches.parquet"
T = 0.1832


def p68(x):
    return float(np.nanpercentile(x, 68)) if len(x) else np.nan


def rho(a, b, lab):
    m = pd.notna(a) & pd.notna(b)
    if m.sum() < 50:
        print(f"  {lab:44s} n={m.sum()}"); return np.nan
    r, p = st.spearmanr(a[m], b[m])
    print(f"  {lab:44s} rho {r:+.4f} (p={p:.1e}, n={m.sum():,})")
    return r


c = pd.read_parquet(COH)
c["t"] = pd.to_datetime(c.t, format="%Y%m%dT%H%M%S", errors="coerce").dt.floor("h")
c["t"] = c.t.dt.tz_localize("UTC")
c = c[c.node_id > 0].drop_duplicates(["node_id", "t"])

d = pq.read_table(NODES, columns=["reach_id", "node_id", "time_str", "lat", "lon",
                                  "flow_angle", "far_isa", "far_building",
                                  "error_node"]).to_pandas()
d["node_id"] = d.node_id.astype("int64")
d["reach_id"] = d.reach_id.astype("int64")
d["e"] = d.error_node.abs()
fa = d.flow_angle % 360
d["ang"] = np.minimum.reduce([fa, (fa - 180).abs(), 360 - fa])
d["ang_par"] = -d.ang
d["isa"] = (d.far_isa * 100 if d.far_isa.max() <= 1 else d.far_isa).fillna(0)
d["bld"] = d.far_building.fillna(0)
d["t"] = pd.to_datetime(d.time_str, errors="coerce", utc=True).dt.floor("h")
m = d.dropna(subset=["e", "t"]).merge(
    c.drop(columns=["reach_id"]), on=["node_id", "t"], how="inner")

net = pd.read_parquet(f"{RC}/obstruction_network_distance.parquet",
                      columns=["reach_id", "nearest_km"])
net["reach_id"] = net.reach_id.astype("int64")
m = m.merge(net, on="reach_id", how="left")
m["nearest_km"] = m.nearest_km.fillna(1e9)
try:
    b = pd.read_parquet(f"{OUT}/nbi_waterway_bridges.parquet")
    lat0 = np.radians(m.lat.mean())
    kx, ky = 111.32 * np.cos(lat0), 111.32
    tree = cKDTree(np.column_stack([b.lon * kx, b.lat * ky]))
    m["bdist_m"] = tree.query(np.column_stack([m.lon * kx, m.lat * ky]))[0] * 1000
except Exception:
    m["bdist_m"] = np.nan

base = (m.e >= T).mean() * 100
print(f"joined {len(m):,} node observations | {m.reach_id.nunique()} reaches | "
      f"high-error base {base:.1f}%")
m.to_parquet(f"{RC}/node_coherence_all.parquet", index=False)

print("\n[1] coherence against node error, all reaches")
rho(m.coh_median, m.e, "normalized coherence")
rho(m.coh_p25, m.e, "coherence lower quartile")
rho(m.lowcoh_pct, m.e, "low-coherence class share (shipped)")
rho(m.layover_impact_median, m.e, "layover_impact (shipped)")
rho(m.bld, m.e, "far building height (ours)")
rho(m.isa, m.e, "far ISA (ours)")
rho(m.ang, m.e, "angle deviation (ours)")

print("\n[2] the same, by structure and bridge proximity")
rows = []
for lab, g in [("all", m),
               ("structure > 1 km", m[m.nearest_km > 1]),
               ("structure <= 1 km", m[m.nearest_km <= 1]),
               ("bridge > 100 m", m[m.bdist_m > 100]),
               ("bridge <= 100 m", m[m.bdist_m <= 100]),
               ("clean (no structure, no bridge)",
                m[(m.nearest_km > 1) & (m.bdist_m > 100)])]:
    if len(g) < 200:
        continue
    r1 = rho(g.coh_median, g.e, f"  coherence, {lab}")
    r2 = rho(g.layover_impact_median, g.e, f"  layover_impact, {lab}")
    rows.append({"stratum": lab, "n": len(g), "reaches": g.reach_id.nunique(),
                 "rho_coherence": r1, "rho_layover_impact": r2})
pd.DataFrame(rows).to_csv(f"{OUT}/coherence_strata_all.csv", index=False)

print("\n[3] fitted predictor sets, five-fold grouped by reach")
SETS = {
    "angle + ISA + building (ours)": ["ang_par", "isa", "bld"],
    "coherence alone": ["coh_median"],
    "low-coh class alone": ["lowcoh_pct"],
    "layover_impact alone (node-aggregated)": ["layover_impact_median"],
    "coherence + ours": ["coh_median", "ang_par", "isa", "bld"],
    "coherence + ours + class": ["coh_median", "ang_par", "isa", "bld", "lowcoh_pct"],
}
y = m.e.values
yr = pd.Series(y).rank().values
groups = m.reach_id.values
gkf = GroupKFold(n_splits=5)
res = []
for name, cols in SETS.items():
    X = m[cols]
    Xr = X.rank().values
    pl = np.full(len(m), np.nan)
    pg = np.full(len(m), np.nan)
    for tr, te in gkf.split(X, y, groups):
        pl[te] = LinearRegression().fit(Xr[tr], yr[tr]).predict(Xr[te])
        pg[te] = HistGradientBoostingRegressor(max_depth=3, max_iter=200,
                                               learning_rate=0.08, random_state=0
                                               ).fit(X.values[tr], yr[tr]).predict(X.values[te])
    top = pg >= np.nanpercentile(pg, 90)
    share = (m.e[top] >= T).mean() * 100
    lo, hi = proportion_confint(int((m.e[top] >= T).sum()), int(top.sum()), method="wilson")
    res.append({"predictors": name, "rank_rho": st.spearmanr(pl, y).statistic,
                "boosted_rho": st.spearmanr(pg, y).statistic,
                "top_decile_pct": share, "ci_lo": 100 * lo, "ci_hi": 100 * hi})
    print(f"  {name:32s} rank {res[-1]['rank_rho']:+.3f} | "
          f"boosted {res[-1]['boosted_rho']:+.3f} | top decile {share:5.1f}%")
pd.DataFrame(res).to_csv(f"{OUT}/predictor_combinations_all.csv", index=False)

print("\n[4] cross-track consistency, all reaches")
for lo_, hi_ in [(10, 20), (20, 30), (30, 40), (40, 50), (50, 60)]:
    g = m[(m.xtrk_km >= lo_) & (m.xtrk_km < hi_)]
    if len(g) < 200:
        continue
    print(f"  {lo_:>2}-{hi_:<2} km: coherence {g.coh_median.median():.3f} | "
          f"low-coh class {g.lowcoh_pct.median():5.1f}% | "
          f"high-error {100*(g.e >= T).mean():5.1f}% | n={len(g):,}")
print(f"\nsaved {RC}/node_coherence_all.parquet, {OUT}/coherence_strata_all.csv, "
      f"{OUT}/predictor_combinations_all.csv")
