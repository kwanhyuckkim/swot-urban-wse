"""2026-10-05: proposed V.C Table III on the manuscript population, COMPLETE CASES ONLY, (node table, 543,045 rows), non-overlapping
far-nadir ISA bins. Columns: n, reaches, P68 of |error_node|, P68 of li_w (= liabs_wmean_A from coh_matrix2:
inverse-variance weighted node mean of |layover_impact| over PIXC classes 3-7, as for_brent/deck_0924/prep_strata.py),
P68 of |layovr_val|. Parallel zone = angle deviation < 13 deg. Also the >= 60 % bin with and without node_q = 3."""
import glob
import numpy as np, pandas as pd, pyarrow.parquet as pq
P = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results"; RC = f"{P}/reviewer_checks"; FILL, T = -1e11, 0.1832
hour = lambda s: pd.to_datetime(s, errors="coerce", utc=True).dt.floor("h")
d = pq.read_table(f"{P}/swot_urban_nearfar_v6.5_results.parquet", columns=["reach_id", "node_id", "time_str", "flow_angle", "far_isa", "error_node"]).to_pandas()
d["e"] = d.error_node.abs(); fa = d.flow_angle % 360; d["ang"] = np.minimum.reduce([fa, (fa - 180).abs(), 360 - fa])
d["isa"] = d.far_isa * 100 if d.far_isa.max() <= 1 else d.far_isa
d["node_id"] = d.node_id.astype("int64"); d["t"] = hour(d.time_str)
sch = pq.read_schema(f"{RC}/layovr_val_node.parquet").names
cols = ["node_id", "time_str", "layovr_val", "node_q_b"] + (["node_q"] if "node_q" in sch else [])
n = pd.read_parquet(f"{RC}/layovr_val_node.parquet", columns=cols)
n["node_id"] = n.node_id.astype("int64"); n["layovr_val"] = n.layovr_val.where(n.layovr_val > FILL); n["t"] = hour(n.time_str)
n = n.dropna(subset=["t"]).drop_duplicates(["node_id", "t"]).drop(columns=["time_str"])
m2 = pd.concat([pd.read_parquet(f, columns=["node_id", "t", "liabs_wmean_A"]) for f in sorted(glob.glob(f"{RC}/coh_matrix2/coh_shard*.parquet"))], ignore_index=True)
tt = pd.to_datetime(m2.t.astype(str), format="%Y%m%dT%H%M%S", errors="coerce")
m2["t"] = (tt if tt.notna().mean() > 0.5 else pd.to_datetime(m2.t, errors="coerce")).dt.floor("h")
m2["t"] = m2.t.dt.tz_localize("UTC") if m2.t.dt.tz is None else m2.t.dt.tz_convert("UTC")
m2 = m2[m2.node_id > 0].drop_duplicates(["node_id", "t"]); m2["node_id"] = m2.node_id.astype("int64")
x = d.merge(n, on=["node_id", "t"], how="left").merge(m2, on=["node_id", "t"], how="left")
x["L"] = x.layovr_val.abs(); x = x.rename(columns={"liabs_wmean_A": "li_w"})
x = x[x.e.notna() & x.li_w.notna() & x.L.notna()].copy()  # complete cases: all three quantities (V.C Table III footnote)
print(f"rows {len(x):,}; layovr_val {x.L.notna().sum():,}; li_w {x.li_w.notna().sum():,}; node_q col: {'node_q' in x}")
print(f"angle exactly 13: {(x.ang == 13).sum()} | parallel <13 {(x.ang < 13).sum():,} | <=13 {(x.ang <= 13).sum():,}")
par = x.ang < 13
p68 = lambda s: float(np.nanpercentile(s.dropna(), 68)) if s.notna().any() else np.nan
groups = [("All", np.ones(len(x), bool)), ("Parallel zone", par),
          ("Parallel, far-nadir ISA <= 5%", par & (x.isa <= 5)), ("Parallel, 5% < ISA < 10%", par & (x.isa > 5) & (x.isa < 10)),
          ("Parallel, 10% <= ISA < 30%", par & (x.isa >= 10) & (x.isa < 30)), ("Parallel, 30% <= ISA < 60%", par & (x.isa >= 30) & (x.isa < 60)),
          ("Parallel, ISA >= 60%", par & (x.isa >= 60)),
          ("(nested) Parallel, ISA >= 10%", par & (x.isa >= 10)), ("(nested) Parallel, ISA >= 30%", par & (x.isa >= 30))]
print(f"\n{'group':34s}{'n':>9}{'reach':>6}{'err p68':>9}{'>T %':>7}{'li_w p68':>10}{'n li_w':>9}{'|lv| p68':>10}{'n lv':>9}")
for lab, g in groups:
    s = x[g]
    print(f"{lab:34s}{len(s):>9,}{s.reach_id.nunique():>6}{p68(s.e):>9.3f}{100*(s.e>=T).mean():>7.1f}{p68(s.li_w):>10.3f}{s.li_w.notna().sum():>9,}{p68(s.L):>10.3f}{s.L.notna().sum():>9,}")
g60 = x[par & (x.isa >= 60)]
bad = (g60.node_q == 3) if "node_q" in g60 else ((g60.node_q_b.fillna(0).astype("int64") & 8388608) > 0)
print(f"\n>=60%: bad (node_q=3{' ' if 'node_q' in g60 else ' via wse_outlier bit'}) {int(bad.sum())}/{len(g60)} = {100*bad.mean():.1f}%; "
      f"err p68 all {p68(g60.e):.3f}, without bad {p68(g60.e[~bad]):.3f}, bad only {p68(g60.e[bad]):.3f}")
