#!/usr/bin/env python
"""
Does the shipped layover estimate fail because of hydraulic structures, or on its own?

Brent's question is whether layover_impact ranks the errors we see. Before answering
we separate the cases he flagged in the same meeting: gauges near weirs and dams, and
gauges on bridges. If the estimate were merely defeated by structure-driven errors it
would work once those cases are removed. Each stratum is therefore evaluated the same
way, at both scales that carry a layover number.

Layer 1  does the estimate rank the gauge error
Layer 2  does the estimate even track urbanization, against the bright-land flag
"""
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from scipy import stats as st

PROJ = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results"
RC = f"{PROJ}/reviewer_checks"
WORK = "/work/pi_kandread_umass_edu/swot-urban"
OUT = f"{PROJ}/round2"
OBS = f"{WORK}/2_urban_reaches/all_final_shape.parquet"
FILL = -1e11


def rho(a, b):
    m = pd.notna(a) & pd.notna(b)
    if m.sum() < 12:
        return np.nan, np.nan, int(m.sum())
    r, p = st.spearmanr(a[m], b[m])
    return r, p, int(m.sum())


def strata(df, structure_col, bridge_col):
    yield "all", df
    yield "structure > 1 km", df[df[structure_col] > 1]
    yield "structure <= 1 km", df[df[structure_col] <= 1]
    if bridge_col in df:
        yield "bridge > 100 m", df[df[bridge_col] > 100]
        yield "bridge <= 100 m", df[df[bridge_col] <= 100]
        yield "clean (no structure <1 km, no bridge <100 m)", df[
            (df[structure_col] > 1) & (df[bridge_col] > 100)]


# ───────────────────────── inputs shared by both scales
net = pd.read_parquet(f"{RC}/obstruction_network_distance.parquet",
                      columns=["reach_id", "nearest_km"])
net["reach_id"] = net.reach_id.astype("int64")
net["nearest_km"] = net.nearest_km.fillna(1e9)
br = pd.read_csv(f"{OUT}/bridge_check.csv", dtype={"site_no": str})
br = br.groupby("reach_id").bridge_dist_m.min().reset_index()
br["reach_id"] = br.reach_id.astype("int64")

rows = []

# ───────────────────────── scale 1: PIXC layover_impact, 166 gauge-verified passes
c = pd.read_parquet(f"{RC}/cross_level_pixc_vs_riversp.parquet")
c["reach_id"] = c.reach_id.astype("int64")
c["isa_pct"] = c.isa * 100 if c.isa.max() <= 1.5 else c.isa
c = c.merge(net, on="reach_id", how="left").merge(br, on="reach_id", how="left")
c["nearest_km"] = c.nearest_km.fillna(1e9)
c["bridge_dist_m"] = c.bridge_dist_m.fillna(1e9)
print("=" * 78)
print("SCALE 1  PIXC layover_impact  (per-pass medians, gauge-verified)")
print("=" * 78)
for name, g in strata(c, "nearest_km", "bridge_dist_m"):
    r1, p1, n1 = rho(g.layover_impact_median_m, g.abs_error_m)
    r2, p2, _ = rho(g.layover_impact_median_m, g.isa_pct)
    r3, p3, _ = rho(g.bright_land_fire_pct, g.isa_pct)
    rows.append({"scale": "PIXC layover_impact", "stratum": name, "n": n1,
                 "rho_vs_error": r1, "p_vs_error": p1,
                 "rho_vs_isa": r2, "bright_flag_rho_vs_isa": r3})
    print(f"  {name:46s} n={n1:4d} | vs error {r1:+.3f} (p={p1:.2f}) "
          f"| vs ISA {r2:+.3f} | bright flag vs ISA {r3:+.3f}")

# ───────────────────────── scale 2: RiverSP layovr_val, reach medians, 409 reaches
r_ = pd.read_parquet(f"{RC}/layovr_val_reach.parquet",
                     columns=["reach_id", "layovr_val", "wse_u", "wse_r_u"])
for col in ("layovr_val", "wse_u", "wse_r_u"):
    r_[col] = r_[col].where(r_[col] > FILL)
r_["reach_id"] = r_.reach_id.astype("int64")
lv = r_.groupby("reach_id").layovr_val.median().rename("layovr_val").reset_index()

ob = pq.read_table(OBS, columns=["reach_id", "wse_relative_m", "wse_usgs_relative_m",
                                 "fctimp_mean"]).to_pandas()
ob["reach_id"] = ob.reach_id.astype("int64")
ob["e"] = (ob.wse_relative_m - ob.wse_usgs_relative_m).abs()
er = ob.groupby("reach_id").agg(med_err=("e", "median"),
                                isa_pct=("fctimp_mean", "first")).reset_index()
d = er.merge(lv, on="reach_id", how="inner").merge(net, on="reach_id", how="left") \
      .merge(br, on="reach_id", how="left")
d["nearest_km"] = d.nearest_km.fillna(1e9)
d["bridge_dist_m"] = d.bridge_dist_m.fillna(1e9)
print()
print("=" * 78)
print("SCALE 2  RiverSP layovr_val  (reach medians, 409 reaches)")
print("=" * 78)
for name, g in strata(d, "nearest_km", "bridge_dist_m"):
    r1, p1, n1 = rho(g.layovr_val, g.med_err)
    r2, p2, _ = rho(g.layovr_val, g.isa_pct)
    rows.append({"scale": "RiverSP layovr_val", "stratum": name, "n": n1,
                 "rho_vs_error": r1, "p_vs_error": p1,
                 "rho_vs_isa": r2, "bright_flag_rho_vs_isa": np.nan})
    print(f"  {name:46s} n={n1:4d} | vs error {r1:+.3f} (p={p1:.2f}) "
          f"| vs ISA {r2:+.3f}")

out = pd.DataFrame(rows)
out.to_csv(f"{OUT}/layover_by_structure.csv", index=False)
print(f"\nsaved {OUT}/layover_by_structure.csv")

sig = out[(out.p_vs_error < 0.05) & out.p_vs_error.notna()]
print(f"\nstrata where the estimate ranks the error at p < 0.05: "
      f"{len(sig)} of {out.p_vs_error.notna().sum()}")
print(f"|rho vs error| range across strata: "
      f"{out.rho_vs_error.abs().min():.3f} to {out.rho_vs_error.abs().max():.3f}")
