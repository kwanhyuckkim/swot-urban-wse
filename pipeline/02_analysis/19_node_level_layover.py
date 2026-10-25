#!/usr/bin/env python
"""
The layover indicators judged at node scale, which is the scale this study works at.

Earlier passes evaluated layovr_val against reach-median error over 409 reaches. The
node table carries an error for every node observation, and the node pull carries
layovr_val and node_q_b for the same nodes, so the two join on node identity and
acquisition hour. That gives 541k paired observations instead of 409, and it lets the
node quality bits be checked directly rather than inferred from their reach counterpart.

Sections
  1  join and coverage
  2  layovr_val against node error, overall and by stratum
  3  node_q_b bits, including the bright-land bit we have only checked at reach scale
  4  does layovr_val add anything once our geometry is in hand
"""
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from scipy import stats as st
from scipy.spatial import cKDTree

PROJ = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results"
RC = f"{PROJ}/reviewer_checks"
WORK = "/work/pi_kandread_umass_edu/swot-urban"
OUT = f"{PROJ}/round2"
NODES = f"{PROJ}/swot_urban_nearfar_v6.5_results.parquet"
FILL = -1e11
BITS = {"sig0_suspect": 1, "class_suspect": 2, "geo_suspect": 4, "water_frac": 8,
        "blocking_width": 16, "bright_land": 128, "few_sig0": 512, "few_area": 1024,
        "few_wse": 2048, "far_range": 8192, "near_range": 16384,
        "class_degraded": 262144, "geo_degraded": 524288, "lake": 4194304,
        "wse_outlier": 8388608, "wse_bad": 16777216}


def rho(a, b, lab, extra=""):
    m = pd.notna(a) & pd.notna(b)
    if m.sum() < 30:
        print(f"  {lab:52s} n<30"); return np.nan
    r, p = st.spearmanr(a[m], b[m])
    print(f"  {lab:52s} rho {r:+.4f} (p={p:.1e}, n={m.sum():,}) {extra}")
    return r


# 1. join
d = pq.read_table(NODES, columns=["reach_id", "node_id", "time_str", "lat", "lon",
                                  "flow_angle", "far_isa", "far_building",
                                  "error_node", "width_node"]).to_pandas()
d["e"] = d.error_node.abs()
fa = d.flow_angle % 360
d["ang"] = np.minimum.reduce([fa, (fa - 180).abs(), 360 - fa])
d["isa_pct"] = (d.far_isa * 100 if d.far_isa.max() <= 1 else d.far_isa).fillna(0)
d["bld"] = d.far_building.fillna(0)
d["t"] = pd.to_datetime(d.time_str, errors="coerce", utc=True).dt.floor("h")
d["node_id"] = d.node_id.astype("int64")
d = d.dropna(subset=["e", "t"])

n = pd.read_parquet(f"{RC}/layovr_val_node.parquet",
                    columns=["node_id", "time_str", "layovr_val", "node_q_b",
                             "wse_u", "wse_r_u"])
n["node_id"] = n.node_id.astype("int64")
for c in ("layovr_val", "wse_u", "wse_r_u"):
    n[c] = n[c].where(n[c] > FILL)
n["t"] = pd.to_datetime(n.time_str, errors="coerce", utc=True).dt.floor("h")
n = n.dropna(subset=["t"]).drop_duplicates(["node_id", "t"])

m = d.merge(n.drop(columns=["time_str"]), on=["node_id", "t"], how="inner")
m["sys_u"] = np.sqrt(np.maximum(m.wse_u ** 2 - m.wse_r_u ** 2, 0))
print(f"[1] joined {len(m):,} node observations "
      f"({100*len(m)/len(d):.1f}% of the node table), "
      f"{m.reach_id.nunique()} reaches, layovr_val valid {m.layovr_val.notna().sum():,}")

# node-level bridge distance
try:
    b = pd.read_parquet(f"{OUT}/nbi_waterway_bridges.parquet")
    lat0 = np.radians(m.lat.mean())
    kx, ky = 111.32 * np.cos(lat0), 111.32
    tree = cKDTree(np.column_stack([b.lon * kx, b.lat * ky]))
    m["bdist_m"] = tree.query(np.column_stack([m.lon * kx, m.lat * ky]))[0] * 1000
except Exception as ex:
    print("  bridge distance unavailable:", ex)
    m["bdist_m"] = np.nan
net = pd.read_parquet(f"{RC}/obstruction_network_distance.parquet",
                      columns=["reach_id", "nearest_km"])
net["reach_id"] = net.reach_id.astype("int64")
m = m.merge(net, on="reach_id", how="left")
m["nearest_km"] = m.nearest_km.fillna(1e9)

# 2. layovr_val against node error
print("\n[2] node layovr_val against node |error|")
rho(m.layovr_val, m.e, "all node observations")
rho(m.layovr_val, m.sys_u, "  against the reported systematic uncertainty")
rows = []
strata = [
    ("structure > 1 km", m[m.nearest_km > 1]),
    ("structure <= 1 km", m[m.nearest_km <= 1]),
    ("bridge > 100 m", m[m.bdist_m > 100]),
    ("bridge <= 100 m", m[m.bdist_m <= 100]),
    ("clean (no structure, no bridge)", m[(m.nearest_km > 1) & (m.bdist_m > 100)]),
    ("parallel (angle <= 13)", m[m.ang <= 13]),
    ("perpendicular (angle >= 77)", m[m.ang >= 77]),
    ("far ISA >= 30%", m[m.isa_pct >= 30]),
    ("far ISA < 5%", m[m.isa_pct < 5]),
    ("far building > 8 m", m[m.bld > 8]),
]
for lab, g in strata:
    r = rho(g.layovr_val, g.e, f"  {lab}")
    rows.append({"stratum": lab, "n": len(g), "rho": r})

# 3. node quality bits
print("\n[3] node_q_b bits at node scale, and whether they mark the bad nodes")
q = m.node_q_b.fillna(0).astype("int64")
T = 0.1832
base = (m.e >= T).mean() * 100
print(f"  node high-error base rate ({m.e.notna().sum():,} obs): {base:.1f}%")
brows = []
for name, bit in BITS.items():
    f = (q & bit) > 0
    nfire = int(f.sum())
    share = (m.e[f] >= T).mean() * 100 if nfire > 30 else np.nan
    brows.append({"bit": name, "n_fired": nfire, "coverage_pct": 100 * nfire / len(m),
                  "high_error_pct": share, "lift": share / base if nfire > 30 else np.nan})
    flag = "" if nfire else "   <- never fires"
    print(f"  {name:16s} fired {nfire:8,} ({100*nfire/len(m):5.2f}%)  "
          f"high-error {('%5.1f%%' % share) if nfire > 30 else '   -  '}"
          f"  lift {('%.2f' % (share/base)) if nfire > 30 else ' -  '}{flag}")
pd.DataFrame(brows).to_csv(f"{OUT}/node_qb_bits.csv", index=False)

print("\n  bright-land bit, node against reach")
bl_node = int(((q & 128) > 0).sum())
ob = pq.read_table(f"{WORK}/2_urban_reaches/all_final_shape.parquet",
                   columns=["reach_id", "reach_q_b"]).to_pandas()
bl_reach = int(((ob.reach_q_b.astype("int64") & 128) > 0).sum())
print(f"    node_q_b  bit 7 set in {bl_node:,} of {len(m):,} node observations")
print(f"    reach_q_b bit 7 set in {bl_reach:,} of {len(ob):,} reach observations")

# 4. incremental value over our geometry
print("\n[4] does layovr_val add anything once the geometry is known")
sub = m[(m.ang <= 13) & m.layovr_val.notna()].copy()
if len(sub) > 1000:
    rk = lambda s: s.rank()
    resid_e = rk(sub.e) - np.polyval(np.polyfit(rk(sub.isa_pct), rk(sub.e), 1), rk(sub.isa_pct))
    resid_l = rk(sub.layovr_val) - np.polyval(
        np.polyfit(rk(sub.isa_pct), rk(sub.layovr_val), 1), rk(sub.isa_pct))
    r, p = st.pearsonr(resid_l, resid_e)
    print(f"  parallel zone, ISA partialled out: rho {r:+.4f} (p={p:.1e}, n={len(sub):,})")
    rho(sub.layovr_val, sub.e, "  parallel zone, raw")
    rho(sub.isa_pct, sub.e, "  far ISA, raw (our predictor, same rows)")
    rho(sub.bld, sub.e, "  far building height, raw (ours)")

pd.DataFrame(rows).to_csv(f"{OUT}/node_layover_strata.csv", index=False)
print(f"\nsaved {OUT}/node_layover_strata.csv, node_qb_bits.csv")
