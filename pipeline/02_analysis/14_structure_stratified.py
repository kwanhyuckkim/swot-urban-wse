#!/usr/bin/env python
"""
Two closures for the structure question (R3-G1 follow-up).

Part 1  Stratify the paper's geometric results by structure proximity: reaches with
        a catalogued structure within 1 km of the reach boundary (n=102) against the
        rest. If boundary locks and dams mattered, the zone hierarchy, the ISA
        gradient, the angle effect, and the mask performance would differ between
        the groups.

Part 2  Bridges. The reviewer's list includes bridges, which no river inventory
        carries. The FHWA National Bridge Inventory does: pull the 2024 delimited
        files, keep bridges over waterways, convert the DMS coordinates, and test
        whether gauge-to-bridge distance organizes the reach error.
"""
import io
import time
import numpy as np, pandas as pd
import pyarrow.parquet as pq
import requests
from scipy import stats as st
from scipy.spatial import cKDTree

PROJ = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results"
WORK = "/work/pi_kandread_umass_edu/swot-urban"
OUT = f"{PROJ}/round2"
NODES = f"{PROJ}/swot_urban_nearfar_v6.5_results.parquet"
OBS = f"{WORK}/2_urban_reaches/all_final_shape.parquet"
NETP = f"{PROJ}/reviewer_checks/obstruction_network_distance.parquet"
ANGLE, HEIGHT, ISA = 13.0, 8.0, 19.0


def p68(x):
    return float(np.nanpercentile(x, 68)) if len(x) else np.nan


# Part 1: stratified geometry
def part1():
    d = pq.read_table(NODES, columns=["reach_id", "flow_angle", "far_isa", "far_building",
                                      "error_node", "sample_geometry_category"]).to_pandas()
    d["reach_id"] = d.reach_id.astype("int64")
    d["e"] = d.error_node.abs()
    fa = d.flow_angle % 360
    d["angle_dev"] = np.minimum.reduce([fa, (fa - 180).abs(), 360 - fa])
    d["isa_pct"] = (d.far_isa * 100 if d.far_isa.max() <= 1 else d.far_isa).fillna(0)
    d["bld"] = d.far_building.fillna(0)
    d = d.dropna(subset=["e"])
    net = pd.read_parquet(NETP, columns=["reach_id", "nearest_km"])
    net["reach_id"] = net.reach_id.astype("int64")
    d = d.merge(net, on="reach_id", how="left")
    d["grp"] = np.where(d.nearest_km.fillna(1e9) <= 1.0, "near(<=1km)", "far(>1km)")
    zone = d.sample_geometry_category.astype(str).str.lower()

    rows = []
    for g, sub in d.groupby("grp"):
        par = sub[sub.angle_dev <= ANGLE]
        z = zone.loc[sub.index]
        zp = zone.loc[par.index]
        r = {"group": g, "reaches": sub.reach_id.nunique(), "n_obs": len(sub),
             "n_parallel": len(par)}
        p10 = sub[sub.angle_dev <= 10]
        z10 = zone.loc[p10.index]
        for zz, name in [("na", "p68_par_neither"), ("near", "p68_par_near"),
                         ("far", "p68_par_far"), ("near/far", "p68_par_both")]:
            r[name] = p68(p10[(z10 == zz).values].e)
        perp = sub[sub.angle_dev >= 80]
        zperp = zone.loc[perp.index]
        r["p68_perp_both"] = p68(perp[(zperp == "near/far").values].e)
        r["angle_ratio_both"] = (r["p68_par_both"] / r["p68_perp_both"]
                                 if r["p68_perp_both"] else np.nan)
        pi = par[par.isa_pct > 0]
        r["rho_farISA_par"] = st.spearmanr(pi.isa_pct, pi.e).statistic if len(pi) > 500 else np.nan
        lo = par[par.bld.between(0.1, 5)]
        hi = par[par.bld > 9]
        r["rho_ISA_lowbld"] = st.spearmanr(lo.isa_pct, lo.e).statistic if len(lo) > 300 else np.nan
        r["rho_ISA_highbld"] = st.spearmanr(hi.isa_pct, hi.e).statistic if len(hi) > 300 else np.nan
        flag = (par.bld > HEIGHT) | (par.isa_pct > ISA)
        base = p68(par.e)
        r["mask_reduction_pct"] = (base - p68(par[~flag].e)) / base * 100
        r["mask_flagged_pct"] = 100 * flag.mean()
        f, u = par[flag], par[~flag]
        r["response_p68_ratio"] = p68(f.e) / p68(u.e) if len(f) > 200 else np.nan
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT}/structure_stratified.csv", index=False)
    print("PART 1: geometry stratified by structure proximity")
    print(out.round(3).to_string(index=False))
    return out


# Part 2: bridges (NBI)
STATES = ("AL AR AZ CA CO CT DC DE FL GA IA ID IL IN KS KY LA MA MD ME MI MN MO MS "
          "MT NC ND NE NH NJ NM NV NY OH OK OR PA RI SC SD TN TX UT VA VT WA WI WV WY").split()


def dms_lat(v):
    v = pd.to_numeric(v, errors="coerce")
    dd = v // 1_000_000
    mm = (v // 10_000) % 100
    ss = (v % 10_000) / 100.0
    out = dd + mm / 60 + ss / 3600
    return out.where((out > 17) & (out < 72))


def dms_lon(v):
    v = pd.to_numeric(v, errors="coerce")
    dd = v // 1_000_000
    mm = (v // 10_000) % 100
    ss = (v % 10_000) / 100.0
    out = -(dd + mm / 60 + ss / 3600)
    return out.where((out > -180) & (out < -60))


def fetch_nbi():
    frames = []
    for stx in STATES:
        url = f"https://www.fhwa.dot.gov/bridge/nbi/2024/delimited/{stx}24.txt"
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=240)
                if r.status_code == 200 and "STRUCTURE_NUMBER" in r.text[:2000]:
                    df = pd.read_csv(io.StringIO(r.text), dtype=str, low_memory=False)
                    frames.append(df[["LAT_016", "LONG_017", "SERVICE_UND_042B"]])
                    break
            except Exception:
                time.sleep(2 ** attempt)
        else:
            print(f"  NBI fetch failed: {stx}")
    nbi = pd.concat(frames, ignore_index=True)
    nbi["lat"] = dms_lat(nbi.LAT_016)
    nbi["lon"] = dms_lon(nbi.LONG_017)
    water = nbi.SERVICE_UND_042B.isin(["5", "6", "7", "8"])
    nbi = nbi[water].dropna(subset=["lat", "lon"])
    return nbi


def gauge_coords():
    obs = pq.read_table(OBS, columns=["reach_id", "site_no", "wse_relative_m",
                                      "wse_usgs_relative_m"]).to_pandas()
    obs["reach_id"] = obs.reach_id.astype("int64")
    obs["e"] = (obs.wse_relative_m - obs.wse_usgs_relative_m).abs()
    err = obs.groupby(["reach_id", "site_no"]).e.median().rename("med_err").reset_index()
    sites = sorted(obs.site_no.dropna().astype(str).unique())
    recs = []
    for i in range(0, len(sites), 100):
        batch = ",".join(sites[i:i + 100])
        r = requests.get("https://waterservices.usgs.gov/nwis/site/",
                         params={"format": "rdb", "sites": batch,
                                 "siteOutput": "expanded"}, timeout=60)
        hdr = None
        for line in r.text.splitlines():
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if hdr is None:
                if "site_no" in cols:
                    hdr = {c: k for k, c in enumerate(cols)}
                continue
            if line.startswith("5s") or len(cols) < len(hdr):
                continue
            try:
                recs.append({"site_no": cols[hdr["site_no"]],
                             "glat": float(cols[hdr["dec_lat_va"]]),
                             "glon": float(cols[hdr["dec_long_va"]])})
            except Exception:
                pass
    g = pd.DataFrame(recs).drop_duplicates("site_no")
    return err.merge(g, on="site_no", how="inner")


def part2():
    print("PART 2: bridges (FHWA NBI 2024, waterway-crossing structures)")
    nbi = fetch_nbi()
    print(f"  NBI bridges over waterways: {len(nbi):,}")
    g = gauge_coords()
    print(f"  gauges with coordinates: {g.site_no.nunique()} (of study pairs {len(g)})")
    lat0 = np.radians(g.glat.mean())
    kx = 111.32 * np.cos(lat0)
    tree = cKDTree(np.column_stack([nbi.lon * kx, nbi.lat * 111.32]))
    dist_km, _ = tree.query(np.column_stack([g.glon * kx, g.glat * 111.32]))
    g["bridge_dist_m"] = dist_km * 1000
    g.to_csv(f"{OUT}/bridge_check.csv", index=False)
    r, p = st.spearmanr(g.bridge_dist_m, g.med_err)
    for cut in (100, 250, 500):
        a = g[g.bridge_dist_m <= cut].med_err
        b = g[g.bridge_dist_m > cut].med_err
        if len(a) > 10 and len(b) > 10:
            mw = st.mannwhitneyu(a, b)
            print(f"  <= {cut:>3} m: n={len(a):3d} median {a.median():.3f} m | "
                  f"beyond: n={len(b):3d} median {b.median():.3f} m | MW p={mw.pvalue:.2f}")
    print(f"  Spearman(bridge distance, reach median error) = {r:+.3f} (p={p:.2f}, n={len(g)})")
    print(f"  gauges within 100 m of a bridge: {100 * (g.bridge_dist_m <= 100).mean():.0f}%  "
          f"median distance {g.bridge_dist_m.median():.0f} m")
    print(f"  saved {OUT}/bridge_check.csv")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "part2":
        part2()
    else:
        part1()
        part2()
