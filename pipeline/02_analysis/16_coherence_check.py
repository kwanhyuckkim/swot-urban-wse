#!/usr/bin/env python
"""
Normalized coherence, the check Brent asked for (2026-06-17 meeting).

He made three points we had not tested. First, we had looked at coherent power, not
at the normalized coherence, and the two behave differently because coherent power
rises with brightness. Second, the layover signature he described is a target that is
brighter than it should be while its coherence is lower than that brightness implies.
Third, the thresholds behind the low-coherence classes were never optimized and need
not hold across the swath, so there should be places where layover corrupts the height
without the coherence dropping.

The pixel cloud ships the pieces. Its coherent_power combines the two channels after
co-aligning their phases, which the product description caps at a 3 dB gain over the
incoherent combination, so the ratio below runs from 1 to 2 and the coherence magnitude
is that ratio minus one:

    gain  = coherent_power / sqrt(power_plus_y * power_minus_y)
    gamma = gain - 1

Each array task handles one site. Per pass we write the coherence distribution, the
bright-but-incoherent share, the low-coherence class rates, and the same quantities
split by cross-track distance, then join everything to the gauge error.

  python 16_coherence_check.py extract --site-index 0
  python 16_coherence_check.py merge
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

WORK = "/work/pi_kandread_umass_edu/swot-urban"
PROJ = "/project/pi_kandread_umass_edu/swot-urban"
ERROR_TABLE = f"{WORK}/2_urban_reaches/all_final_shape.parquet"
SITES_SHP = f"{WORK}/2_urban_reaches/selected_12_vD.shp"
OUT_DIR = f"{PROJ}/3_analysis_results/reviewer_checks"
T = 0.1832
LAYOVER_SIGNIFICANT_BIT = 1          # geolocation_qual bit 0
BRIGHT_LAND_CLASS_BIT = 8            # classification_qual bit 3
LOW_COH_CLASSES = (6, 7)             # low_coh_water_near_land, low_coh_water
XT_BINS = [10, 20, 30, 40, 50, 60]   # km, the valid range used in the paper


def best_versions(files):
    keep = {}
    for f in files:
        m = re.match(r"(.+)_(P[IG]D\d)_(\d+)\.nc", os.path.basename(f))
        if not m:
            continue
        base, proc, cnt = m.group(1), m.group(2), int(m.group(3))
        if base not in keep:
            keep[base] = (f, proc, cnt)
        else:
            _, op, oc = keep[base]
            if (proc == "PGD0" and op == "PID0") or (proc == op and cnt > oc):
                keep[base] = (f, proc, cnt)
    return [v[0] for v in keep.values()]


def extract_pass(pixc_file, poly):
    """Coherence and brightness for the water pixels inside one reach polygon."""
    import xarray as xr
    import shapely

    ds = xr.open_dataset(pixc_file, group="pixel_cloud")
    lon, lat = ds["longitude"].values, ds["latitude"].values
    b = poly.bounds
    bbox = (lon >= b[0]) & (lon <= b[2]) & (lat >= b[1]) & (lat <= b[3])
    if bbox.sum() == 0:
        ds.close(); return None
    cls = ds["classification"].values[bbox]
    idx = np.where(bbox)[0][cls > 2]
    if idx.size == 0:
        ds.close(); return None
    inside = shapely.contains_xy(poly, lon[idx], lat[idx])
    idx = idx[inside]
    if idx.size < 50:
        ds.close(); return None

    def g(name):
        return ds[name].values[idx].astype("float64") if name in ds \
            else np.full(idx.size, np.nan)

    cp = g("coherent_power")
    pp, pm = g("power_plus_y"), g("power_minus_y")
    denom = np.sqrt(np.clip(pp, 0, None) * np.clip(pm, 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma = np.where(denom > 0, cp / denom, np.nan)
    gamma = np.clip(gamma, 0, 2) - 1.0     # coherent gain to coherence magnitude
    sig0 = g("sig0")
    cq = g("classification_qual")
    gq = g("geolocation_qual")
    li = g("layover_impact")
    xt = np.abs(g("cross_track")) / 1000.0
    cl = ds["classification"].values[idx]
    ds.close()

    fin = np.isfinite(gamma) & np.isfinite(sig0)
    if fin.sum() < 50:
        return None
    gm, sg = gamma[fin], sig0[fin]

    # Brent's signature: brighter than typical, less coherent than that brightness implies.
    # Regress coherence on log brightness, then take the negative residual tail.
    ls = np.log10(np.clip(sg, 1e-3, None))
    ok = np.isfinite(ls)
    resid = np.full(gm.shape, np.nan)
    if ok.sum() > 30 and np.nanstd(ls[ok]) > 1e-6:
        slope, icept = np.polyfit(ls[ok], gm[ok], 1)
        resid[ok] = gm[ok] - (slope * ls[ok] + icept)
    bright = sg >= np.nanpercentile(sg, 75)
    incoh = resid <= np.nanpercentile(resid[np.isfinite(resid)], 25) \
        if np.isfinite(resid).sum() > 30 else np.zeros(gm.shape, bool)

    lay = np.where(np.isfinite(gq), (gq.astype("int64") & LAYOVER_SIGNIFICANT_BIT) > 0, False)
    lowcoh = np.isin(cl, LOW_COH_CLASSES)
    rec = {
        "n_pixels": int(idx.size),
        "gamma_median": float(np.nanmedian(gm)),
        "gamma_p25": float(np.nanpercentile(gm, 25)),
        "gamma_p10": float(np.nanpercentile(gm, 10)),
        "coh_power_median": float(np.nanmedian(cp)),
        "sig0_median": float(np.nanmedian(sig0)),
        "bright_incoherent_pct": float(100 * np.nanmean(bright & incoh)),
        "lowcoh_class_pct": float(100 * lowcoh.mean()),
        "layover_flag_pct": float(100 * lay.mean()),
        # Brent's third point: layover flagged, coherence did not drop
        "layover_but_coherent_pct": float(
            100 * np.mean(lay[fin] & (gm >= np.nanmedian(gm)))),
        "lowcoh_missed_pct": float(
            100 * np.mean((~lowcoh[fin]) & (gm <= np.nanpercentile(gm, 10)))),
        "layover_impact_median_m": float(np.nanmedian(li)),
        "xtrk_km_median": float(np.nanmedian(xt)),
    }
    for lo, hi in zip(XT_BINS[:-1], XT_BINS[1:]):
        m = (xt >= lo) & (xt < hi)
        rec[f"gamma_xt{lo}_{hi}"] = float(np.nanmedian(gamma[m])) if m.sum() > 30 else np.nan
        rec[f"lowcoh_xt{lo}_{hi}"] = float(100 * lowcoh[m].mean()) if m.sum() > 30 else np.nan
    return rec


def run_extract(site_index):
    import geopandas as gpd
    import pyarrow.parquet as pq

    dirs = sorted(glob.glob(f"{PROJ}/0_data_collection/SWOT_PIXC/fctimp_*"))
    if site_index >= len(dirs):
        print(f"site index {site_index} beyond {len(dirs)} sites, nothing to do")
        return
    d = dirs[site_index]
    isa = os.path.basename(d).replace("fctimp_", "")
    sites = gpd.read_file(SITES_SHP)
    col = "fctimp_mea" if "fctimp_mea" in sites.columns else "fctimp_mean"
    poly_gdf = sites[np.isclose(sites[col], float(isa), atol=0.01)].to_crs("EPSG:4326")
    sample = f"{WORK}/1_data_processing/fctimp_mean_{isa}_sample.parquet"
    if poly_gdf.empty or not os.path.exists(sample):
        print(f"site {isa}: no polygon or sample table")
        return
    poly = poly_gdf.geometry.iloc[0]

    obs = pd.read_parquet(sample)
    obs["t"] = pd.to_datetime(obs.time_str).dt.tz_localize(None)
    obs["error"] = obs.wse_relative_m - obs.wse_usgs_relative_m
    files = best_versions(glob.glob(f"{d}/SWOT_L2_HR_PIXC_*.nc"))
    print(f"site {isa}: {len(obs)} observations, {len(files)} granules", flush=True)

    rows = []
    for _, o in obs.iterrows():
        match = None
        for f in files:
            parts = os.path.basename(f).split("_")
            try:
                start = pd.to_datetime(parts[7], format="%Y%m%dT%H%M%S")
            except Exception:
                continue
            if abs((start - o.t).total_seconds()) <= 300:
                match = f
                break
        if match is None:
            continue
        try:
            rec = extract_pass(match, poly)
        except Exception as ex:
            print(f"  {os.path.basename(match)}: {type(ex).__name__}: {ex}", flush=True)
            continue
        if rec is None:
            continue
        rec.update({"isa_site": float(isa) * 100, "reach_id": int(o.reach_id),
                    "time_str": str(o.time_str), "error_m": float(o.error),
                    "abs_error_m": abs(float(o.error)),
                    "high_error": bool(abs(float(o.error)) >= T)})
        rows.append(rec)
        if len(rows) % 5 == 0:
            print(f"  {len(rows)} passes done", flush=True)
    if not rows:
        print("no passes extracted")
        return
    out = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    p = f"{OUT_DIR}/coherence_site{site_index:02d}.parquet"
    out.to_parquet(p, index=False)
    print(f"site {isa}: wrote {len(out)} passes -> {p}")


def run_merge():
    from scipy import stats as st
    files = sorted(glob.glob(f"{OUT_DIR}/coherence_site*.parquet"))
    if not files:
        raise SystemExit("no shards")
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    d.to_parquet(f"{OUT_DIR}/coherence_check.parquet", index=False)
    print(f"merged {len(d)} passes from {len(files)} sites\n")

    def rho(a, b, lab):
        m = pd.notna(a) & pd.notna(b)
        if m.sum() < 12:
            print(f"  {lab:44s} n<12"); return
        r, p = st.spearmanr(a[m], b[m])
        print(f"  {lab:44s} rho {r:+.3f} (p={p:.3f}, n={m.sum()})")

    print("[1] Brent: normalized coherence vs coherent power against gauge error")
    rho(d.gamma_median, d.abs_error_m, "median normalized coherence")
    rho(d.gamma_p10, d.abs_error_m, "10th-pct coherence (tail)")
    rho(d.coh_power_median, d.abs_error_m, "median coherent power (what we had)")
    rho(d.lowcoh_class_pct, d.abs_error_m, "low-coherence class share (shipped flag)")
    rho(d.layover_impact_median_m, d.abs_error_m, "layover_impact (reference)")

    print("\n[2] Brent's layover signature: bright but less coherent than brightness implies")
    rho(d.bright_incoherent_pct, d.abs_error_m, "bright-and-incoherent pixel share")
    rho(d.bright_incoherent_pct, d.isa_site, "  same, against reach ISA")
    hi, lo = d[d.high_error], d[~d.high_error]
    if len(hi) > 5 and len(lo) > 5:
        mw = st.mannwhitneyu(hi.bright_incoherent_pct, lo.bright_incoherent_pct)
        print(f"  high-error passes {hi.bright_incoherent_pct.median():.1f}% vs "
              f"low-error {lo.bright_incoherent_pct.median():.1f}%  (MW p={mw.pvalue:.3f})")

    print("\n[3] Brent: layover flagged where coherence did not drop, and the reverse")
    print(f"  layover-flagged but coherent, per pass median: "
          f"{d.layover_but_coherent_pct.median():.1f}%")
    print(f"  deeply incoherent but not in the low-coh class: "
          f"{d.lowcoh_missed_pct.median():.1f}%")
    rho(d.layover_but_coherent_pct, d.abs_error_m, "layover-but-coherent share vs error")

    print("\n[4] Brent: are the low-coherence thresholds consistent across the swath")
    for lo_, hi_ in zip(XT_BINS[:-1], XT_BINS[1:]):
        g, l = d.get(f"gamma_xt{lo_}_{hi_}"), d.get(f"lowcoh_xt{lo_}_{hi_}")
        if g is None:
            continue
        print(f"  cross-track {lo_:>2}-{hi_:<2} km: coherence {g.median():.3f} | "
              f"low-coh class {l.median():5.1f}% | n={g.notna().sum()}")
    cols = [f"lowcoh_xt{a}_{b}" for a, b in zip(XT_BINS[:-1], XT_BINS[1:])]
    med = d[cols].median()
    if med.notna().sum() > 2:
        print(f"  -> class rate spans {med.min():.1f}% to {med.max():.1f}% across the swath "
              f"({med.max() - med.min():.1f} points)")
    print(f"\nsaved {OUT_DIR}/coherence_check.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["extract", "merge"])
    ap.add_argument("--site-index", type=int, default=0)
    a = ap.parse_args()
    if a.task == "extract":
        run_extract(a.site_index)
    else:
        run_merge()
