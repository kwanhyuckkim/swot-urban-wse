#!/usr/bin/env python
"""
Reviewer-driven checks on the shipped SWOT products (TGRS-2026-02661 revision).

Two independent checks, each answering specific reviewer comments:

  sword    Reviewer 3 General 1 - do the study reaches carry catalogued in-channel
           obstructions? Validates the SWORD obstruction attributes against the full
           SWORD v17b NA database (reach and node level) so the all-zero result is
           shown to be a real property and not an unpopulated column.

  partial  Separates the urbanization signal from any genuine error signal in the
           pixel indicators, by partial correlation and within-site correlation.

  crosslevel  Does the pixel cloud hold information the river vector product
           discards? Tests whether excluding the pixels that the product itself
           flags moves the reach elevation toward the gauge.

  sweep    The same check across every site with PIXC granules, related to the
           gauge-validated error per pass.

  network  Reviewer 3 General 1 extended - how far along the river network, in
           reaches crossed and in kilometres, does one travel upstream or
           downstream from each study reach before meeting a catalogued
           obstruction, and does that distance explain the observed error.

  pixc     Reviewer 4 overarching, 12 and 16 - does the upstream pixel cloud already
           flag urban layover? Extracts the shipped per-pixel layover fields inside the
           study reaches and compares a high-error pass with a low-error pass.

Per-pixel fields used (SWOT L2_HR_PIXC Product Description D-56411 Rev C):
  bright_land_flag  p.27  pixel is not typically water but may be radar reflective
  layover_impact    p.28  estimate of the height error caused by layover (m)
  geolocation_qual  p.54  bit 0 (decimal 1) = layover_significant, set when the
                          estimated layover impact exceeds a threshold (p.66)
  classification_qual, sig0, inc, cross_track, water_frac, height, pixel_area
Satellite state comes from the tvp group (latitude, longitude, altitude,
velocity_heading), following the extraction pattern used elsewhere in this pipeline.

Usage:
  python 06_reviewer_product_checks.py sword
  python 06_reviewer_product_checks.py network
  python 06_reviewer_product_checks.py pixc  [--isa 0.06] [--cycle 039]
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

WORK = "/work/pi_kandread_umass_edu/swot-urban"
PROJ = "/project/pi_kandread_umass_edu/swot-urban"
SWORD_SHP = "/nas/cee-ice/data/SWORD/SWORD_v17b/shp"
REACH_TABLE = f"{WORK}/2_urban_reaches/all_reaches_vD.parquet"
ERROR_TABLE = f"{WORK}/2_urban_reaches/all_final_shape.parquet"
SITES_SHP = f"{WORK}/2_urban_reaches/selected_12_vD.shp"
OUT_DIR = f"{PROJ}/3_analysis_results/reviewer_checks"

# PIXC quality bit assignments, Product Description D-56411 Rev C, table on p.65
# with the condition definitions on p.66.
GEOLOC_QUAL_BITS = {
    "layover_significant": 1,        # bit 0: estimated layover impact exceeds a threshold
    "phase_noise_suspect": 2,        # bit 1
    "phase_unwrapping_suspect": 4,   # bit 2
}
CLASS_QUAL_BITS = {
    "detected_water_but_no_prior_water": 4,   # bit 2: water detected, prior water probability low
    "detected_water_but_bright_land": 8,      # bit 3: water detected, prior bright-land flag nonzero
    "water_false_detection_rate_suspect": 16, # bit 4
}
# bright_land_flag values, Product Description p.46
BRIGHT_LAND_VALUES = {0: "not_bright_land", 1: "bright_land", 2: "bright_land_or_water"}
LAYOVER_SIGNIFICANT_BIT = GEOLOC_QUAL_BITS["layover_significant"]
LAYOVER_IMPACT_THRESHOLD_M = 5.0  # ATBD discussion of a significant impact


# Reviewer 3 General 1: catalogued obstructions
def check_sword_obstructions(verbose=True):
    """Compare the study reaches against the full SWORD NA database."""
    import geopandas as gpd
    import pyarrow.parquet as pq

    cols = ["reach_id", "obstr_type", "grod_id", "hfalls_id", "p_dam_id"]
    study = pq.read_table(REACH_TABLE, columns=cols).to_pandas().drop_duplicates("reach_id")
    study["reach_id"] = study.reach_id.astype("int64")

    reach_files = sorted(glob.glob(f"{SWORD_SHP}/NA/*reaches*.shp"))
    node_files = sorted(glob.glob(f"{SWORD_SHP}/NA/*nodes*.shp"))
    keep = ["reach_id", "obstr_type", "grod_id", "hfalls_id"]
    sword = pd.concat([gpd.read_file(f)[keep] for f in reach_files], ignore_index=True)
    sword["reach_id"] = sword.reach_id.astype("int64")
    nodes = pd.concat(
        [gpd.read_file(f)[["reach_id", "node_id", "obstr_type"]] for f in node_files],
        ignore_index=True,
    )
    nodes["reach_id"] = nodes.reach_id.astype("int64")

    ours = set(study.reach_id)
    matched = sword[sword.reach_id.isin(ours)]
    our_nodes = nodes[nodes.reach_id.isin(ours)]
    base_zero = float((sword.obstr_type == 0).mean())

    res = {
        "n_study_reaches": int(len(study)),
        "study_obstr_counts": study.obstr_type.value_counts().to_dict(),
        "study_grod_gt0": int((study.grod_id > 0).sum()),
        "study_hfalls_gt0": int((study.hfalls_id > 0).sum()),
        "study_pdam_gt0": int((study.p_dam_id > 0).sum()),
        "sword_na_reaches": int(len(sword)),
        "sword_obstr_counts": sword.obstr_type.value_counts().sort_index().to_dict(),
        "sword_obstr_nonzero_pct": float((sword.obstr_type != 0).mean() * 100),
        "matched_in_sword": int(len(matched)),
        "matched_obstr_counts": matched.obstr_type.value_counts().to_dict(),
        "sword_zero_rate": base_zero,
        "p_all_zero_by_chance": float(base_zero ** len(study)),
        "study_nodes": int(len(our_nodes)),
        "study_node_obstr_counts": our_nodes.obstr_type.value_counts().to_dict(),
        "sword_node_nonzero_pct": float((nodes.obstr_type != 0).mean() * 100),
    }
    if verbose:
        print("Reviewer 3 General 1: catalogued in-channel obstructions")
        print(f"study reaches                 : {res['n_study_reaches']}")
        print(f"  obstr_type                  : {res['study_obstr_counts']}")
        print(f"  grod_id>0 / hfalls_id>0 / p_dam_id>0 : "
              f"{res['study_grod_gt0']} / {res['study_hfalls_gt0']} / {res['study_pdam_gt0']}")
        print(f"SWORD v17b NA reaches         : {res['sword_na_reaches']:,}")
        print(f"  obstr_type distribution     : {res['sword_obstr_counts']}")
        print(f"  non-zero share              : {res['sword_obstr_nonzero_pct']:.2f}%"
              "   (column is populated, so an all-zero study set is informative)")
        print(f"cross-check in SWORD original : {res['matched_in_sword']}/{res['n_study_reaches']} matched, "
              f"obstr_type {res['matched_obstr_counts']}")
        print(f"node-level cross-check        : {res['study_nodes']:,} nodes, "
              f"obstr_type {res['study_node_obstr_counts']}")
        print(f"probability all {res['n_study_reaches']} are zero by chance : "
              f"{res['p_all_zero_by_chance']:.2e}")
    return res


# Reviewer 4 overarching, 12, 16: shipped per-pixel layover fields
def _best_versions(files):
    """Keep the newest processor and counter per granule base name."""
    import re
    best = {}
    for f in files:
        m = re.match(r"(.+)_(P[IG]D\d)_(\d+)\.nc", os.path.basename(f))
        if not m:
            continue
        base, proc, cnt = m.group(1), m.group(2), int(m.group(3))
        if base not in best:
            best[base] = (f, proc, cnt)
        else:
            _, oproc, ocnt = best[base]
            if (proc == "PGD0" and oproc == "PID0") or (proc == oproc and cnt > ocnt):
                best[base] = (f, proc, cnt)
    return [v[0] for v in best.values()]


def extract_pixc_layover_fields(pixc_file, pixcvec_file, polygon_gdf):
    """Return a dataframe of the shipped layover-related fields for water pixels
    inside the reach polygon, plus the satellite state from the tvp group."""
    import xarray as xr
    from shapely.geometry import Point

    ds = xr.open_dataset(pixc_file, group="pixel_cloud")
    lon, lat = ds["longitude"].values, ds["latitude"].values
    bounds = polygon_gdf.total_bounds
    bbox = (lon >= bounds[0]) & (lon <= bounds[2]) & (lat >= bounds[1]) & (lat <= bounds[3])
    if bbox.sum() == 0:
        ds.close()
        return None

    classification = ds["classification"].values[bbox]
    water = classification > 2
    if water.sum() == 0:
        ds.close()
        return None
    idx = np.where(bbox)[0][water]

    poly = polygon_gdf.geometry.iloc[0]
    try:  # shapely 2 vectorises this; the per-point loop dominates otherwise
        import shapely
        inside = shapely.contains_xy(poly, lon[idx], lat[idx])
    except Exception:
        inside = np.array([poly.contains(Point(x, y)) for x, y in zip(lon[idx], lat[idx])])
    if inside.sum() == 0:
        ds.close()
        return None
    idx = idx[inside]

    def get(name):
        return ds[name].values[idx] if name in ds else np.full(idx.size, np.nan)

    df = pd.DataFrame({
        "longitude": lon[idx], "latitude": lat[idx],
        "classification": ds["classification"].values[idx],
        "height": get("height"),
        "bright_land_flag": get("bright_land_flag"),
        "layover_impact": get("layover_impact"),
        "geolocation_qual": get("geolocation_qual"),
        "classification_qual": get("classification_qual"),
        "sig0": get("sig0"),
        "inc": get("inc"),
        "cross_track": get("cross_track"),
        "water_frac": get("water_frac"),
        "pixel_area": get("pixel_area"),
        "prior_water_prob": get("prior_water_prob"),
    })

    tvp = xr.open_dataset(pixc_file, group="tvp")
    df["sat_lat"] = np.nanmedian(tvp["latitude"].values)
    df["sat_lon"] = np.nanmedian(tvp["longitude"].values)
    df["sat_altitude"] = np.nanmedian(tvp["altitude"].values)
    df["heading"] = np.nanmedian(tvp["velocity_heading"].values)
    tvp.close()

    if pixcvec_file and os.path.exists(pixcvec_file):
        vec = xr.open_dataset(pixcvec_file)
        reach = vec["reach_id"].values[idx]
        node = vec["node_id"].values[idx]

        def valid(a):
            return (np.array([len(x) > 0 and x != b"" for x in a])
                    if a.dtype.kind == "S" else a > 0)

        df["assigned_to_reach"] = valid(reach) & valid(node)
        vec.close()
    else:
        df["assigned_to_reach"] = False

    df["granule"] = os.path.basename(pixc_file)
    ds.close()
    return df


def summarize_layover_fields(df, label=""):
    """Summarize the shipped layover indicators for one pass."""
    gq = df.geolocation_qual.values
    sig_bit = np.where(np.isfinite(gq), (gq.astype("int64") & LAYOVER_SIGNIFICANT_BIT) > 0, False)
    li = df.layover_impact.values
    finite = np.isfinite(li)
    bl = df.bright_land_flag.values
    bl_fin = bl[np.isfinite(bl)] if bl.dtype.kind == "f" else bl
    inc = df["inc"].values
    inc_fin = inc[np.isfinite(inc)]
    cq = df.classification_qual.values
    cq_int = np.where(np.isfinite(cq), cq, 0).astype("int64")
    gq_int = np.where(np.isfinite(gq), gq, 0).astype("int64")
    out = {
        "label": label,
        "n_water_pixels": int(len(df)),
        "n_assigned_to_reach": int(df.assigned_to_reach.sum()),
        # every layover or bright-land related indicator the product ships
        "pct_detected_water_but_bright_land": float(
            ((cq_int & CLASS_QUAL_BITS["detected_water_but_bright_land"]) > 0).mean() * 100),
        "pct_detected_water_but_no_prior_water": float(
            ((cq_int & CLASS_QUAL_BITS["detected_water_but_no_prior_water"]) > 0).mean() * 100),
        "pct_water_false_detection_suspect": float(
            ((cq_int & CLASS_QUAL_BITS["water_false_detection_rate_suspect"]) > 0).mean() * 100),
        "pct_phase_noise_suspect": float(
            ((gq_int & GEOLOC_QUAL_BITS["phase_noise_suspect"]) > 0).mean() * 100),
        "pct_bright_land_value1": float((bl == 1).mean() * 100) if bl.size else np.nan,
        "pct_bright_land_value2": float((bl == 2).mean() * 100) if bl.size else np.nan,
        "prior_water_prob_median": float(np.nanmedian(df.prior_water_prob.values))
        if "prior_water_prob" in df else np.nan,
        "bright_land_values": {int(k): int(v) for k, v in
                               zip(*np.unique(bl_fin, return_counts=True))} if bl_fin.size else {},
        "bright_land_fire_pct": float((bl_fin > 0).mean() * 100) if bl_fin.size else np.nan,
        "layover_significant_pct": float(sig_bit.mean() * 100),
        "layover_impact_median_m": float(np.median(li[finite])) if finite.any() else np.nan,
        "layover_impact_p95_abs_m": float(np.percentile(np.abs(li[finite]), 95)) if finite.any() else np.nan,
        "layover_impact_gt5m_pct": float((np.abs(li[finite]) > LAYOVER_IMPACT_THRESHOLD_M).mean() * 100)
        if finite.any() else np.nan,
        "inc_median_deg": float(np.median(inc_fin)) if inc_fin.size else np.nan,
        "sig0_median": float(np.nanmedian(df.sig0.values)),
        "height_std_m": float(np.nanstd(df.height.values)),
    }
    return out


def run_pixc_check(isa="0.06", cycle="039", verbose=True):
    """Compare the shipped per-pixel layover fields between the passes of one site."""
    import geopandas as gpd

    sites = gpd.read_file(SITES_SHP)
    col = "fctimp_mea" if "fctimp_mea" in sites.columns else "fctimp_mean"
    poly = sites[np.isclose(sites[col], float(isa), atol=0.01)].to_crs("EPSG:4326")
    if poly.empty:
        raise SystemExit(f"no site with ISA ~ {isa}")

    pixc_dir = f"{PROJ}/0_data_collection/SWOT_PIXC/fctimp_{isa}"
    vec_dir = f"{PROJ}/0_data_collection/SWOT_PIXCVec/fctimp_{isa}"
    pixc = _best_versions(glob.glob(f"{pixc_dir}/SWOT_L2_HR_PIXC_{cycle}_*.nc"))
    if not pixc:
        raise SystemExit(f"no PIXC granules for cycle {cycle} in {pixc_dir}")

    rows, frames = [], []
    for f in sorted(pixc):
        vec = os.path.join(vec_dir, os.path.basename(f).replace("PIXC_", "PIXCVec_"))
        df = extract_pixc_layover_fields(f, vec, poly)
        if df is None or df.empty:
            continue
        parts = os.path.basename(f).split("_")
        tag = f"pass {parts[5]} tile {parts[6]}"
        rows.append(summarize_layover_fields(df, tag))
        frames.append(df)

    if not frames:
        raise SystemExit("no water pixels found inside the reach polygon")

    allpix = pd.concat(frames, ignore_index=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = f"{OUT_DIR}/pixc_layover_fields_isa{isa}_cycle{cycle}.parquet"
    allpix.to_parquet(out, index=False)
    summary = pd.DataFrame(rows)

    if verbose:
        print("Reviewer 4 overarching, 12 and 16: shipped per-pixel layover fields")
        print(f"site ISA {isa}, cycle {cycle}, granules used: {len(frames)}")
        with pd.option_context("display.width", 200, "display.max_columns", 40):
            print(summary.drop(columns=["bright_land_values"]).to_string(index=False))
        print("\nbright_land_flag values per pass:")
        for r in rows:
            print(f"  {r['label']:24s} {r['bright_land_values']}")
        print(f"\nsaved pixel table: {out}")
    return summary, allpix


def run_pixc_sweep(threshold=0.1832, max_per_site=6, verbose=True):
    """Sweep every site with PIXC granules and relate the shipped per-pixel layover
    indicators to the gauge-validated reach error.

    For each site we take the high-error observations and a time-matched low-error
    observation, summarize the shipped layover fields over the water pixels of each
    pass, and join the summary to the gauge error. This tests whether the operational
    pixel-level indicators track the error that the gauges actually record.
    """
    import geopandas as gpd
    import re
    from datetime import timedelta

    sites = gpd.read_file(SITES_SHP)
    col = "fctimp_mea" if "fctimp_mea" in sites.columns else "fctimp_mean"
    isa_dirs = sorted(glob.glob(f"{PROJ}/0_data_collection/SWOT_PIXC/fctimp_*"))
    rows = []

    for d in isa_dirs:
        isa = os.path.basename(d).replace("fctimp_", "")
        sample = f"{WORK}/1_data_processing/fctimp_mean_{isa}_sample.parquet"
        if not os.path.exists(sample):
            continue
        poly = sites[np.isclose(sites[col], float(isa), atol=0.01)].to_crs("EPSG:4326")
        if poly.empty:
            continue

        obs = pd.read_parquet(sample)
        obs["time_dt"] = pd.to_datetime(obs["time_str"]).dt.tz_localize(None)
        obs["error"] = obs["wse_relative_m"] - obs["wse_usgs_relative_m"]
        obs["abs_error"] = obs["error"].abs()
        obs = obs.sort_values("abs_error", ascending=False)
        picks = pd.concat([obs.head(max_per_site // 2), obs.tail(max_per_site // 2)])

        pixc = _best_versions(glob.glob(f"{d}/SWOT_L2_HR_PIXC_*.nc"))
        vec_dir = f"{PROJ}/0_data_collection/SWOT_PIXCVec/fctimp_{isa}"

        for _, o in picks.iterrows():
            match = None
            for f in pixc:
                parts = os.path.basename(f).split("_")
                try:
                    start = pd.to_datetime(parts[7], format="%Y%m%dT%H%M%S")
                except Exception:
                    continue
                if abs((start - o.time_dt).total_seconds()) <= 300:
                    match = f
                    break
            if match is None:
                continue
            vec = os.path.join(vec_dir, os.path.basename(match).replace("PIXC_", "PIXCVec_"))
            df = extract_pixc_layover_fields(match, vec, poly)
            if df is None or df.empty:
                continue
            s = summarize_layover_fields(df, os.path.basename(match))
            s.update({
                "isa": float(isa),
                "reach_id": int(o.reach_id),
                "time_str": str(o.time_str),
                "abs_error_m": float(o.abs_error),
                "high_error": bool(o.abs_error >= threshold),
            })
            rows.append(s)
            if verbose:
                print(f"  {isa} {os.path.basename(match)[:46]} err={o.abs_error:.3f} "
                      f"bright={s['bright_land_fire_pct']:.2f}% "
                      f"layover_sig={s['layover_significant_pct']:.2f}% "
                      f"sig0={s['sig0_median']:.1f}")

    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("no matched granules")
    os.makedirs(OUT_DIR, exist_ok=True)
    path = f"{OUT_DIR}/pixc_layover_sweep.parquet"
    # bright_land_values holds a dict keyed by flag value; parquet needs string keys
    out["bright_land_values"] = out.bright_land_values.apply(
        lambda d: {str(k): int(v) for k, v in (d or {}).items()})
    out.to_parquet(path, index=False)

    if verbose:
        from scipy import stats as st
        print()
        print("PIXC sweep: do the shipped per-pixel indicators track the gauge error?")
        print(f"passes analyzed: {len(out)} across {out.isa.nunique()} sites")
        for c in ["bright_land_fire_pct", "layover_significant_pct",
                  "layover_impact_gt5m_pct", "sig0_median"]:
            sub = out[["abs_error_m", c]].dropna()
            if len(sub) > 3:
                r, p = st.spearmanr(sub.abs_error_m, sub[c])
                hi = out[out.high_error][c].median()
                lo = out[~out.high_error][c].median()
                print(f"  {c:26s} Spearman vs |error| = {r:+.3f} (p={p:.3f}) | "
                      f"median high-error {hi:.2f} vs low-error {lo:.2f}")
        print(f"\nsaved: {path}")
    return out


# Reviewer 3 General 1 (extended): distance along the network to the nearest
# catalogued obstruction, traversing SWORD topology upstream and downstream.
#
# SWORD variables used (Product Description v17b):
#   obstr_type  p.11  0 no dam, 1 dam, 2 lock, 3 low permeable dam, 4 waterfall
#   grod_id     p.11  GROD identifier for obstr_type 1-3
#   hfalls_id   p.11  HydroFALLS identifier for obstr_type 4
#   lakeflag    p.11  0 river, 1 lake or reservoir, 2 canal, 3 tidally influenced
#   dist_out    p.11  distance from the river outlet
#   reach_len   p.15  reach length along the high-resolution centerline
#   rch_id_up / rch_id_dn, n_rch_up / n_rch_down  p.16  network topology
OBSTR_LABELS = {0: "none", 1: "dam", 2: "lock", 3: "low permeable dam",
                4: "waterfall", 5: "undocumented code 5"}


def _parse_ids(value):
    """SWORD stores neighbour reach ids as a space separated string."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    out = []
    for tok in str(value).replace(",", " ").split():
        tok = tok.strip()
        if tok and tok.lower() != "none":
            try:
                out.append(int(float(tok)))
            except ValueError:
                continue
    return out


def load_sword_network(verbose=True):
    """Load the SWORD NA reach table and build the topology lookup."""
    import geopandas as gpd

    files = sorted(glob.glob(f"{SWORD_SHP}/NA/*reaches*.shp"))
    want = ["reach_id", "rch_id_up", "rch_id_dn", "n_rch_up", "n_rch_dn",
            "reach_len", "obstr_type", "grod_id", "hfalls_id", "lakeflag",
            "type", "dist_out", "facc", "n_chan_mod", "width"]
    parts = []
    for f in files:
        g = gpd.read_file(f)
        parts.append(g[[c for c in want if c in g.columns]])
    sw = pd.concat(parts, ignore_index=True)
    sw["reach_id"] = sw.reach_id.astype("int64")
    if verbose:
        print(f"SWORD NA reaches loaded: {len(sw):,}")
    net = {}
    for row in sw.itertuples(index=False):
        net[row.reach_id] = {
            "up": _parse_ids(getattr(row, "rch_id_up", None)),
            "dn": _parse_ids(getattr(row, "rch_id_dn", None)),
            "len_m": float(getattr(row, "reach_len", np.nan) or np.nan),
            "obstr": int(getattr(row, "obstr_type", 0) or 0),
            "grod": int(getattr(row, "grod_id", 0) or 0),
            "hfalls": int(getattr(row, "hfalls_id", 0) or 0),
            "lakeflag": int(getattr(row, "lakeflag", 0) or 0),
            "dist_out": float(getattr(row, "dist_out", np.nan) or np.nan),
        }
    return sw, net


def nearest_obstruction(reach_id, net, direction, max_km=200.0, max_hops=400):
    """Walk the network in one direction and return the first catalogued obstruction.

    Distance is the cumulative length of the reaches crossed after leaving the study
    reach, measured to the near edge of the obstructed reach, so it is a lower bound
    on the distance from the gauge to the structure.
    """
    from collections import deque

    start = net.get(reach_id)
    if start is None:
        return {"found": False, "reason": "reach not in network"}
    seen = {reach_id}
    queue = deque((nb, 1, 0.0) for nb in start[direction] if nb in net)
    best = None
    reached_end = True
    while queue:
        rid, hops, dist_m = queue.popleft()
        if rid in seen:
            continue
        seen.add(rid)
        info = net[rid]
        if info["obstr"] != 0:
            best = {"found": True, "reach_id": int(rid), "hops": hops,
                    "distance_km": dist_m / 1000.0, "obstr_type": info["obstr"],
                    "obstr_label": OBSTR_LABELS.get(info["obstr"], str(info["obstr"])),
                    "grod_id": info["grod"], "hfalls_id": info["hfalls"]}
            break
        nd = dist_m + (info["len_m"] if np.isfinite(info["len_m"]) else 0.0)
        if hops >= max_hops or nd / 1000.0 > max_km:
            reached_end = False
            continue
        for nb in info[direction]:
            if nb in net and nb not in seen:
                queue.append((nb, hops + 1, nd))
    if best:
        return best
    return {"found": False, "hops_searched": len(seen),
            "reason": "network end reached" if reached_end else f"search limit {max_km} km"}


def run_network_check(max_km=200.0, verbose=True):
    """Distance to the nearest obstruction for every study reach, and the
    relationship between that distance and the gauge-validated error."""
    import pyarrow.parquet as pq
    from scipy import stats as st

    sw, net = load_sword_network(verbose=verbose)
    study = pq.read_table(REACH_TABLE, columns=[
        "reach_id", "obstr_type", "grod_id", "hfalls_id", "p_dam_id", "lakeflag",
        "type", "n_chan_mod", "facc", "reach_len", "fctimp_mean",
        "site_no"]).to_pandas()
    study["reach_id"] = study.reach_id.astype("int64")
    reaches = study.drop_duplicates("reach_id").set_index("reach_id")

    # gauge-validated errors live in the matched observation table
    obs = pq.read_table(ERROR_TABLE, columns=[
        "reach_id", "wse_relative_m", "wse_usgs_relative_m"]).to_pandas()
    obs["reach_id"] = obs.reach_id.astype("int64")
    obs["abs_error_m"] = (obs.wse_relative_m - obs.wse_usgs_relative_m).abs()
    err = obs.groupby("reach_id").abs_error_m.median()

    rows = []
    for rid in reaches.index:
        up = nearest_obstruction(rid, net, "up", max_km=max_km)
        dn = nearest_obstruction(rid, net, "dn", max_km=max_km)
        cand = [(d["distance_km"], k, d) for k, d in (("upstream", up), ("downstream", dn))
                if d.get("found")]
        near = min(cand) if cand else None
        rows.append({
            "reach_id": rid,
            "isa": float(reaches.at[rid, "fctimp_mean"]),
            "median_abs_error_m": float(err.get(rid, np.nan)),
            "up_found": up.get("found", False),
            "up_hops": up.get("hops", np.nan),
            "up_km": up.get("distance_km", np.nan),
            "up_type": up.get("obstr_label", ""),
            "dn_found": dn.get("found", False),
            "dn_hops": dn.get("hops", np.nan),
            "dn_km": dn.get("distance_km", np.nan),
            "dn_type": dn.get("obstr_label", ""),
            "nearest_km": near[0] if near else np.nan,
            "nearest_dir": near[1] if near else "none within search",
            "nearest_type": near[2]["obstr_label"] if near else "",
            "nearest_hops": near[2]["hops"] if near else np.nan,
        })
    out = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = f"{OUT_DIR}/obstruction_network_distance.parquet"
    out.to_parquet(path, index=False)

    if verbose:
        print()
        print("Distance along the SWORD network to the nearest catalogued obstruction")
        n = len(out)
        for d, col in (("upstream", "up_km"), ("downstream", "dn_km")):
            f = out[col].notna()
            print(f"{d:11s} obstruction found for {int(f.sum())}/{n} reaches")
            if f.any():
                v = out.loc[f, col]
                print(f"            distance km: median {v.median():.1f}, "
                      f"p25 {v.quantile(.25):.1f}, p75 {v.quantile(.75):.1f}, min {v.min():.1f}")
                h = out.loc[f, col.replace('_km', '_hops')]
                print(f"            reaches crossed: median {h.median():.0f}, min {int(h.min())}")
        f = out.nearest_km.notna()
        print(f"\nnearest in either direction: {int(f.sum())}/{n}")
        if f.any():
            v = out.loc[f, "nearest_km"]
            print(f"  median {v.median():.1f} km, p25 {v.quantile(.25):.1f}, "
                  f"p75 {v.quantile(.75):.1f}, min {v.min():.1f}")
            for cut in (1, 5, 10, 20, 50):
                print(f"  within {cut:3d} km: {int((out.nearest_km <= cut).sum()):3d} reaches "
                      f"({(out.nearest_km <= cut).mean() * 100:.1f}%)")
            print(f"  types: {out.loc[f, 'nearest_type'].value_counts().to_dict()}")
            print(f"  direction: {out.loc[f, 'nearest_dir'].value_counts().to_dict()}")

        print()
        print("Confound test: does proximity to a structure explain the error?")
        sub = out[["nearest_km", "median_abs_error_m", "isa"]].dropna()
        if len(sub) > 10:
            r, p = st.spearmanr(sub.nearest_km, sub.median_abs_error_m)
            r2, p2 = st.spearmanr(sub.nearest_km, sub.isa)
            print(f"  Spearman(distance, median |error|) = {r:+.3f} (p={p:.3f}, n={len(sub)})")
            print(f"  Spearman(distance, ISA)            = {r2:+.3f} (p={p2:.3f})")
            near = sub[sub.nearest_km <= 10]
            far = sub[sub.nearest_km > 10]
            if len(near) > 3 and len(far) > 3:
                u, pu = st.mannwhitneyu(near.median_abs_error_m, far.median_abs_error_m)
                print(f"  median |error|: within 10 km {near.median_abs_error_m.median():.3f} m "
                      f"(n={len(near)}) vs beyond {far.median_abs_error_m.median():.3f} m "
                      f"(n={len(far)}), Mann-Whitney p={pu:.3f}")

        print()
        print("All obstruction-related SWORD variables, study reaches vs SWORD NA")
        for col, labels in (("obstr_type", OBSTR_LABELS),
                            ("lakeflag", {0: "river", 1: "lake or reservoir",
                                          2: "canal", 3: "tidally influenced"})):
            if col in reaches.columns and col in sw.columns:
                ours = reaches[col].value_counts().to_dict()
                base = sw[col].value_counts(normalize=True).mul(100).round(2).to_dict()
                print(f"  {col:11s} study {ours}")
                print(f"              SWORD NA % {base}")
        for col in ("grod_id", "hfalls_id", "p_dam_id"):
            if col in reaches.columns:
                print(f"  {col:11s} study non-zero {int((reaches[col] > 0).sum())}/{len(reaches)}"
                      + (f"   SWORD NA {(sw[col] > 0).mean() * 100:.2f}%" if col in sw.columns else ""))
        print(f"\nsaved: {path}")
    return out


# Cross-level test: is there information in the pixel cloud that the river vector
# product discards? For each pass we ask whether excluding the pixels the product
# itself flags would move the reach elevation toward the gauge.

def _cross_level_one(payload):
    """Process one matched observation. Defined at module level so it can be pickled."""
    task, riversp, threshold = payload
    (pixc_file, vec_file, poly, isa, reach_id, time_str, error, abs_error) = task
    df = extract_pixc_layover_fields(pixc_file, vec_file, poly)
    if df is None or len(df) < 50:
        return None

    h = df.height.values
    gq = df.geolocation_qual.values
    lay = np.where(np.isfinite(gq), (gq.astype("int64") & LAYOVER_SIGNIFICANT_BIT) > 0, False)
    bl = df.bright_land_flag.values
    bright = np.isfinite(bl) & (bl > 0)
    fin = np.isfinite(h)

    def med(mask):
        m = mask & fin
        return float(np.median(h[m])) if m.sum() >= 20 else np.nan

    h_all = med(np.ones(len(df), bool))
    h_nolay = med(~lay)
    h_nobright = med(~bright)
    h_lay_only = med(lay)

    t = pd.to_datetime(time_str).tz_localize(None) if pd.to_datetime(time_str).tzinfo \
        else pd.to_datetime(time_str)
    rs = riversp[riversp.reach_id == reach_id]
    rs = rs.iloc[(rs.t - t).abs().argsort()[:1]] if len(rs) else rs
    qb = int(rs.reach_q_b.iloc[0]) if len(rs) else -1

    summ = summarize_layover_fields(df, os.path.basename(pixc_file))
    summ.pop("label", None)
    summ.pop("bright_land_values", None)
    return {**summ,
            "isa": isa, "reach_id": reach_id, "time_str": time_str,
            "error_m": error, "abs_error_m": abs_error,
            "high_error": bool(abs_error >= threshold),
            "n_pixels": int(len(df)),
            "pct_layover_flagged": float(lay.mean() * 100),
            "pct_bright_flagged": float(bright.mean() * 100),
            "h_all": h_all, "h_no_layover": h_nolay, "h_no_bright": h_nobright,
            "h_layover_only": h_lay_only,
            "delta_layover_m": (h_nolay - h_all) if np.isfinite(h_nolay) else np.nan,
            "delta_bright_m": (h_nobright - h_all) if np.isfinite(h_nobright) else np.nan,
            "layover_pixel_bias_m": (h_lay_only - h_nolay)
            if np.isfinite(h_lay_only) and np.isfinite(h_nolay) else np.nan,
            "riversp_q_b": qb,
            "riversp_bright_land_bit": int((qb & 128) > 0) if qb >= 0 else -1}

def run_cross_level_check(max_per_site=None, threshold=0.1832, workers=1, verbose=True):
    import geopandas as gpd
    import pyarrow.parquet as pq
    from scipy import stats as st

    sites = gpd.read_file(SITES_SHP)
    col = "fctimp_mea" if "fctimp_mea" in sites.columns else "fctimp_mean"
    riversp = pq.read_table(ERROR_TABLE, columns=[
        "reach_id", "time_str", "reach_q_b", "wse_relative_m",
        "wse_usgs_relative_m"]).to_pandas()
    riversp["reach_id"] = riversp.reach_id.astype("int64")
    riversp["t"] = pd.to_datetime(riversp.time_str).dt.tz_localize(None)

    tasks = []
    for d in sorted(glob.glob(f"{PROJ}/0_data_collection/SWOT_PIXC/fctimp_*")):
        isa = os.path.basename(d).replace("fctimp_", "")
        sample = f"{WORK}/1_data_processing/fctimp_mean_{isa}_sample.parquet"
        if not os.path.exists(sample):
            continue
        poly = sites[np.isclose(sites[col], float(isa), atol=0.01)].to_crs("EPSG:4326")
        if poly.empty:
            continue
        obs = pd.read_parquet(sample)
        obs["t"] = pd.to_datetime(obs.time_str).dt.tz_localize(None)
        obs["error"] = obs.wse_relative_m - obs.wse_usgs_relative_m
        obs["abs_error"] = obs.error.abs()
        obs = obs.sort_values("abs_error", ascending=False)
        # Sampling the extremes inflates any correlation, so take every observation
        # that has a matching granule unless a cap is requested.
        if max_per_site is None:
            picks = obs
        else:
            picks = pd.concat([obs.head(max_per_site // 2), obs.tail(max_per_site // 2)])
        pixc = _best_versions(glob.glob(f"{d}/SWOT_L2_HR_PIXC_*.nc"))
        vec_dir = f"{PROJ}/0_data_collection/SWOT_PIXCVec/fctimp_{isa}"

        for _, o in picks.iterrows():
            match = None
            for f in pixc:
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
            vec = os.path.join(vec_dir, os.path.basename(match).replace("PIXC_", "PIXCVec_"))
            tasks.append((match, vec, poly, float(isa), int(o.reach_id), str(o.time_str),
                          float(o.error), float(o.abs_error)))
    if verbose:
        print(f"matched observations to process: {len(tasks)}", flush=True)

    if workers and workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_cross_level_one,
                                  [(t, riversp, threshold) for t in tasks], chunksize=1))
    else:
        results = [_cross_level_one((t, riversp, threshold)) for t in tasks]
    rows = [r for r in results if r]
    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("no matched passes")
    os.makedirs(OUT_DIR, exist_ok=True)
    path = f"{OUT_DIR}/cross_level_pixc_vs_riversp.parquet"
    out.to_parquet(path, index=False)
    # the same extraction also serves the indicator sweep
    sweep_cols = [c for c in out.columns if c.startswith(("pct_", "n_", "layover_", "bright_",
                  "prior_", "sig0_", "inc_", "height_"))] + ["isa", "reach_id", "time_str",
                  "abs_error_m", "high_error"]
    out[[c for c in dict.fromkeys(sweep_cols) if c in out.columns]].to_parquet(
        f"{OUT_DIR}/pixc_layover_sweep.parquet", index=False)

    if verbose:
        print()
        print("Would applying the product's own pixel flags move WSE toward the gauge?")
        print(f"passes: {len(out)}, sites: {out.isa.nunique()}")
        print(f"RiverSP bright_land bit set in any pass: "
              f"{int((out.riversp_bright_land_bit == 1).sum())}/{len(out)}")
        for tag, dcol in (("layover_significant", "delta_layover_m"),
                          ("bright_land", "delta_bright_m")):
            sub = out[["error_m", "abs_error_m", dcol]].dropna()
            if len(sub) < 5:
                continue
            corrected = (sub.error_m + sub[dcol]).abs()
            better = int((corrected < sub.abs_error_m).sum())
            r, p = st.spearmanr(sub.error_m, sub[dcol])
            print(f"\nexcluding pixels flagged {tag}:")
            print(f"  median height shift        : {sub[dcol].median():+.4f} m "
                  f"(p5 {sub[dcol].quantile(.05):+.3f}, p95 {sub[dcol].quantile(.95):+.3f})")
            print(f"  Spearman(shift, error)     : {r:+.3f} (p={p:.3f}) "
                  "[negative would mean the flag pushes back against the error]")
            print(f"  passes where |error| falls : {better}/{len(sub)}")
            print(f"  mean |error| before / after: {sub.abs_error_m.mean():.3f} / "
                  f"{corrected.mean():.3f} m")
        sub = out.dropna(subset=["layover_pixel_bias_m"])
        if len(sub) > 5:
            print(f"\nheight offset of flagged pixels relative to unflagged: "
                  f"median {sub.layover_pixel_bias_m.median():+.4f} m, "
                  f"|median| > 0.05 m in {int((sub.layover_pixel_bias_m.abs() > 0.05).sum())}/{len(sub)} passes")
        print(f"\nsaved: {path}")
    return out


# Do the shipped indicators predict the error, or do they only track urbanization?
# Every pixel-level indicator correlates strongly with ISA, so the raw correlation
# with the error is confounded. Two ways to separate them:
#   partial Spearman controlling for ISA, and the within-site correlation, where
#   ISA is constant by construction.
INDICATORS = [
    "bright_land_fire_pct", "pct_bright_land_value1", "pct_bright_land_value2",
    "pct_detected_water_but_bright_land", "pct_detected_water_but_no_prior_water",
    "pct_water_false_detection_suspect", "layover_significant_pct",
    "layover_impact_gt5m_pct", "pct_phase_noise_suspect",
    "prior_water_prob_median", "sig0_median", "inc_median_deg",
]


def partial_spearman(x, y, z):
    """Spearman correlation between x and y with z partialled out."""
    from scipy import stats as st
    rxy = st.spearmanr(x, y).statistic
    rxz = st.spearmanr(x, z).statistic
    ryz = st.spearmanr(y, z).statistic
    denom = np.sqrt(max((1 - rxz ** 2) * (1 - ryz ** 2), 1e-12))
    r = (rxy - rxz * ryz) / denom
    n = len(x)
    if n > 4 and abs(r) < 1:
        t = r * np.sqrt((n - 3) / max(1 - r ** 2, 1e-12))
        p = 2 * st.t.sf(abs(t), n - 3)
    else:
        p = np.nan
    return r, p, rxy


def run_partial_check(verbose=True):
    """Separate the urbanization signal from any genuine error signal."""
    from scipy import stats as st
    path = f"{OUT_DIR}/pixc_layover_sweep.parquet"
    d = pd.read_parquet(path)
    rows = []
    for c in INDICATORS:
        if c not in d.columns:
            continue
        sub = d[["abs_error_m", c, "isa"]].dropna()
        if len(sub) < 8 or sub[c].nunique() < 3:
            continue
        r, p, raw = partial_spearman(sub[c].values, sub.abs_error_m.values, sub.isa.values)
        r_isa = st.spearmanr(sub[c], sub.isa).statistic
        # within-site correlations, where ISA is fixed
        within = []
        for _, g in d.dropna(subset=[c, "abs_error_m"]).groupby(d.isa.round(3)):
            if len(g) >= 4 and g[c].nunique() >= 3:
                within.append((st.spearmanr(g[c], g.abs_error_m).statistic, len(g)))
        if within:
            zs = [np.arctanh(np.clip(w, -0.999, 0.999)) for w, _ in within]
            wts = [n - 3 for _, n in within]
            pooled = float(np.tanh(np.average(zs, weights=wts))) if sum(wts) > 0 else np.nan
        else:
            pooled = np.nan
        rows.append({"indicator": c, "n": len(sub), "raw_rho": raw, "rho_vs_ISA": r_isa,
                     "partial_rho": r, "partial_p": p, "within_site_rho": pooled,
                     "n_sites_used": len(within)})
    out = pd.DataFrame(rows).sort_values("partial_rho", key=abs, ascending=False)
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_parquet(f"{OUT_DIR}/pixc_partial_correlation.parquet", index=False)
    if verbose:
        print("Pixel-level indicators against the gauge error, before and after removing ISA")
        print(f"{'indicator':40s} {'n':>4s} {'raw':>7s} {'vs ISA':>8s} "
              f"{'partial':>8s} {'p':>7s} {'within':>8s} {'sites':>6s}")
        for r in out.itertuples(index=False):
            wr = f"{r.within_site_rho:+.3f}" if np.isfinite(r.within_site_rho) else "   n/a"
            print(f"{r.indicator:40s} {r.n:4d} {r.raw_rho:+7.3f} {r.rho_vs_ISA:+8.3f} "
                  f"{r.partial_rho:+8.3f} {r.partial_p:7.3f} {wr:>8s} {r.n_sites_used:6d}")
        print("\nraw     : Spearman with |error|, confounded by urbanization")
        print("vs ISA  : how strongly the indicator tracks urbanization")
        print("partial : Spearman with |error| after partialling out ISA")
        print("within  : Fisher-z pooled correlation inside sites, where ISA is fixed")
        print(f"\nsaved: {OUT_DIR}/pixc_partial_correlation.parquet")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("check", choices=["sword", "network", "pixc", "sweep", "crosslevel", "partial", "all"])
    ap.add_argument("--isa", default="0.06")
    ap.add_argument("--cycle", default="039")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-per-site", dest="max_per_site", type=int, default=None,
                    help="cap passes per site; omit to use every matched observation")
    a = ap.parse_args()
    if a.check in ("sword", "all"):
        check_sword_obstructions()
        print()
    if a.check in ("pixc", "all"):
        run_pixc_check(isa=a.isa, cycle=a.cycle)
    if a.check in ("network", "all"):
        run_network_check()
        print()
    if a.check == "sweep":
        run_pixc_sweep()
    if a.check == "crosslevel":
        run_cross_level_check(max_per_site=a.max_per_site, workers=a.workers)
    if a.check == "partial":
        run_partial_check()
