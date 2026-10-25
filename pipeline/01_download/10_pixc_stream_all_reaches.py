#!/usr/bin/env python
"""
Node-resolved coherence for every study reach, by streaming the pixel cloud.

Holding the pixel cloud for all 409 reaches would take roughly ten terabytes, which the
project filesystem cannot host. This pipeline never keeps a granule: each task downloads
one to node-local scratch, extracts the per-node coherence statistics for every study
reach the granule covers, writes those few kilobytes, and deletes the granule before
moving to the next. Peak disk stays at a few gigabytes per task.

  scope    build the granule manifest for the gauge-matched observations
  extract  one array task per manifest slice, download and process and delete
  merge    concatenate the shards

  python 10_pixc_stream_all_reaches.py scope
  python 10_pixc_stream_all_reaches.py extract --shard 0 --nshards 40 --workers 4
  python 10_pixc_stream_all_reaches.py merge
"""
import argparse
import gc
import glob
import os
import shutil
import time

import numpy as np
import pandas as pd

WORK = "/work/pi_kandread_umass_edu/swot-urban"
PROJ = "/project/pi_kandread_umass_edu/swot-urban"
OUT = f"{PROJ}/3_analysis_results/reviewer_checks/coh_all"
MANIFEST = f"{OUT}/granule_manifest.parquet"
OBS = f"{WORK}/2_urban_reaches/all_final_shape.parquet"
REACHES = f"{WORK}/2_urban_reaches/all_reaches_vD.parquet"
LOW_COH_CLASSES = (6, 7)
LAYOVER_BIT = 1
MIN_PIX = 20
WINDOW_S = 300


def scope():
    import earthaccess
    import geopandas as gpd
    import pyarrow.parquet as pq

    earthaccess.login(strategy="netrc")
    os.makedirs(OUT, exist_ok=True)
    ob = pq.read_table(OBS, columns=["reach_id", "time_str"]).to_pandas()
    ob["reach_id"] = ob.reach_id.astype("int64")
    ob["t"] = pd.to_datetime(ob.time_str, errors="coerce", utc=True).dt.tz_localize(None)
    ob = ob.dropna(subset=["t"])
    g = gpd.read_parquet(REACHES)[["reach_id", "geometry"]]
    g["reach_id"] = g.reach_id.astype("int64")
    g = g.drop_duplicates("reach_id").set_index("reach_id")

    rows = []
    reaches = sorted(ob.reach_id.unique())
    for i, rid in enumerate(reaches, 1):
        if rid not in g.index:
            continue
        b = g.loc[rid].geometry.bounds
        times = ob[ob.reach_id == rid].t
        lo, hi = times.min(), times.max()
        try:
            res = earthaccess.search_data(
                short_name="SWOT_L2_HR_PIXC_D",
                bounding_box=(b[0], b[1], b[2], b[3]),
                temporal=(str(lo.date()), str(hi.date() + pd.Timedelta(days=1))),
                count=3000)
        except Exception as ex:
            print(f"  reach {rid}: search failed {ex}", flush=True)
            continue
        tv = times.values.astype("datetime64[s]")
        for r in res:
            name = r["meta"]["native-id"]
            parts = name.split("_")
            if len(parts) < 9:
                continue
            try:
                start = np.datetime64(pd.to_datetime(parts[7], format="%Y%m%dT%H%M%S"), "s")
            except Exception:
                continue
            if np.min(np.abs((tv - start).astype("int64"))) > WINDOW_S:
                continue
            rows.append({"reach_id": rid, "granule": name,
                         "url": r.data_links()[0] if r.data_links() else "",
                         "size_mb": float(r.size()), "start": str(start)})
        if i % 25 == 0:
            uq = len({x['granule'] for x in rows})
            print(f"  {i}/{len(reaches)} reaches, {len(rows)} pairs, {uq} unique granules",
                  flush=True)
    m = pd.DataFrame(rows).drop_duplicates(["reach_id", "granule"])
    # One version per (tile, start time): PGD0 before PID0, then the highest product counter
    # (the rule of the 12-site notebook 05_pixc_pixcvec_download). Added 2026-09-11 after the
    # earlier manifest was found to carry 509 tile-passes twice (both CRIDs or both counters).
    parts = m.granule.str.extract(r"SWOT_L2_HR_PIXC_(\d{3})_(\d{3})_(\d{3}[LR])_(\d{8}T\d{6})_\d{8}T\d{6}_(P\w{3})_(\d+)")
    m["_tile"], m["_t0"] = parts[2], parts[3]
    m["_crank"] = parts[4].map({"PGD0": 0, "PID0": 1}).fillna(9)
    m["_cnt"] = pd.to_numeric(parts[5], errors="coerce")
    best = (m.drop_duplicates("granule")
             .sort_values(["_tile", "_t0", "_crank", "_cnt"], ascending=[True, True, True, False])
             .drop_duplicates(["_tile", "_t0"]).granule)
    n_before = m.granule.nunique()
    m = m[m.granule.isin(set(best))].drop(columns=["_tile", "_t0", "_crank", "_cnt"])
    print(f"version rule (PGD0 > PID0, highest counter): {n_before:,} -> {m.granule.nunique():,} unique granules")
    m.to_parquet(MANIFEST, index=False)
    uq = m.granule.nunique()
    tot = m.drop_duplicates("granule").size_mb.sum() / 1024
    print(f"\nmanifest: {len(m):,} reach-granule pairs, {uq:,} unique granules, "
          f"{tot:.1f} GB to stream, {m.reach_id.nunique()} reaches")
    print(f"saved {MANIFEST}")


def granule_node_stats(path, polys, vec_path=None):
    """Per-node coherence for every study reach inside this granule.

    Node identity comes from the pixel cloud vector product, which is the assignment
    the river processor itself uses. Without it the statistics would only resolve to
    the reach, and this study works at node scale."""
    import xarray as xr
    import shapely

    nid = None
    if vec_path and os.path.exists(vec_path):
        try:
            v = xr.open_dataset(vec_path)
            raw = v["node_id"].values
            v.close()
            ser = pd.Series(raw).astype(str).str.strip()
            nid = np.where(ser.str.isdigit().values,
                           pd.to_numeric(ser, errors="coerce").fillna(0).values, 0).astype("int64")
        except Exception:
            nid = None

    ds = xr.open_dataset(path, group="pixel_cloud")
    n = ds.sizes.get("points", 0)
    if not n:
        ds.close(); return None
    lon = ds["longitude"].values
    lat = ds["latitude"].values
    cls = ds["classification"].values

    def g(name):
        return ds[name].values.astype("float64") if name in ds else np.full(n, np.nan)

    cp, pp, pm = g("coherent_power"), g("power_plus_y"), g("power_minus_y")
    sig0, gq = g("sig0"), g("geolocation_qual")
    li, xt = g("layover_impact"), np.abs(g("cross_track")) / 1000.0
    ds.close()
    if nid is not None and nid.size != n:
        nid = None
    denom = np.sqrt(np.clip(pp, 0, None) * np.clip(pm, 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma = np.where(denom > 0, cp / denom, np.nan)
    gamma = np.clip(gamma, 0, 2) - 1.0
    water = cls > 2

    out = []
    for rid, poly in polys:
        b = poly.bounds
        m = water & (lon >= b[0]) & (lon <= b[2]) & (lat >= b[1]) & (lat <= b[3])
        if m.sum() < MIN_PIX:
            continue
        idx = np.where(m)[0]
        inside = shapely.contains_xy(poly, lon[idx], lat[idx])
        idx = idx[inside]
        if idx.size < MIN_PIX:
            continue
        gm, sg = gamma[idx], sig0[idx]
        fin = np.isfinite(gm) & np.isfinite(sg)
        if fin.sum() < MIN_PIX:
            continue
        sel = idx[fin]
        df = pd.DataFrame({
            "node_id": (nid[sel] if nid is not None else np.zeros(sel.size, "int64")),
            "gamma": gamma[sel], "sig0": sig0[sel],
            "lay": (np.nan_to_num(gq[sel]).astype("int64") & LAYOVER_BIT) > 0,
            "lowcoh": np.isin(cls[sel], LOW_COH_CLASSES),
            "li": li[sel], "xt": xt[sel]})
        ls = np.log10(np.clip(df.sig0.values, 1e-3, None))
        ok = np.isfinite(ls) & np.isfinite(df.gamma.values)
        if ok.sum() > 30 and np.nanstd(ls[ok]) > 1e-6:
            sl, ic = np.polyfit(ls[ok], df.gamma.values[ok], 1)
            df["resid"] = df.gamma - (sl * ls + ic)
        else:
            df["resid"] = np.nan
        bc = np.nanpercentile(df.sig0, 75)
        rc = (np.nanpercentile(df.resid.dropna(), 25)
              if df.resid.notna().sum() > 50 else -np.inf)
        df["bi"] = (df.sig0 >= bc) & (df.resid <= rc)
        keys = ["node_id"] if nid is not None else []
        grouped = df.groupby("node_id") if keys else [(0, df)]
        for node, gdf in (grouped if keys else grouped):
            if keys and (node == 0 or len(gdf) < MIN_PIX):
                continue
            out.append({"reach_id": rid, "node_id": int(node),
                        "n_pix": int(len(gdf)),
                        "coh_median": float(np.nanmedian(gdf.gamma)),
                        "coh_p25": float(np.nanpercentile(gdf.gamma, 25)),
                        "sig0_median": float(np.nanmedian(gdf.sig0)),
                        "bright_incoh_pct": float(100 * gdf.bi.mean()),
                        "lowcoh_pct": float(100 * gdf.lowcoh.mean()),
                        "layover_pct": float(100 * gdf.lay.mean()),
                        "layover_impact_median": float(np.nanmedian(gdf.li)),
                        "xtrk_km": float(np.nanmedian(gdf.xt))})
    return out


def extract(shard, nshards, workers):
    import earthaccess
    import geopandas as gpd
    from concurrent.futures import ThreadPoolExecutor

    earthaccess.login(strategy="netrc")
    m = pd.read_parquet(MANIFEST)
    g = gpd.read_parquet(REACHES)[["reach_id", "geometry"]]
    g["reach_id"] = g.reach_id.astype("int64")
    g = g.drop_duplicates("reach_id").set_index("reach_id")
    by_gran = m.groupby("granule")
    names = sorted(by_gran.groups)[shard::nshards]
    scratch = os.environ.get("SLURM_TMPDIR", "/tmp") + f"/pixc_{shard}"
    os.makedirs(scratch, exist_ok=True)
    done_path = f"{OUT}/done_shard{shard:02d}.txt"
    done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()
    print(f"shard {shard}: {len(names)} granules, {len(done)} already done", flush=True)

    rows, t0, nbytes = [], time.time(), 0

    def fetch(name):
        """Pixel cloud plus its vector attribute file, which supplies node identity
        and costs about one percent of the pixel cloud in bytes."""
        sub = by_gran.get_group(name)
        url = sub.url.iloc[0]
        try:
            f = earthaccess.download([url], scratch, threads=1)
            pixc = f[0] if f else None
        except Exception as ex:
            print(f"  {name}: download failed {ex}", flush=True)
            return name, None, None
        vec = None
        try:
            v = earthaccess.download([url.replace("PIXC", "PIXCVec")], scratch, threads=1)
            vec = v[0] if v else None
        except Exception:
            vec = None
        return name, pixc, vec

    todo = [n for n in names if n not in done]
    with ThreadPoolExecutor(max_workers=workers) as ex, open(done_path, "a") as dh:
        for i, (name, path, vec) in enumerate(ex.map(fetch, todo), 1):
            if path and os.path.exists(path):
                nbytes += os.path.getsize(path)
                rids = by_gran.get_group(name).reach_id.unique()
                polys = [(int(r), g.loc[r].geometry) for r in rids if r in g.index]
                try:
                    st = granule_node_stats(path, polys, vec)
                except Exception as exc:
                    print(f"  {name}: {type(exc).__name__}: {exc}", flush=True)
                    st = None
                if st:
                    parts = name.split("_")
                    for rec in st:
                        rec["granule"] = name
                        rec["t"] = parts[7] if len(parts) > 7 else ""
                    rows.extend(st)
                os.remove(path)
                if vec and os.path.exists(vec):
                    os.remove(vec)
            dh.write(name + "\n"); dh.flush()
            if i % 20 == 0:
                el = time.time() - t0
                print(f"  {i}/{len(todo)} granules, {nbytes/1073741824:.1f} GB, "
                      f"{nbytes/1048576/el:.0f} MB/s, {len(rows)} node rows", flush=True)
                if rows:
                    pd.DataFrame(rows).to_parquet(
                        f"{OUT}/coh_shard{shard:02d}.parquet", index=False)
                gc.collect()
    if rows:
        pd.DataFrame(rows).to_parquet(f"{OUT}/coh_shard{shard:02d}.parquet", index=False)
    shutil.rmtree(scratch, ignore_errors=True)
    print(f"shard {shard}: {len(rows)} node rows, {nbytes/1073741824:.1f} GB streamed")


def merge():
    files = sorted(glob.glob(f"{OUT}/coh_shard*.parquet"))
    if not files:
        raise SystemExit("no shards")
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    d.to_parquet(f"{OUT}/coherence_all_reaches.parquet", index=False)
    print(f"merged {len(d):,} reach-pass rows, {d.reach_id.nunique()} reaches, "
          f"{d.granule.nunique():,} granules")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["scope", "extract", "merge"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.task == "scope":
        scope()
    elif a.task == "extract":
        extract(a.shard, a.nshards, a.workers)
    else:
        merge()
