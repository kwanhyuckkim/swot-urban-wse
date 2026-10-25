#!/usr/bin/env python
"""
Version-D SWOT pull for the non-urban control group (TGRS-2026-02661).

The legacy pairing file (paired_reach_SWOT_gage.csv) holds RiverSP 2.0 data, so a
control group built from it is not comparable with the paper's Version D numbers.
This script rebuilds the control on Version D: it lists every gauged reach the legacy
pairing knows about, tags the published urban (reach, gauge) pairs, and pulls the
Version D reach series from Hydrocron with every field the manuscript's Table 1
filters need.

Usage:
  python 08_control_reaches_pull.py list
  python 08_control_reaches_pull.py hydrocron --shard 0 --nshards 8 --workers 12
  python 08_control_reaches_pull.py merge --nshards 8
"""
import argparse
import glob
import io
import os
import time

import numpy as np
import pandas as pd
import requests
import pyarrow.parquet as pq

URL = "https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries"
COLLECTION = "SWOT_L2_HR_RiverSP_D"
START, END = "2023-07-26T00:00:00Z", "2025-12-31T23:59:59Z"
WORK = "/work/pi_kandread_umass_edu/swot-urban"
OUT_DIR = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results/control_versionD"
PAIRED = f"{WORK}/0_data_collection/5_sword_usgs/paired_reach_SWOT_gage.csv"
BUFFERED = f"{WORK}/1_data_processing/buffered_polygons_with_nlcd.parquet"
URBAN_OBS = f"{WORK}/2_urban_reaches/all_final_shape.parquet"
FIELDS = ("reach_id,time_str,wse,reach_q,reach_q_b,dark_frac,ice_clim_f,xovr_cal_q,"
          "partial_f,n_good_nod,p_n_nodes,p_width,p_length,p_wse,xtrk_dist,crid")
FILL = -999999999999.0


def build_list():
    pairs = pd.read_csv(PAIRED, usecols=["reach_id", "gage_id"], dtype={"gage_id": str})
    pairs["reach_id"] = pairs.reach_id.astype("int64")
    pairs["gage_id"] = pairs.gage_id.str.zfill(8)
    pairs = pairs.drop_duplicates()
    buf = pq.read_table(BUFFERED, columns=["reach_id", "fctimp_mean", "lakeflag"]).to_pandas()
    buf["reach_id"] = buf.reach_id.astype("int64")
    pairs = pairs.merge(buf.drop_duplicates("reach_id"), on="reach_id", how="left")
    pairs = pairs[pairs.lakeflag.fillna(0) == 0]

    pub = pq.read_table(URBAN_OBS, columns=["reach_id", "site_no"]).to_pandas()
    pub["reach_id"] = pub.reach_id.astype("int64")
    pub["gage_id"] = pub.site_no.astype(str).str.zfill(8)
    pub_pairs = pub[["reach_id", "gage_id"]].drop_duplicates()
    pub_pairs["group"] = "urban409"

    pairs = pairs.merge(pub_pairs, on=["reach_id", "gage_id"], how="outer")
    pairs["group"] = pairs.group.fillna("candidate")
    # published pairs may be missing from the legacy file; carry them regardless
    pairs = pairs.dropna(subset=["reach_id"])
    pairs["reach_id"] = pairs.reach_id.astype("int64")
    os.makedirs(OUT_DIR, exist_ok=True)
    pairs.to_csv(f"{OUT_DIR}/control_pairs.csv", index=False)
    print(f"pairs {len(pairs):,}  reaches {pairs.reach_id.nunique():,}  "
          f"urban409 pairs {(pairs.group == 'urban409').sum():,}  "
          f"fctimp<2 reaches {pairs[pairs.fctimp_mean < 2].reach_id.nunique():,}")


def query(fid, retries=3):
    params = {"feature": "Reach", "feature_id": str(fid), "output": "csv",
              "start_time": START, "end_time": END, "fields": FIELDS,
              "collection_name": COLLECTION}
    for attempt in range(retries):
        try:
            r = requests.get(URL, params=params, timeout=90)
            j = r.json()
            if "results" in j and j["results"].get("csv"):
                return pd.read_csv(io.StringIO(j["results"]["csv"]))
            return None
        except Exception:
            time.sleep(2 ** attempt)
    return None


def pull(shard, nshards, workers):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    ids = sorted(pd.read_csv(f"{OUT_DIR}/control_pairs.csv").reach_id.unique())
    ids = [int(i) for i in ids][shard::nshards]
    parts, got = [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(query, i): i for i in ids}
        for n, f in enumerate(as_completed(futures), 1):
            d = f.result()
            if d is not None and len(d):
                parts.append(d)
                got += 1
            if n % 100 == 0:
                print(f"  shard {shard}: {n}/{len(ids)} queried, {got} with data", flush=True)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    out.to_parquet(f"{OUT_DIR}/swot_reach_D_shard{shard:02d}.parquet", index=False)
    print(f"shard {shard}: {len(ids)} reaches, {got} with data, {len(out):,} rows")


def merge(nshards):
    files = sorted(glob.glob(f"{OUT_DIR}/swot_reach_D_shard*.parquet"))
    if len(files) < nshards:
        raise SystemExit(f"only {len(files)} of {nshards} shards present")
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d.loc[d[c] <= FILL / 10, c] = np.nan
    d.to_parquet(f"{OUT_DIR}/swot_reach_D.parquet", index=False)
    print(f"merged {len(d):,} rows, {d.reach_id.nunique():,} reaches "
          f"-> {OUT_DIR}/swot_reach_D.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["list", "hydrocron", "merge"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=8)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    if a.task == "list":
        build_list()
    elif a.task == "hydrocron":
        pull(a.shard, a.nshards, a.workers)
    else:
        merge(a.nshards)
