#!/usr/bin/env python
"""
Pull the shipped layover estimate from Hydrocron for the study reaches.

RiverSP Product Description D-56413 Rev C:
  layovr_val (p.23) estimate of the systematic WSE error due to layover, from
    models and reference data rather than from the SWOT measurement
  wse_u, wse_r_u (p.20) total and random-only uncertainty; the systematic
    component follows as sqrt(wse_u^2 - wse_r_u^2)

Node and reach levels are both available, which is what makes the pixel to node
to reach comparison possible. The node list is large, so the work is shardable:
each shard queries every nth node and writes its own file, and the shards are
merged afterwards.

Usage:
  python 07_hydrocron_layovr_val.py reach
  python 07_hydrocron_layovr_val.py node --shard 0 --nshards 8 --workers 12
  python 07_hydrocron_layovr_val.py merge --nshards 8
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
FILL = -999999999999.0
WORK = "/work/pi_kandread_umass_edu/swot-urban"
OUT_DIR = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results/reviewer_checks"
REACH_FIELDS = ("reach_id,time_str,wse,wse_u,wse_r_u,layovr_val,reach_q,reach_q_b,"
                "dark_frac,xtrk_dist,width,slope2,crid")
NODE_FIELDS = ("reach_id,node_id,time_str,wse,wse_u,wse_r_u,layovr_val,node_q,node_q_b,"
               "dark_frac,xtrk_dist,flow_angle,width,crid")


def study_reaches():
    return (pq.read_table(f"{WORK}/2_urban_reaches/all_reaches_vD.parquet",
                          columns=["reach_id"]).to_pandas()
            .reach_id.astype("int64").drop_duplicates().tolist())


def study_nodes():
    nodes = pd.read_parquet(f"{WORK}/2_urban_reaches/selected_swot_vD_nodeonly_ver2.parquet",
                            columns=["reach_id", "node_id"])
    nodes["reach_id"] = nodes.reach_id.astype("int64")
    keep = set(study_reaches())
    return (nodes[nodes.reach_id.isin(keep)].node_id.astype("int64")
            .drop_duplicates().tolist())


def query(feature, fid, fields, retries=3):
    params = {"feature": feature, "feature_id": str(fid), "output": "csv",
              "start_time": START, "end_time": END, "fields": fields,
              "collection_name": COLLECTION}
    for attempt in range(retries):
        try:
            r = requests.get(URL, params=params, timeout=60)
            j = r.json()
            if "results" in j and j["results"].get("csv"):
                return pd.read_csv(io.StringIO(j["results"]["csv"]))
            return None
        except Exception:
            time.sleep(2 ** attempt)
    return None


def pull(feature, ids, fields, out_path, workers=12, report_every=500):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    parts, got = [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(query, feature, i, fields): i for i in ids}
        for n, f in enumerate(as_completed(futures), 1):
            d = f.result()
            if d is not None and len(d):
                parts.append(d)
                got += 1
            if n % report_every == 0:
                print(f"{n}/{len(ids)} queried, {got} with data", flush=True)
    if not parts:
        print("no data returned", flush=True)
        return None
    df = pd.concat(parts, ignore_index=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"wrote {len(df):,} rows to {out_path}", flush=True)
    return df


def summarize(df, label):
    for c in ("wse_u", "wse_r_u", "layovr_val"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").replace(FILL, np.nan)
    sysc = np.sqrt(np.clip(df.wse_u ** 2 - df.wse_r_u ** 2, 0, None))
    print(f"\n{label}: {len(df):,} observations")
    print(f"  layovr_val median {df.layovr_val.median():.3f} m, "
          f"p75 {df.layovr_val.quantile(.75):.3f} m, valid {df.layovr_val.notna().sum():,}")
    print(f"  systematic uncertainty sqrt(wse_u^2 - wse_r_u^2) median {sysc.median():.4f} m")
    m = df.layovr_val.notna() & sysc.notna() & (sysc > 0)
    if m.sum() > 10:
        from scipy import stats as st
        r, p = st.spearmanr(df.layovr_val[m], sysc[m])
        print(f"  layovr_val vs that uncertainty: Spearman {r:+.3f} (p={p:.2e}, n={int(m.sum()):,})")
        print(f"  ratio median {(df.layovr_val[m] / sysc[m]).median():.1f}, "
              f"layovr_val larger in {(df.layovr_val[m] > sysc[m]).mean() * 100:.1f}% of cases")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("level", choices=["reach", "node", "merge"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    if a.level == "reach":
        df = pull("Reach", study_reaches(), REACH_FIELDS,
                  f"{OUT_DIR}/layovr_val_reach.parquet", workers=a.workers)
        if df is not None:
            summarize(df, "reach level")
    elif a.level == "node":
        ids = study_nodes()[a.shard::a.nshards]
        print(f"shard {a.shard}/{a.nshards}: {len(ids):,} nodes, {a.workers} workers", flush=True)
        pull("Node", ids, NODE_FIELDS,
             f"{OUT_DIR}/layovr_val_node_shard{a.shard:02d}.parquet", workers=a.workers)
    else:
        files = sorted(glob.glob(f"{OUT_DIR}/layovr_val_node_shard*.parquet"))
        print(f"merging {len(files)} shards")
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df = df.drop_duplicates(subset=["node_id", "time_str"])
        df.to_parquet(f"{OUT_DIR}/layovr_val_node.parquet", index=False)
        print(f"wrote {len(df):,} rows, {df.node_id.nunique():,} nodes")
        summarize(df, "node level")
