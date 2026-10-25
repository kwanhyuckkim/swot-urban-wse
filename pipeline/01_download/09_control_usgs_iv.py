#!/usr/bin/env python
"""
USGS instantaneous gage-height pull for the Version-D control group.

For every gauge in control_pairs.csv whose reach returned Version D data, pull the
15-minute 00065 record over the study period from the NWIS IV service, keep only the
readings that fall within 90 minutes of one of that reach's SWOT observation times,
convert feet to meters, and write one shard per task.

Usage:
  python 09_control_usgs_iv.py --shard 0 --nshards 8 --workers 8
  python 09_control_usgs_iv.py --merge
"""
import argparse
import glob
import io
import os
import time

import numpy as np
import pandas as pd
import requests

OUT_DIR = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results/control_versionD"
NWIS = "https://waterservices.usgs.gov/nwis/iv/"
START, END = "2023-07-01", "2025-12-31"
FT_TO_M = 0.3048


def load_targets():
    pairs = pd.read_csv(f"{OUT_DIR}/control_pairs.csv", dtype={"gage_id": str})
    pairs["gage_id"] = pairs.gage_id.str.zfill(8)
    swot = pd.read_parquet(f"{OUT_DIR}/swot_reach_D.parquet",
                           columns=["reach_id", "time_str"])
    swot = swot[swot.time_str.astype(str).str.startswith("20")]
    swot["t"] = pd.to_datetime(swot.time_str, errors="coerce", utc=True)
    swot = swot.dropna(subset=["t"])
    m = pairs.merge(swot, on="reach_id")
    times_by_site = m.groupby("gage_id").t.apply(lambda s: np.sort(s.unique()))
    return times_by_site


def pull_site(site, times, retries=3):
    params = {"format": "rdb", "sites": site, "parameterCd": "00065",
              "startDT": START, "endDT": END}
    txt = None
    for attempt in range(retries):
        try:
            r = requests.get(NWIS, params=params, timeout=180)
            if r.status_code == 200 and "agency_cd" in r.text:
                txt = r.text
                break
            if r.status_code in (404, 400):
                return None
        except Exception:
            time.sleep(2 ** attempt)
    if txt is None:
        return None
    body = [l for l in txt.splitlines() if not l.startswith("#")]
    d = pd.read_csv(io.StringIO("\n".join(body)), sep="\t", dtype=str)
    d = d[d.agency_cd == "USGS"]
    vcol = next((c for c in d.columns if c.endswith("_00065")), None)
    if vcol is None or "datetime" not in d.columns or "tz_cd" not in d.columns:
        return None
    dt = pd.to_datetime(d.datetime, errors="coerce")
    off = d.tz_cd.map({"EST": -5, "EDT": -4, "CST": -6, "CDT": -5, "MST": -7,
                       "MDT": -6, "PST": -8, "PDT": -7, "AKST": -9, "AKDT": -8,
                       "HST": -10, "UTC": 0, "GMT": 0})
    d = d.assign(t=dt, off=off, stage_ft=pd.to_numeric(d[vcol], errors="coerce"))
    d = d.dropna(subset=["t", "off", "stage_ft"])
    d["t_utc"] = d.t.dt.tz_localize(None) - pd.to_timedelta(d.off, unit="h")
    d["t_utc"] = d.t_utc.dt.tz_localize("UTC")
    tv = pd.DatetimeIndex(times)
    tt = pd.DatetimeIndex(d.t_utc)
    if not len(tt) or not len(tv):
        return None
    pos = np.searchsorted(tv.asi8, tt.asi8)
    pos_lo = np.clip(pos - 1, 0, len(tv) - 1)
    pos_hi = np.clip(pos, 0, len(tv) - 1)
    near = np.minimum(np.abs(tt.asi8 - tv.asi8[pos_lo]),
                      np.abs(tt.asi8 - tv.asi8[pos_hi]))
    keep = near <= 90 * 60 * 1_000_000_000
    d = d[keep]
    if not len(d):
        return None
    return pd.DataFrame({"gage_id": site, "t_utc": d.t_utc.values,
                         "stage_m": d.stage_ft.values * FT_TO_M})


def run(shard, nshards, workers):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    targets = load_targets()
    sites = sorted(targets.index)[shard::nshards]
    parts, got = [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(pull_site, s, targets[s]): s for s in sites}
        for n, f in enumerate(as_completed(futures), 1):
            d = f.result()
            if d is not None and len(d):
                parts.append(d)
                got += 1
            if n % 50 == 0:
                print(f"  shard {shard}: {n}/{len(sites)} sites, {got} with data", flush=True)
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    out.to_parquet(f"{OUT_DIR}/usgs_iv_shard{shard:02d}.parquet", index=False)
    print(f"shard {shard}: {len(sites)} sites, {got} with data, {len(out):,} readings")


def merge():
    files = sorted(glob.glob(f"{OUT_DIR}/usgs_iv_shard*.parquet"))
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    d.to_parquet(f"{OUT_DIR}/usgs_iv.parquet", index=False)
    print(f"merged {len(d):,} readings, {d.gage_id.nunique():,} sites")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        merge()
    else:
        run(a.shard, a.nshards, a.workers)
