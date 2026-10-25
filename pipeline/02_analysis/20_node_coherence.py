#!/usr/bin/env python
"""
Normalized coherence carried down to the node, which is the scale this study works at.

The pass-level version answered Brent's question about whether coherence beats coherent
power, but it compared a pass median against a reach error. The pixel cloud vector
product assigns every water pixel to a node, so the same coherence can be summarised
per node and matched to that node's own error. That turns 166 paired points into tens
of thousands and lets the coherence be compared against the node quality bits and our
geometric screen on one axis.

    gain  = coherent_power / sqrt(power_plus_y * power_minus_y)     (1 to 2, PD 3 dB cap)
    gamma = gain - 1                                                 coherence magnitude

  python 20_node_coherence.py extract --site-index 0
  python 20_node_coherence.py merge
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

WORK = "/work/pi_kandread_umass_edu/swot-urban"
PROJ = "/project/pi_kandread_umass_edu/swot-urban"
OUT_DIR = f"{PROJ}/3_analysis_results/reviewer_checks"
NODES = f"{PROJ}/3_analysis_results/swot_urban_nearfar_v6.5_results.parquet"
LOW_COH_CLASSES = (6, 7)
LAYOVER_BIT = 1
MIN_PIX = 20


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


def node_stats(pixc_file, vec_file):
    """Per-node coherence and brightness for one granule."""
    import xarray as xr

    v = xr.open_dataset(vec_file)
    nid_raw = v["node_id"].values
    v.close()
    nid = pd.Series(nid_raw).astype(str).str.strip()
    have = nid.str.isdigit().values
    if have.sum() < MIN_PIX:
        return None

    ds = xr.open_dataset(pixc_file, group="pixel_cloud")
    n_pix = ds.sizes.get("points", 0)
    if n_pix != nid_raw.size:
        ds.close()
        return None

    def g(name):
        return ds[name].values.astype("float64") if name in ds else np.full(n_pix, np.nan)

    cls = ds["classification"].values
    cp, pp, pm = g("coherent_power"), g("power_plus_y"), g("power_minus_y")
    sig0, gq = g("sig0"), g("geolocation_qual")
    li, xt = g("layover_impact"), np.abs(g("cross_track")) / 1000.0
    ds.close()

    denom = np.sqrt(np.clip(pp, 0, None) * np.clip(pm, 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma = np.where(denom > 0, cp / denom, np.nan)
    gamma = np.clip(gamma, 0, 2) - 1.0

    sel = have & (cls > 2) & np.isfinite(gamma) & np.isfinite(sig0)
    if sel.sum() < MIN_PIX:
        return None

    df = pd.DataFrame({
        "node_id": nid[sel].astype("int64").values,
        "gamma": gamma[sel], "sig0": sig0[sel],
        "lay": (np.nan_to_num(gq[sel]).astype("int64") & LAYOVER_BIT) > 0,
        "lowcoh": np.isin(cls[sel], LOW_COH_CLASSES),
        "li": li[sel], "xt": xt[sel],
    })
    # Brent's signature is relative to brightness, so fit coherence on log sig0
    # across the granule and keep the negative residual tail.
    ls = np.log10(np.clip(df.sig0.values, 1e-3, None))
    ok = np.isfinite(ls) & np.isfinite(df.gamma.values)
    if ok.sum() > 100 and np.nanstd(ls[ok]) > 1e-6:
        sl, ic = np.polyfit(ls[ok], df.gamma.values[ok], 1)
        df["resid"] = df.gamma - (sl * ls + ic)
    else:
        df["resid"] = np.nan
    bright_cut = np.nanpercentile(df.sig0, 75)
    resid_cut = np.nanpercentile(df.resid.dropna(), 25) if df.resid.notna().sum() > 50 else -np.inf
    df["bright_incoh"] = (df.sig0 >= bright_cut) & (df.resid <= resid_cut)

    grp = df.groupby("node_id")
    out = grp.agg(n_pix=("gamma", "size"),
                  coh_median=("gamma", "median"),
                  coh_p25=("gamma", lambda s: float(np.nanpercentile(s, 25))),
                  sig0_median=("sig0", "median"),
                  bright_incoh_pct=("bright_incoh", lambda s: 100 * float(np.mean(s))),
                  lowcoh_pct=("lowcoh", lambda s: 100 * float(np.mean(s))),
                  layover_pct=("lay", lambda s: 100 * float(np.mean(s))),
                  layover_impact_median=("li", "median"),
                  xtrk_km=("xt", "median")).reset_index()
    return out[out.n_pix >= MIN_PIX]


def run_extract(site_index):
    import geopandas as gpd
    dirs = sorted(glob.glob(f"{PROJ}/0_data_collection/SWOT_PIXC/fctimp_*"))
    if site_index >= len(dirs):
        print(f"site index {site_index} beyond {len(dirs)}"); return
    d = dirs[site_index]
    isa = os.path.basename(d).replace("fctimp_", "")
    vec_dir = f"{PROJ}/0_data_collection/SWOT_PIXCVec/fctimp_{isa}"
    files = best_versions(glob.glob(f"{d}/SWOT_L2_HR_PIXC_*.nc"))
    print(f"site {isa}: {len(files)} granules", flush=True)

    rows = []
    for i, f in enumerate(files, 1):
        vec = os.path.join(vec_dir, os.path.basename(f).replace("PIXC_", "PIXCVec_"))
        if not os.path.exists(vec):
            continue
        try:
            s = node_stats(f, vec)
        except Exception as ex:
            print(f"  {os.path.basename(f)}: {type(ex).__name__}: {ex}", flush=True)
            continue
        if s is None or not len(s):
            continue
        parts = os.path.basename(f).split("_")
        try:
            t = pd.to_datetime(parts[7], format="%Y%m%dT%H%M%S")
        except Exception:
            continue
        s["t"] = t.floor("h")
        s["granule"] = os.path.basename(f)
        rows.append(s)
        if i % 20 == 0:
            print(f"  {i}/{len(files)} granules, {sum(len(r) for r in rows):,} node rows",
                  flush=True)
    if not rows:
        print("nothing extracted"); return
    out = pd.concat(rows, ignore_index=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    p = f"{OUT_DIR}/node_coherence_site{site_index:02d}.parquet"
    out.to_parquet(p, index=False)
    print(f"site {isa}: {len(out):,} node observations -> {p}")


def run_merge():
    import pyarrow.parquet as pq
    from scipy import stats as st
    from statsmodels.stats.proportion import proportion_confint

    files = sorted(glob.glob(f"{OUT_DIR}/node_coherence_site*.parquet"))
    if not files:
        raise SystemExit("no shards")
    c = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    c["t"] = pd.to_datetime(c.t, utc=True).dt.floor("h")
    c = c.drop_duplicates(["node_id", "t"])

    d = pq.read_table(NODES, columns=["reach_id", "node_id", "time_str", "flow_angle",
                                      "far_isa", "far_building", "error_node"]).to_pandas()
    d["node_id"] = d.node_id.astype("int64")
    d["e"] = d.error_node.abs()
    fa = d.flow_angle % 360
    d["ang"] = np.minimum.reduce([fa, (fa - 180).abs(), 360 - fa])
    d["isa"] = (d.far_isa * 100 if d.far_isa.max() <= 1 else d.far_isa).fillna(0)
    d["bld"] = d.far_building.fillna(0)
    d["t"] = pd.to_datetime(d.time_str, errors="coerce", utc=True).dt.floor("h")
    m = d.dropna(subset=["e", "t"]).merge(c, on=["node_id", "t"], how="inner")
    m.to_parquet(f"{OUT_DIR}/node_coherence.parquet", index=False)
    T = 0.1832
    base = (m.e >= T).mean() * 100
    print(f"joined {len(m):,} node observations, {m.reach_id.nunique()} reaches, "
          f"{len(files)} sites | high-error base {base:.1f}%\n")

    def rho(a, b, lab):
        k = pd.notna(a) & pd.notna(b)
        r, p = st.spearmanr(a[k], b[k])
        print(f"  {lab:44s} rho {r:+.4f} (p={p:.1e}, n={k.sum():,})")

    print("[1] node coherence against node error")
    rho(m.coh_median, m.e, "normalized coherence (Brent)")
    rho(m.coh_p25, m.e, "coherence lower quartile")
    rho(m.lowcoh_pct, m.e, "low-coherence class share (shipped)")
    rho(m.layover_impact_median, m.e, "layover_impact (shipped)")
    rho(m.sig0_median, m.e, "sig0 (brightness alone)")
    rho(m.bld, m.e, "far building height (ours)")
    rho(m.ang, m.e, "angle deviation (ours)")

    print("\n[2] the bright-but-incoherent signature at node scale")
    rho(m.bright_incoh_pct, m.e, "bright-and-incoherent pixel share")
    rho(m.bright_incoh_pct, m.isa, "  same, against far-bank ISA")

    print("\n[3] screening power on one axis")

    def line(lab, mask):
        n = int(mask.sum())
        if n < 100:
            print(f"  {lab:44s} n={n}"); return
        s = (m.e[mask] >= T).mean() * 100
        lo, hi = proportion_confint(int((m.e[mask] >= T).sum()), n, method="wilson")
        print(f"  {lab:44s} cover {100*n/len(m):5.2f}%  high-error {s:5.1f}%  "
              f"lift {s/base:.2f}  [{100*lo:.1f}, {100*hi:.1f}]")

    line("coherence in the lowest quartile", m.coh_median <= m.coh_median.quantile(.25))
    line("coherence in the lowest decile", m.coh_median <= m.coh_median.quantile(.10))
    line("bright-and-incoherent share above 10%", m.bright_incoh_pct > 10)
    line("low-coherence class above 50% of pixels", m.lowcoh_pct > 50)
    scr = (m.ang <= 13) & ((m.bld > 8) | (m.isa > 19))
    line("our geometric screen", scr)
    line("screen and low coherence together", scr & (m.coh_median <= m.coh_median.quantile(.25)))

    print("\n[4] cross-track consistency at node scale")
    for lo_, hi_ in [(10, 20), (20, 30), (30, 40), (40, 50), (50, 60)]:
        g = m[(m.xtrk_km >= lo_) & (m.xtrk_km < hi_)]
        if len(g) < 200:
            continue
        print(f"  {lo_:>2}-{hi_:<2} km: coherence {g.coh_median.median():.3f} | "
              f"low-coh class {g.lowcoh_pct.median():5.1f}% | "
              f"high-error {100*(g.e >= T).mean():5.1f}% | n={len(g):,}")
    print(f"\nsaved {OUT_DIR}/node_coherence.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["extract", "merge"])
    ap.add_argument("--site-index", type=int, default=0)
    a = ap.parse_args()
    run_extract(a.site_index) if a.task == "extract" else run_merge()
