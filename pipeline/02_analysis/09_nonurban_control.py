#!/usr/bin/env python
"""
Non-urban control group for TGRS-2026-02661.

The gauge-matched urban dataset shows a 51.1% share of observations above the SWOT
science requirement (T = 0.1832 m). This script asks whether that share is an urban
feature or an artifact of our pairing pipeline, by pushing low-impervious gauged
reaches through the same code path under the manuscript's Table 1 filters: one-hour
window, median normalization, the eleven inclusion criteria, a Hampel outlier pass,
and a five-observation minimum per series.

The 409 published study reaches are recomputed here restricted to their published
(reach, gauge) pairs, so the recomputation can be checked against the published
12,145 / 51.1%. Control reaches are gauged reaches whose corridor mean fractional
impervious cover sits below 2%; a dose-response table across fctimp bins backs the
binary contrast.
"""
import os
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from scipy import stats as st
from statsmodels.stats.proportion import proportion_confint

WORK = "/work/pi_kandread_umass_edu/swot-urban"
PROJ = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results"
OUT = f"{PROJ}/round2"
PAIRED = f"{WORK}/0_data_collection/5_sword_usgs/paired_reach_SWOT_gage.csv"
BUFFERED = f"{WORK}/1_data_processing/buffered_polygons_with_nlcd.parquet"
URBAN_OBS = f"{WORK}/2_urban_reaches/all_final_shape.parquet"
T = 0.1832
MIN_PAIRS = 5

COLS = ["gage_id", "gage_time_UTC", "gage_stage_m", "reach_id", "SWOT_time", "wse",
        "reach_q", "reach_q_b", "partial_f", "ice_clim_f", "xovr_cal_q", "dark_frac",
        "xtrk_dist", "n_good_nod", "p_n_nodes", "p_width", "p_length", "p_wse"]


def hampel_mask(x, window=11, k=3.0):
    med = x.rolling(window, center=True, min_periods=1).median()
    mad = (x - med).abs().rolling(window, center=True, min_periods=1).median()
    return (x - med).abs() <= k * 1.4826 * mad.replace(0, np.nan).fillna(np.inf)


def load_pairs():
    d = pd.read_csv(PAIRED, usecols=COLS, dtype={"gage_id": str})
    d["reach_id"] = d.reach_id.astype("int64")
    d["gage_id"] = d.gage_id.str.zfill(8)
    gt = pd.to_datetime(d.gage_time_UTC, errors="coerce", utc=True)
    stt = pd.to_datetime(d.SWOT_time, errors="coerce", utc=True)
    d["dt_h"] = (gt - stt).dt.total_seconds().abs() / 3600
    n0 = len(d)
    d = d[(d.dt_h <= 1.0) & d.wse.between(-1000, 9000) & d.gage_stage_m.notna()]
    # Table 1 inclusion criteria, in manuscript order
    d = d[(d.reach_q < 3) & (d.partial_f == 0) & (d.ice_clim_f == 0)
          & (d.xovr_cal_q < 2) & (d.reach_q_b < 4_194_304) & (d.dark_frac <= 0.5)
          & d.xtrk_dist.abs().between(10_000, 60_000)
          & ((d.n_good_nod / d.p_n_nodes.replace(0, np.nan)) > 0.5)
          & (d.p_width >= 80) & (d.p_length >= 7_000)
          & ((d.wse - d.p_wse).abs() < 10)]
    d = d.sort_values("SWOT_time")
    key = d.reach_id.astype(str) + "_" + d.gage_id
    d = d[d.groupby(key, group_keys=False).wse.apply(hampel_mask)]
    key = d.reach_id.astype(str) + "_" + d.gage_id
    d["wse_rel"] = d.wse - d.groupby(key).wse.transform("median")
    d["stage_rel"] = d.gage_stage_m - d.groupby(key).gage_stage_m.transform("median")
    d["npair"] = d.groupby(key).wse.transform("size")
    d = d[d.npair >= MIN_PAIRS]
    d["e"] = (d.wse_rel - d.stage_rel).abs()
    d = d[d.e <= 100]
    print(f"rows {n0:,} -> after window+filters+hampel+min{MIN_PAIRS} {len(d):,}")
    return d


def p68(x):
    return float(np.nanpercentile(x, 68)) if len(x) else np.nan


def row(name, g):
    n = len(g)
    hi = int((g.e >= T).sum())
    lo, hiCI = proportion_confint(hi, n, method="wilson") if n else (np.nan, np.nan)
    return {"group": name, "n": n, "reaches": g.reach_id.nunique(),
            "median_m": float(g.e.median()), "p68_m": p68(g.e),
            "share_over_T_pct": 100 * hi / n if n else np.nan,
            "ci_lo": 100 * lo, "ci_hi": 100 * hiCI}


def run():
    d = load_pairs()
    buf = pq.read_table(BUFFERED, columns=["reach_id", "fctimp_mean", "lakeflag"]).to_pandas()
    buf["reach_id"] = buf.reach_id.astype("int64")
    d = d.merge(buf.drop_duplicates("reach_id"), on="reach_id", how="left")
    d = d[d.lakeflag.fillna(0) == 0]

    pub = pq.read_table(URBAN_OBS, columns=["reach_id", "site_no", "wse_relative_m",
                                            "wse_usgs_relative_m"]).to_pandas()
    pub["e"] = (pub.wse_relative_m - pub.wse_usgs_relative_m).abs()
    pub["reach_id"] = pub.reach_id.astype("int64")
    pub_pairs = set(zip(pub.reach_id, pub.site_no.astype(str).str.zfill(8)))
    urb = d[[p in pub_pairs for p in zip(d.reach_id, d.gage_id)]]

    print(f"paired obs after QA {len(d):,}  reaches {d.reach_id.nunique():,}  "
          f"with fctimp {d.fctimp_mean.notna().mean()*100:.1f}%")

    rows = [row("published urban file (reference)", pub),
            row("published (reach, gauge) pairs, recomputed", urb),
            row("control  fctimp < 2%", d[d.fctimp_mean < 2]),
            row("intermediate  2-10%", d[d.fctimp_mean.between(2, 10)]),
            row("urban-like  fctimp > 10%", d[d.fctimp_mean > 10])]
    out = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    out.to_csv(f"{OUT}/nonurban_control.csv", index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    ctrl = d[d.fctimp_mean < 2]
    if len(ctrl) and len(urb):
        mw = st.mannwhitneyu(ctrl.e, urb.e, alternative="two-sided")
        ks = st.ks_2samp(ctrl.e, urb.e)
        print(f"\ncontrol vs recomputed urban pairs  Mann-Whitney p {mw.pvalue:.2e}  "
              f"KS D {ks.statistic:.3f} p {ks.pvalue:.2e}")

    bins = [0, 1, 2, 5, 10, 20, 40, 100]
    d["bin"] = pd.cut(d.fctimp_mean, bins, right=False)
    dose = []
    for b, g in d.groupby("bin", observed=True):
        if len(g) < 200:
            continue
        dose.append({"fctimp_bin": str(b), "n": len(g), "reaches": g.reach_id.nunique(),
                     "p68_m": p68(g.e), "share_over_T_pct": 100 * (g.e >= T).mean()})
    dd = pd.DataFrame(dose)
    dd.to_csv(f"{OUT}/fctimp_dose_response.csv", index=False)
    print("\ndose response by impervious fraction (Table 1 filters applied)")
    print(dd.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    rr = d.dropna(subset=["fctimp_mean"]).groupby("reach_id").agg(
        f=("fctimp_mean", "first"), q=("e", p68))
    r, p = st.spearmanr(rr.f, rr.q)
    print(f"\nreach-level Spearman(fctimp, p68 error) {r:+.3f} (p={p:.2e}, n={len(rr)})")
    print(f"\nsaved {OUT}/nonurban_control.csv and fctimp_dose_response.csv")


if __name__ == "__main__":
    run()
