#!/usr/bin/env python
"""
Version-D non-urban control analysis for TGRS-2026-02661.

Matches the Version D SWOT reach series to 15-minute USGS gage heights within one
hour, applies the manuscript's Table 1 inclusion criteria, a Hampel outlier pass,
median normalization per (reach, gauge) series, and a five-observation minimum, then
compares the published urban pairs (recomputed end to end as a pipeline check against
the published 12,145 / 51.1%) with low-impervious control reaches.
"""
import os
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from scipy import stats as st
from statsmodels.stats.proportion import proportion_confint

WORK = "/work/pi_kandread_umass_edu/swot-urban"
DIR = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results/control_versionD"
OUT = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results/round2"
URBAN_OBS = f"{WORK}/2_urban_reaches/all_final_shape.parquet"
T = 0.1832
MIN_PAIRS = 5


def hampel_mask(x, window=11, k=3.0):
    med = x.rolling(window, center=True, min_periods=1).median()
    mad = (x - med).abs().rolling(window, center=True, min_periods=1).median()
    return (x - med).abs() <= k * 1.4826 * mad.replace(0, np.nan).fillna(np.inf)


def p68(x):
    return float(np.nanpercentile(x, 68)) if len(x) else np.nan


def load():
    s = pd.read_parquet(f"{DIR}/swot_reach_D.parquet")
    s["reach_id"] = s.reach_id.astype("int64")
    s["t"] = pd.to_datetime(s.time_str, errors="coerce", utc=True)
    s = s.dropna(subset=["t", "wse"])
    # Table 1 inclusion criteria
    s = s[(s.reach_q < 3) & (s.partial_f == 0) & (s.ice_clim_f == 0)
          & (s.xovr_cal_q < 2) & (s.reach_q_b < 4_194_304) & (s.dark_frac <= 0.5)
          & s.xtrk_dist.abs().between(10_000, 60_000)
          & ((s.n_good_nod / s.p_n_nodes.replace(0, np.nan)) > 0.5)
          & (s.p_width >= 80) & (s.p_length >= 7_000)
          & ((s.wse - s.p_wse).abs() < 10)]
    g = pd.read_parquet(f"{DIR}/usgs_iv.parquet")
    g["gage_id"] = g.gage_id.astype(str).str.zfill(8)
    g["t_utc"] = pd.to_datetime(g.t_utc, utc=True)
    pairs = pd.read_csv(f"{DIR}/control_pairs.csv", dtype={"gage_id": str})
    pairs["gage_id"] = pairs.gage_id.str.zfill(8)
    pairs["reach_id"] = pairs.reach_id.astype("int64")

    m = pairs.merge(s, on="reach_id")
    g = g.sort_values("t_utc")
    m = m.sort_values("t")
    m = pd.merge_asof(m, g, left_on="t", right_on="t_utc", by="gage_id",
                      direction="nearest", tolerance=pd.Timedelta(hours=1))
    m = m.dropna(subset=["stage_m"])
    m = m.sort_values("t")
    key = m.reach_id.astype(str) + "_" + m.gage_id
    m = m[m.groupby(key, group_keys=False).wse.apply(hampel_mask)]
    key = m.reach_id.astype(str) + "_" + m.gage_id
    m["wse_rel"] = m.wse - m.groupby(key).wse.transform("median")
    m["stage_rel"] = m.stage_m - m.groupby(key).stage_m.transform("median")
    m["npair"] = m.groupby(key).wse.transform("size")
    m = m[m.npair >= MIN_PAIRS]
    m["e"] = (m.wse_rel - m.stage_rel).abs()
    return m[m.e <= 100]


def row(name, g):
    n = len(g)
    if not n:
        return {"group": name, "n": 0}
    hi = int((g.e >= T).sum())
    lo, hiCI = proportion_confint(hi, n, method="wilson")
    return {"group": name, "n": n, "reaches": g.reach_id.nunique(),
            "median_m": float(g.e.median()), "p68_m": p68(g.e),
            "share_over_T_pct": 100 * hi / n, "ci_lo": 100 * lo, "ci_hi": 100 * hiCI}


def run():
    m = load()
    pub = pq.read_table(URBAN_OBS, columns=["reach_id", "wse_relative_m",
                                            "wse_usgs_relative_m"]).to_pandas()
    pub["e"] = (pub.wse_relative_m - pub.wse_usgs_relative_m).abs()
    pub["reach_id"] = pub.reach_id.astype("int64")

    urb = m[m.group == "urban409"]
    cand = m[m.group != "urban409"]
    print(f"matched obs {len(m):,}  reaches {m.reach_id.nunique():,}  "
          f"urban409 {len(urb):,}  candidates {len(cand):,}")

    rows = [row("published urban file (reference)", pub),
            row("urban409 pairs, Version D recomputed", urb),
            row("control  fctimp < 2%", cand[cand.fctimp_mean < 2]),
            row("intermediate  2-10%", cand[cand.fctimp_mean.between(2, 10)]),
            row("urban-like  fctimp > 10%", cand[cand.fctimp_mean > 10])]
    out = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    out.to_csv(f"{OUT}/nonurban_control_versionD.csv", index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    ctrl = cand[cand.fctimp_mean < 2]
    if len(ctrl) and len(urb):
        mw = st.mannwhitneyu(ctrl.e, urb.e, alternative="two-sided")
        ks = st.ks_2samp(ctrl.e, urb.e)
        print(f"\ncontrol vs urban409-D  Mann-Whitney p {mw.pvalue:.2e}  "
              f"KS D {ks.statistic:.3f} p {ks.pvalue:.2e}")

    bins = [0, 1, 2, 5, 10, 20, 40, 100]
    m["bin"] = pd.cut(m.fctimp_mean, bins, right=False)
    dose = []
    for b, g in m.groupby("bin", observed=True):
        if len(g) < 200:
            continue
        dose.append({"fctimp_bin": str(b), "n": len(g), "reaches": g.reach_id.nunique(),
                     "p68_m": p68(g.e), "share_over_T_pct": 100 * (g.e >= T).mean()})
    dd = pd.DataFrame(dose)
    dd.to_csv(f"{OUT}/fctimp_dose_response_versionD.csv", index=False)
    print("\ndose response by impervious fraction (Version D, Table 1 filters)")
    print(dd.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    rr = m.dropna(subset=["fctimp_mean"]).groupby("reach_id").agg(
        f=("fctimp_mean", "first"), q=("e", p68))
    if len(rr) > 10:
        r, p = st.spearmanr(rr.f, rr.q)
        print(f"\nreach-level Spearman(fctimp, p68 error) {r:+.3f} (p={p:.2e}, n={len(rr)})")
    print(f"\nsaved {OUT}/nonurban_control_versionD.csv and fctimp_dose_response_versionD.csv")


if __name__ == "__main__":
    run()
