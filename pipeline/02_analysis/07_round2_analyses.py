#!/usr/bin/env python
"""
Second-round analyses for TGRS-2026-02661, driven by an internal adversarial review.

Subcommands
  pops    Population inventory. Every performance figure quoted anywhere, with its
          subset definition, n, reach count, and before/after 68th percentile, so the
          three populations now quoted side by side stop looking inconsistent.
  cv      Reach-grouped cross-validation of the threshold selection, with random and
          oracle references at matched removal, answering the in-sample objection.
  boot    Cluster bootstrap of the selected thresholds themselves.
  sens    Mask performance as a continuous function of each threshold.
  width   Width-binned table and a quantile regression at tau = 0.68.
  flags   Quality-flag high-error share with binomial intervals, plus the geometric
          screen scored on the same axis.
  huc     Fixed thresholds applied within HUC2 regions.
"""
import argparse, glob, io, json, os, sys
import numpy as np, pandas as pd
import pyarrow.parquet as pq

WORK = "/work/pi_kandread_umass_edu/swot-urban"
PROJ = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results"
OUT = f"{PROJ}/round2"
NODES = f"{PROJ}/swot_urban_nearfar_v6.5_results.parquet"
OBS = f"{WORK}/2_urban_reaches/all_final_shape.parquet"
T = 0.1832
ANGLE, HEIGHT, ISA = 13.0, 8.0, 19.0


def load_nodes():
    d = pq.read_table(NODES, columns=[
        "reach_id", "flow_angle", "far_isa", "far_ratio", "far_building",
        "near_isa", "near_ratio", "error_node", "width_node"]).to_pandas()
    d["reach_id"] = d.reach_id.astype("int64")
    d["e"] = d.error_node.abs()
    fa = d.flow_angle % 360
    d["angle_dev"] = np.minimum.reduce([fa, (fa - 180).abs(), 360 - fa])
    d["isa_pct"] = d.far_isa * 100 if d.far_isa.max() <= 1 else d.far_isa
    return d.dropna(subset=["e"])


def p68(x):
    return float(np.nanpercentile(x, 68)) if len(x) else np.nan


def mask_of(d, angle=ANGLE, height=HEIGHT, isa=ISA):
    """Observations the screen flags: parallel and either tall far-bank buildings or high far-bank ISA."""
    par = d.angle_dev <= angle
    return par & ((d.far_building > height) | (d.isa_pct > isa))


# ---------------------------------------------------------------- pops
def run_pops():
    d = load_nodes()
    par = d[d.angle_dev <= ANGLE]
    rows = []

    def add(name, sub, removed):
        if len(sub) < 100:
            return
        keep = sub[~removed]
        rows.append({"population": name, "n": len(sub), "reaches": sub.reach_id.nunique(),
                     "p68_before_m": p68(sub.e), "p68_after_m": p68(keep.e),
                     "abs_drop_cm": (p68(sub.e) - p68(keep.e)) * 100,
                     "rel_drop_pct": (p68(sub.e) - p68(keep.e)) / p68(sub.e) * 100,
                     "flagged_pct": removed.mean() * 100})

    add("all node observations", d, mask_of(d))
    add("parallel zone (angle dev <= 13 deg)", par, mask_of(par))
    fi = par[par.far_isa.fillna(0) > 0]
    add("parallel and far-bank ISA bearing", fi, fi.isa_pct > ISA)
    fb = par[par.far_building.fillna(0) > 0]
    add("parallel and far-bank buildings present", fb, fb.far_building > HEIGHT)
    fr = par[(par.far_isa.fillna(0) > 0) & (par.far_ratio.fillna(0) >= 0.5)]
    add("parallel and far side qualifies (coverage >= 0.5)", fr, fr.isa_pct > ISA)

    out = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    out.to_csv(f"{OUT}/population_inventory.csv", index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nsaved {OUT}/population_inventory.csv")
    return out


# ---------------------------------------------------------------- cv
def select_thresholds(train):
    """Knee-point style selection on a training split: pick the combination that
    maximises reduction per unit of data removed, over the published grids."""
    best, grid = None, []
    for a in range(5, 31, 1):
        par = train[train.angle_dev <= a]
        if len(par) < 500:
            continue
        base = p68(par.e)
        for h in np.arange(4, 20.1, 1.0):
            for i in np.arange(5, 40.1, 2.0):
                rm = (par.far_building > h) | (par.isa_pct > i)
                if rm.mean() < 0.02 or rm.mean() > 0.8:
                    continue
                red = (base - p68(par[~rm].e)) / base * 100
                eff = red / (rm.mean() * 100)
                grid.append((a, h, i, red, rm.mean() * 100, eff))
                if best is None or eff > best[5]:
                    best = (a, h, i, red, rm.mean() * 100, eff)
    return best, grid


def run_cv(folds=5, seed=0):
    d = load_nodes()
    reaches = np.array(sorted(d.reach_id.unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(reaches)
    parts = np.array_split(reaches, folds)
    rows = []
    for k, test_r in enumerate(parts):
        tr = d[~d.reach_id.isin(test_r)]
        te = d[d.reach_id.isin(test_r)]
        best, _ = select_thresholds(tr)
        if best is None:
            continue
        a, h, i = best[0], best[1], best[2]
        par_te = te[te.angle_dev <= a]
        if len(par_te) < 100:
            continue
        base = p68(par_te.e)
        rm = (par_te.far_building > h) | (par_te.isa_pct > i)
        held = (base - p68(par_te[~rm].e)) / base * 100
        # references at matched removal
        frac = rm.mean()
        rnd = np.mean([(base - p68(par_te[rng.random(len(par_te)) > frac].e)) / base * 100
                       for _ in range(20)])
        thr = np.nanquantile(par_te.e, 1 - frac)
        oracle = (base - p68(par_te[par_te.e < thr].e)) / base * 100
        rows.append({"fold": k, "n_test_reaches": len(test_r), "n_test_par": len(par_te),
                     "sel_angle": a, "sel_height": h, "sel_isa": i,
                     "heldout_reduction_pct": held, "random_pct": rnd, "oracle_pct": oracle,
                     "flagged_pct": frac * 100})
    out = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    out.to_csv(f"{OUT}/crossval.csv", index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    if len(out):
        print(f"\nheld-out reduction  mean {out.heldout_reduction_pct.mean():.2f}% "
              f"sd {out.heldout_reduction_pct.std():.2f}")
        print(f"random reference    mean {out.random_pct.mean():.2f}%")
        print(f"oracle ceiling      mean {out.oracle_pct.mean():.2f}%")
        print(f"selected thresholds angle {sorted(out.sel_angle)} height {sorted(out.sel_height)} "
              f"isa {sorted(out.sel_isa)}")
    print(f"\nsaved {OUT}/crossval.csv")
    return out


# ---------------------------------------------------------------- boot
def run_boot(B=300, seed=0):
    d = load_nodes()
    reaches = np.array(sorted(d.reach_id.unique()))
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(B):
        pick = rng.choice(reaches, size=len(reaches), replace=True)
        samp = pd.concat([d[d.reach_id == r] for r in pick[:120]], ignore_index=True)
        best, _ = select_thresholds(samp)
        if best:
            rows.append({"rep": b, "angle": best[0], "height": best[1], "isa": best[2],
                         "reduction_pct": best[3], "flagged_pct": best[4]})
    out = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    out.to_csv(f"{OUT}/threshold_bootstrap.csv", index=False)
    if len(out):
        for c in ("angle", "height", "isa", "reduction_pct"):
            q = out[c].quantile([.025, .5, .975])
            print(f"  {c:14s} median {q[.5]:.1f}  95% CI [{q[.025]:.1f}, {q[.975]:.1f}]")
    print(f"\nsaved {OUT}/threshold_bootstrap.csv  (B={len(out)})")
    return out


# ---------------------------------------------------------------- sens
def run_sens():
    d = load_nodes()
    rows = []
    for a in range(5, 31):
        par = d[d.angle_dev <= a]
        if len(par) < 500:
            continue
        base = p68(par.e)
        rm = (par.far_building > HEIGHT) | (par.isa_pct > ISA)
        rows.append({"vary": "angle_deg", "value": a, "n": len(par), "base_p68": base,
                     "reduction_pct": (base - p68(par[~rm].e)) / base * 100,
                     "flagged_pct": rm.mean() * 100})
    par = d[d.angle_dev <= ANGLE]
    base = p68(par.e)
    for h in np.arange(3, 20.1, 0.5):
        rm = (par.far_building > h) | (par.isa_pct > ISA)
        rows.append({"vary": "height_m", "value": h, "n": len(par), "base_p68": base,
                     "reduction_pct": (base - p68(par[~rm].e)) / base * 100,
                     "flagged_pct": rm.mean() * 100})
    for i in np.arange(2, 50.1, 1.0):
        rm = (par.far_building > HEIGHT) | (par.isa_pct > i)
        rows.append({"vary": "isa_pct", "value": i, "n": len(par), "base_p68": base,
                     "reduction_pct": (base - p68(par[~rm].e)) / base * 100,
                     "flagged_pct": rm.mean() * 100})
    out = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    out.to_csv(f"{OUT}/threshold_sensitivity.csv", index=False)
    for v in out.vary.unique():
        s = out[out.vary == v]
        top = s.loc[s.reduction_pct.idxmax()]
        print(f"  {v:10s} best {top.value:6.1f} -> {top.reduction_pct:5.2f}% "
              f"(flagging {top.flagged_pct:.1f}%); plateau within 1% of best: "
              f"{s[s.reduction_pct >= top.reduction_pct - 1].value.min():.1f} to "
              f"{s[s.reduction_pct >= top.reduction_pct - 1].value.max():.1f}")
    print(f"\nsaved {OUT}/threshold_sensitivity.csv")
    return out


# ---------------------------------------------------------------- width
def run_width():
    from scipy import stats as st
    d = load_nodes()
    par = d[(d.angle_dev <= ANGLE)].dropna(subset=["width_node"])
    bins = [0, 100, 200, 300, 400, 600, 800, 1200, 1e9]
    lab = ["<100", "100-200", "200-300", "300-400", "400-600", "600-800", "800-1200", ">1200"]
    par = par.assign(wbin=pd.cut(par.width_node, bins, labels=lab))
    rows = []
    rng = np.random.default_rng(0)
    for b, g in par.groupby("wbin", observed=True):
        if len(g) < 50:
            continue
        boot = [p68(g.e.sample(len(g), replace=True, random_state=int(rng.integers(1e9)))) for _ in range(300)]
        rows.append({"width_bin_m": b, "n": len(g), "reaches": g.reach_id.nunique(),
                     "median_m": float(g.e.median()), "p68_m": p68(g.e),
                     "p68_lo": float(np.percentile(boot, 2.5)), "p68_hi": float(np.percentile(boot, 97.5))})
    out = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    out.to_csv(f"{OUT}/width_bins.csv", index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    r, p = st.spearmanr(par.width_node, par.e)
    print(f"\n  Spearman(width, |error|) all obs: {r:+.3f} (p={p:.1e}, n={len(par):,})")
    rr = par.groupby("reach_id").agg(w=("width_node", "median"), e=("e", lambda x: p68(x)))
    r2, p2 = st.spearmanr(rr.w, rr.e)
    print(f"  reach level (width vs reach p68): {r2:+.3f} (p={p2:.3f}, n={len(rr)})")
    print(f"\nsaved {OUT}/width_bins.csv")
    return out


# ---------------------------------------------------------------- flags
def run_flags():
    from statsmodels.stats.proportion import proportion_confint
    df = pq.read_table(OBS, columns=["reach_id", "reach_q_b", "wse_relative_m",
                                     "wse_usgs_relative_m"]).to_pandas()
    df["e"] = (df.wse_relative_m - df.wse_usgs_relative_m).abs()
    df["hi"] = df.e >= T
    q = df.reach_q_b.astype("int64").values
    base = df.hi.mean()
    BITS = {"class_suspect": 2, "geo_suspect": 4, "water_frac": 8, "bright_land": 128,
            "few_area": 1024, "few_wse": 2048, "far_range": 8192, "near_range": 16384,
            "partial_obs": 32768, "class_degraded": 262144, "geo_degraded": 524288,
            "lake_flag": 4194304}
    rows = [{"indicator": "dataset base rate", "n": len(df), "coverage_pct": 100.0,
             "high_error_share_pct": base * 100, "ci_lo": np.nan, "ci_hi": np.nan, "lift": 1.0}]
    for k, b in BITS.items():
        m = (q & b) > 0
        n = int(m.sum())
        if n == 0:
            rows.append({"indicator": k, "n": 0, "coverage_pct": 0.0,
                         "high_error_share_pct": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                         "lift": np.nan})
            continue
        s = df.hi[m].mean()
        lo, hi = proportion_confint(int(df.hi[m].sum()), n, method="wilson")
        rows.append({"indicator": k, "n": n, "coverage_pct": n / len(df) * 100,
                     "high_error_share_pct": s * 100, "ci_lo": lo * 100, "ci_hi": hi * 100,
                     "lift": s / base})
    out = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    out.to_csv(f"{OUT}/flag_lift.csv", index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\n  base rate {base*100:.1f}%; flags whose interval excludes it: "
          f"{[r['indicator'] for _, r in out.iterrows() if pd.notna(r.ci_lo) and (r.ci_lo > base*100 or r.ci_hi < base*100)]}")
    print(f"\nsaved {OUT}/flag_lift.csv")
    return out


# ---------------------------------------------------------------- huc
def run_huc():
    import requests
    obs = pq.read_table(OBS, columns=["reach_id", "site_no", "wse_relative_m",
                                      "wse_usgs_relative_m"]).to_pandas()
    obs["reach_id"] = obs.reach_id.astype("int64")
    sites = sorted(obs.site_no.dropna().astype(str).unique())
    recs = []
    for i in range(0, len(sites), 100):
        batch = ",".join(sites[i:i + 100])
        try:
            r = requests.get("https://waterservices.usgs.gov/nwis/site/",
                             params={"format": "rdb", "sites": batch, "siteOutput": "expanded"},
                             timeout=60)
            hdr = None
            for line in r.text.splitlines():
                if line.startswith("#"):
                    continue
                cols = line.split("\t")
                if hdr is None:
                    if "site_no" in cols and "huc_cd" in cols:
                        hdr = {c: k for k, c in enumerate(cols)}
                    continue
                if line.startswith("5s") or len(cols) <= max(hdr["site_no"], hdr["huc_cd"]):
                    continue
                recs.append({"site_no": cols[hdr["site_no"]], "huc_cd": cols[hdr["huc_cd"]]})
        except Exception as e:
            print("  NWIS batch failed:", e)
    hu = pd.DataFrame(recs).drop_duplicates("site_no")
    hu["huc2"] = hu.huc_cd.astype(str).str[:2]
    print(f"  HUC retrieved for {len(hu)} of {len(sites)} sites")
    d = load_nodes()
    link = obs[["reach_id", "site_no"]].drop_duplicates().merge(hu, on="site_no", how="left")
    d = d.merge(link[["reach_id", "huc2"]].drop_duplicates("reach_id"), on="reach_id", how="left")
    rows = []
    par_all = d[d.angle_dev <= ANGLE]
    for h, g in par_all.groupby("huc2"):
        if len(g) < 300 or pd.isna(h) or not str(h).strip():
            continue
        base = p68(g.e)
        rm = (g.far_building > HEIGHT) | (g.isa_pct > ISA)
        rows.append({"huc2": h, "n": len(g), "reaches": g.reach_id.nunique(),
                     "p68_before_m": base, "p68_after_m": p68(g[~rm].e),
                     "reduction_pct": (base - p68(g[~rm].e)) / base * 100,
                     "flagged_pct": rm.mean() * 100})
    out = pd.DataFrame(rows).sort_values("n", ascending=False)
    os.makedirs(OUT, exist_ok=True)
    out.to_csv(f"{OUT}/huc2_table.csv", index=False)
    hu.to_csv(f"{OUT}/site_huc.csv", index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nsaved {OUT}/huc2_table.csv")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["pops", "cv", "boot", "sens", "width", "flags", "huc", "all"])
    a = ap.parse_args()
    fns = {"pops": run_pops, "cv": run_cv, "boot": run_boot, "sens": run_sens,
           "width": run_width, "flags": run_flags, "huc": run_huc}
    for name in (fns if a.task == "all" else [a.task]):
        print(name.upper())
        try:
            fns[name]()
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
        print()
