#!/usr/bin/env python
"""
Reach fixed-effects test for TGRS-2026-02661: does the geometry-error association
survive within reaches, or does it live only in cross-reach differences?

The title claims geometric *controls*. Cross-reach correlation cannot support that
word alone, because reaches differ in gauge quality, width, and hydrology. Here every
comparison stays inside a reach, so any reach-level confounder cancels.

Four escalating tests
  1. Pooled Spearman (reference, the published kind of number).
  2. Partial Spearman absorbing reach identity: rank-transform, demean ranks within
     reach, correlate residuals.
  3. Per-reach Spearman distribution: median rho, share positive, Wilcoxon signed-rank.
  4. Within-reach and within-pass screen contrasts: for each reach (and each single
     overpass of each reach) holding both flagged and unflagged nodes, the p68
     difference flagged minus unflagged. Same river, same day, same gauge; only the
     node geometry differs.
  5. OLS of log|error| on the three predictors with reach fixed effects and
     cluster-robust (reach) standard errors.
"""
import os
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from scipy import stats as st

PROJ = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results"
OUT = f"{PROJ}/round2"
NODES = f"{PROJ}/swot_urban_nearfar_v6.5_results.parquet"
ANGLE, HEIGHT, ISA = 13.0, 8.0, 19.0


def load_nodes():
    d = pq.read_table(NODES, columns=[
        "reach_id", "node_id", "time_str", "flow_angle", "far_isa", "far_ratio",
        "far_building", "error_node"]).to_pandas()
    d["reach_id"] = d.reach_id.astype("int64")
    d["e"] = d.error_node.abs()
    fa = d.flow_angle % 360
    d["angle_dev"] = np.minimum.reduce([fa, (fa - 180).abs(), 360 - fa])
    d["isa_pct"] = d.far_isa * 100 if d.far_isa.max() <= 1 else d.far_isa
    d["isa_pct"] = d.isa_pct.fillna(0)
    d["far_building"] = d.far_building.fillna(0)
    t = pd.to_datetime(d.time_str, errors="coerce", utc=True)
    d["visit"] = d.reach_id.astype(str) + "_" + t.dt.floor("h").astype(str)
    return d.dropna(subset=["e"])


def p68(x):
    return float(np.nanpercentile(x, 68)) if len(x) else np.nan


def demean_by(v, groups):
    return v - v.groupby(groups).transform("mean")


def partial_spearman_fe(d, xcol):
    """Spearman with reach identity absorbed: correlate reach-demeaned ranks."""
    rx = d[xcol].rank()
    ry = d.e.rank()
    ax = demean_by(rx, d.reach_id)
    ay = demean_by(ry, d.reach_id)
    ok = ax.notna() & ay.notna()
    r, p = st.pearsonr(ax[ok], ay[ok])
    return r, p, int(ok.sum())


def per_reach_rho(d, xcol, min_n=30):
    rows = []
    for rid, g in d.groupby("reach_id"):
        if len(g) < min_n or g[xcol].nunique() < 5:
            continue
        r, p = st.spearmanr(g[xcol], g.e)
        if np.isfinite(r):
            rows.append({"reach_id": rid, "x": xcol, "n": len(g), "rho": r})
    return pd.DataFrame(rows)


def contrast_within(d, key, min_each=8):
    """p68(flagged) - p68(unflagged) inside each group of `key`."""
    par = d[d.angle_dev <= ANGLE].copy()
    par["flag"] = (par.far_building > HEIGHT) | (par.isa_pct > ISA)
    rows = []
    for k, g in par.groupby(key):
        f, u = g[g.flag], g[~g.flag]
        if len(f) < min_each or len(u) < min_each:
            continue
        rows.append({"group": k, "n_flag": len(f), "n_unflag": len(u),
                     "p68_flag": p68(f.e), "p68_unflag": p68(u.e),
                     "delta_m": p68(f.e) - p68(u.e)})
    return pd.DataFrame(rows)


def run():
    d = load_nodes()
    print(f"observations {len(d):,}  reaches {d.reach_id.nunique()}  "
          f"visits {d.visit.nunique():,}")
    res = []

    print("\n1) pooled Spearman (no control), all observations")
    for x in ("angle_dev", "far_building", "isa_pct"):
        r, p = st.spearmanr(d[x], d.e)
        res.append({"test": "pooled_spearman", "x": x, "stat": r, "p": p, "n": len(d)})
        print(f"   {x:13s} rho {r:+.3f}  p {p:.2e}")

    print("\n2) partial Spearman with reach identity absorbed")
    for x in ("angle_dev", "far_building", "isa_pct"):
        r, p, n = partial_spearman_fe(d, x)
        res.append({"test": "fe_partial_spearman", "x": x, "stat": r, "p": p, "n": n})
        print(f"   {x:13s} rho {r:+.3f}  p {p:.2e}")

    print("\n3) per-reach Spearman distribution (>=30 obs, >=5 distinct x)")
    for x in ("angle_dev", "far_building", "isa_pct"):
        pr = per_reach_rho(d, x)
        if not len(pr):
            continue
        w = st.wilcoxon(pr.rho)
        res.append({"test": "per_reach_median_rho", "x": x, "stat": pr.rho.median(),
                    "p": w.pvalue, "n": len(pr),
                    "extra": f"share_pos={(pr.rho > 0).mean():.3f}"})
        print(f"   {x:13s} median rho {pr.rho.median():+.3f}  "
              f"positive {100*(pr.rho > 0).mean():.1f}% of {len(pr)} reaches  "
              f"Wilcoxon p {w.pvalue:.2e}")

    print("\n4) screen contrast inside reach and inside single pass")
    for key, name in (("reach_id", "within_reach"), ("visit", "within_pass")):
        c = contrast_within(d, key)
        if not len(c):
            continue
        w = st.wilcoxon(c.delta_m)
        res.append({"test": f"{name}_contrast", "x": "screen", "stat": c.delta_m.median(),
                    "p": w.pvalue, "n": len(c),
                    "extra": f"share_pos={(c.delta_m > 0).mean():.3f}"})
        print(f"   {name:13s} median delta p68 {c.delta_m.median():+.3f} m  "
              f"flagged worse in {100*(c.delta_m > 0).mean():.1f}% of {len(c)} groups  "
              f"Wilcoxon p {w.pvalue:.2e}")
        c.to_csv(f"{OUT}/fe_{name}_contrast.csv", index=False)

    print("\n5) OLS log|error| with reach fixed effects, cluster-robust by reach")
    import statsmodels.api as sm
    y = np.log(d.e + 1e-3)
    X = d[["angle_dev", "far_building", "isa_pct"]].copy()
    yd = demean_by(y, d.reach_id)
    Xd = X.apply(lambda c: demean_by(c, d.reach_id))
    m = sm.OLS(yd, Xd).fit(cov_type="cluster", cov_kwds={"groups": d.reach_id})
    for x in Xd.columns:
        res.append({"test": "fe_ols_log_error", "x": x, "stat": m.params[x],
                    "p": m.pvalues[x], "n": int(m.nobs)})
        print(f"   {x:13s} beta {m.params[x]:+.5f}  cluster p {m.pvalues[x]:.2e}")

    os.makedirs(OUT, exist_ok=True)
    out = pd.DataFrame(res)
    out.to_csv(f"{OUT}/fixed_effects.csv", index=False)
    print(f"\nsaved {OUT}/fixed_effects.csv")

    pooled = out[out.test == "pooled_spearman"].set_index("x").stat
    fe = out[out.test == "fe_partial_spearman"].set_index("x")
    keep = all(np.sign(fe.stat[x]) == np.sign(pooled[x]) and fe.p[x] < 0.05
               for x in fe.index)
    print("\nverdict for the title word 'Controls':",
          "every predictor keeps its sign and significance with reach identity absorbed"
          if keep else "association changes sign or loses significance inside reaches")


if __name__ == "__main__":
    run()
