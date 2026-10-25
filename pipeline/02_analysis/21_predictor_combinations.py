#!/usr/bin/env python
"""
Single predictors are the wrong comparison for our method, which combines three.

Our screen reads flow angle, far-bank ISA, and far-bank building height together, so
comparing each against normalized coherence one at a time understates it. Here every
predictor set is fitted and scored the same way, on the node observations where the
coherence exists, with folds grouped by reach so that a model cannot learn a reach and
be graded on it.

Scoring
  rank model   ordinary least squares on rank-transformed predictors, the natural
               multivariate extension of a Spearman correlation
  boosted      histogram gradient boosting, which can use the interaction between ISA
               and building height that the paper reports
  metric       out-of-fold Spearman between prediction and node absolute error,
               and the high-error share of the top decile of each prediction
"""
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from scipy import stats as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from statsmodels.stats.proportion import proportion_confint

PROJ = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results"
RC = f"{PROJ}/reviewer_checks"
OUT = f"{PROJ}/round2"
T = 0.1832

m = pd.read_parquet(f"{RC}/node_coherence.parquet")
m["ang"] = m["ang"] if "ang" in m else np.nan
need = ["e", "ang", "isa", "bld", "coh_median", "lowcoh_pct",
        "layover_impact_median", "sig0_median", "reach_id"]
m = m.dropna(subset=[c for c in need if c in m]).reset_index(drop=True)
print(f"rows {len(m):,} | reaches {m.reach_id.nunique()} | "
      f"high-error base {(m.e >= T).mean()*100:.1f}%")

# angle enters as its deviation from the track, which is small when layover matters,
# so the sign is flipped to keep every predictor pointing the same way
m["ang_par"] = -m.ang
SETS = {
    "angle alone": ["ang_par"],
    "far ISA alone": ["isa"],
    "building alone": ["bld"],
    "angle + ISA": ["ang_par", "isa"],
    "angle + building": ["ang_par", "bld"],
    "angle + ISA + building (ours)": ["ang_par", "isa", "bld"],
    "coherence alone": ["coh_median"],
    "low-coh class alone": ["lowcoh_pct"],
    "layover_impact alone (node-aggregated)": ["layover_impact_median"],
    "coherence + angle": ["coh_median", "ang_par"],
    "coherence + ours": ["coh_median", "ang_par", "isa", "bld"],
    "coherence + ours + low-coh class": ["coh_median", "ang_par", "isa", "bld", "lowcoh_pct"],
}

y = m.e.values
yr = pd.Series(y).rank().values
groups = m.reach_id.values
gkf = GroupKFold(n_splits=5)
rows = []
for name, cols in SETS.items():
    X = m[cols].copy()
    Xr = X.rank().values
    pred_lin = np.full(len(m), np.nan)
    pred_gb = np.full(len(m), np.nan)
    for tr, te in gkf.split(X, y, groups):
        pred_lin[te] = LinearRegression().fit(Xr[tr], yr[tr]).predict(Xr[te])
        gb = HistGradientBoostingRegressor(max_depth=3, max_iter=200,
                                           learning_rate=0.08, random_state=0)
        pred_gb[te] = gb.fit(X.values[tr], yr[tr]).predict(X.values[te])
    r_lin = st.spearmanr(pred_lin, y).statistic
    r_gb = st.spearmanr(pred_gb, y).statistic
    top = pred_gb >= np.nanpercentile(pred_gb, 90)
    share = (m.e[top] >= T).mean() * 100
    lo, hi = proportion_confint(int((m.e[top] >= T).sum()), int(top.sum()), method="wilson")
    rows.append({"predictors": name, "n_vars": len(cols),
                 "rank_model_rho": r_lin, "boosted_rho": r_gb,
                 "top_decile_high_error_pct": share,
                 "ci_lo": 100 * lo, "ci_hi": 100 * hi})
    print(f"  {name:34s} rank {r_lin:+.3f} | boosted {r_gb:+.3f} | "
          f"top-decile high-error {share:5.1f}% [{100*lo:.1f}, {100*hi:.1f}]")

out = pd.DataFrame(rows).sort_values("boosted_rho", ascending=False)
out.to_csv(f"{OUT}/predictor_combinations.csv", index=False)

print("\nordered by out-of-fold boosted correlation")
print(out[["predictors", "rank_model_rho", "boosted_rho",
           "top_decile_high_error_pct"]].round(3).to_string(index=False))

best_ours = out[out.predictors == "angle + ISA + building (ours)"].boosted_rho.iloc[0]
coh = out[out.predictors == "coherence alone"].boosted_rho.iloc[0]
both = out[out.predictors == "coherence + ours"].boosted_rho.iloc[0]
print(f"\nours {best_ours:+.3f} | coherence {coh:+.3f} | together {both:+.3f}")
print(f"coherence adds {both - best_ours:+.3f} over ours; "
      f"ours adds {both - coh:+.3f} over coherence")
print(f"\nsaved {OUT}/predictor_combinations.csv")
