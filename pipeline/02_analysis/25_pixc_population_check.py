#!/usr/bin/env python
"""2026-09-28: which population do the Section V.C pixel-cloud numbers use?
23_number_checks.py computes +0.046 (layover_impact) and +0.034 (layovr_val) over the
520,958 node observations in node_coherence_all.parquet; the response letter (R4-13) quotes
520,731 node observations and 53,166 parallel ones for the same join. This script checks
whether 520,731 = the |e| <= 100 m subset and recomputes the V.C numbers on both."""
import numpy as np, pandas as pd
from scipy import stats as st
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

P = '/project/pi_kandread_umass_edu/swot-urban/3_analysis_results'
RC = f'{P}/reviewer_checks'; FILL = -1e11; T = 0.1832
m = pd.read_parquet(f'{RC}/node_coherence_all.parquet')
print(f"rows {len(m):,} | e<=100 {(m.e <= 100).sum():,} | e NaN {m.e.isna().sum():,} | ang<=10 {(m.ang <= 10).sum():,}")
n = pd.read_parquet(f'{RC}/layovr_val_node.parquet', columns=['node_id', 'time_str', 'layovr_val'])
n['node_id'] = n.node_id.astype('int64'); n['layovr_val'] = n.layovr_val.where(n.layovr_val > FILL)
n['t'] = pd.to_datetime(n.time_str, errors='coerce', utc=True).dt.floor('h')
n = n.dropna(subset=['t']).drop_duplicates(['node_id', 't'])[['node_id', 't', 'layovr_val']]
m2 = m.merge(n, on=['node_id', 't'], how='left')


def rho(a, b):
    k = a.notna() & b.notna()
    return f"{st.spearmanr(a[k], b[k]).statistic:+.4f} (n={int(k.sum()):,})"


def boosted(df, col):
    mm = df.dropna(subset=[col]).reset_index(drop=True)
    y = mm.e.values; yr = pd.Series(y).rank().values; g = mm.reach_id.values
    pg = np.full(len(mm), np.nan)
    for tr, te in GroupKFold(n_splits=5).split(mm[[col]], y, g):
        pg[te] = HistGradientBoostingRegressor(max_depth=3, max_iter=200, learning_rate=0.08,
                                               random_state=0).fit(mm[[col]].values[tr], yr[tr]).predict(mm[[col]].values[te])
    return f"{st.spearmanr(pg, y).statistic:+.4f} (n={len(mm):,})"


for name, sub in (('all rows', m2), ('e<=100', m2[m2.e <= 100])):
    br = sub[sub.bdist_m <= 100]
    print(f"\n[{name}] n {len(sub):,} | parallel (ang<=10) {(sub.ang <= 10).sum():,} | base {(sub.e >= T).mean()*100:.1f}%")
    print(f"  layover_impact vs e      {rho(sub.layover_impact_median, sub.e)}")
    print(f"  layovr_val vs e          {rho(sub.layovr_val, sub.e)}")
    print(f"  bridge<=100 m n {len(br):,}: layover_impact {rho(br.layover_impact_median, br.e)}, layovr_val {rho(br.layovr_val, br.e)}")
    print(f"  boosted out-of-fold: layovr_val {boosted(sub, 'layovr_val')}, layover_impact {boosted(sub, 'layover_impact_median')}")
