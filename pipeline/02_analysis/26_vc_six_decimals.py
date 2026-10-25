#!/usr/bin/env python
"""2026-09-28: six-decimal Section V.C correlations on the 520,958 pixel-cloud rows (rounding check)."""
import pandas as pd
from scipy import stats as st
RC = '/project/pi_kandread_umass_edu/swot-urban/3_analysis_results/reviewer_checks'; FILL = -1e11
m = pd.read_parquet(f'{RC}/node_coherence_all.parquet')
n = pd.read_parquet(f'{RC}/layovr_val_node.parquet', columns=['node_id', 'time_str', 'layovr_val'])
n['node_id'] = n.node_id.astype('int64'); n['layovr_val'] = n.layovr_val.where(n.layovr_val > FILL)
n['t'] = pd.to_datetime(n.time_str, errors='coerce', utc=True).dt.floor('h')
n = n.dropna(subset=['t']).drop_duplicates(['node_id', 't'])[['node_id', 't', 'layovr_val']]
m = m.merge(n, on=['node_id', 't'], how='left')
def r6(a, b):
    k = a.notna() & b.notna(); return f"{st.spearmanr(a[k], b[k]).statistic:+.6f} (n={int(k.sum()):,})"
br = m[m.bdist_m <= 100]
print('rows', f'{len(m):,}')
print('layover_impact', r6(m.layover_impact_median, m.e)); print('layovr_val', r6(m.layovr_val, m.e))
print('bridge layover_impact', r6(br.layover_impact_median, br.e)); print('bridge layovr_val', r6(br.layovr_val, br.e))
