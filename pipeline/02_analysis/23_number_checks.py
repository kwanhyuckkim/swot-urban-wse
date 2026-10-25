#!/usr/bin/env python
"""Re-check every number the manuscript cites against the results, resolve the layovr_val / layover_impact labeling, and test the two wording
claims that need data (reach bit base rates, far-range class firing)."""
import numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy import stats as st
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

P='/project/pi_kandread_umass_edu/swot-urban/3_analysis_results'
RC=f'{P}/reviewer_checks'; W='/work/pi_kandread_umass_edu/swot-urban'
FILL=-1e11; T=0.1832

m=pd.read_parquet(f'{RC}/node_coherence_all.parquet')
print(f"[0] 기반: {len(m):,} rows, {m.reach_id.nunique()} reaches, base {(m.e>=T).mean()*100:.1f}%")

def rho(a,b,lab):
    k=pd.notna(a)&pd.notna(b); r,p=st.spearmanr(a[k],b[k])
    print(f"  {lab:52s} {r:+.4f} (n={k.sum():,})"); return r

print("\n[1] §V.C 인용 수치 재계산 (parquet 직접)")
rho(m.coh_median,m.e,"coherence")
rho(m.lowcoh_pct,m.e,"low-coh class share")
rho(m.layover_impact_median,m.e,"layover_impact (PIXC, node-aggregated) ← 원고의 '+0.046'")
rho(m.bld,m.e,"far building"); rho(m.isa,m.e,"far ISA"); rho(m.ang,m.e,"angle dev")

print("\n[2] 진짜 layovr_val을 같은 행에 조인")
n=pd.read_parquet(f'{RC}/layovr_val_node.parquet',columns=['node_id','time_str','layovr_val'])
n['node_id']=n.node_id.astype('int64'); n['layovr_val']=n.layovr_val.where(n.layovr_val>FILL)
n['t']=pd.to_datetime(n.time_str,errors='coerce',utc=True).dt.floor('h')
n=n.dropna(subset=['t']).drop_duplicates(['node_id','t'])[['node_id','t','layovr_val']]
m2=m.merge(n,on=['node_id','t'],how='left')
print(f"  layovr_val 매칭: {m2.layovr_val.notna().sum():,}/{len(m2):,}")
r_lv=rho(m2.layovr_val,m2.e,"layovr_val (RiverSP 배포 필드, same rows)")
rho(m2.layovr_val,m2.layover_impact_median,"layovr_val vs layover_impact (서로)")
br=m2[m2.bdist_m<=100]
r_lv_br=rho(br.layovr_val,br.e,"layovr_val, bridge<=100m")
rho(br.layover_impact_median,br.e,"layover_impact, bridge<=100m ← 원고의 '+0.105'")
# boosted single, true layovr_val
mm=m2.dropna(subset=['layovr_val']).reset_index(drop=True)
y=mm.e.values; yr=pd.Series(y).rank().values; g=mm.reach_id.values
pg=np.full(len(mm),np.nan)
for tr,te in GroupKFold(n_splits=5).split(mm[['layovr_val']],y,g):
    pg[te]=HistGradientBoostingRegressor(max_depth=3,max_iter=200,learning_rate=0.08,
          random_state=0).fit(mm[['layovr_val']].values[tr],yr[tr]).predict(mm[['layovr_val']].values[te])
top=pg>=np.nanpercentile(pg,90)
print(f"  layovr_val boosted single: rho {st.spearmanr(pg,y).statistic:+.3f} | 상위10% 고오차 {(mm.e[top]>=T).mean()*100:.1f}%")

print("\n[3] 50-60 km 저코히런스 'never fires' 정밀검사")
g5=m[(m.xtrk_km>=50)&(m.xtrk_km<60)]
print(f"  n={len(g5):,} | lowcoh_pct 중앙 {g5.lowcoh_pct.median():.2f} | 평균 {g5.lowcoh_pct.mean():.2f} | "
      f">0 비율 {(g5.lowcoh_pct>0).mean()*100:.1f}% | 최대 {g5.lowcoh_pct.max():.1f}")

print("\n[4] 원고 '0.10/0.06' 정의 재현 (노드 layovr_val vs 동시각 reach 관측 오차)")
ob=pq.read_table(f'{W}/2_urban_reaches/all_final_shape.parquet',
    columns=['reach_id','time_str','wse_relative_m','wse_usgs_relative_m']).to_pandas()
ob['reach_id']=ob.reach_id.astype('int64')
ob['e_reach']=(ob.wse_relative_m-ob.wse_usgs_relative_m).abs()
ob['t']=pd.to_datetime(ob.time_str,errors='coerce',utc=True).dt.floor('h')
nv=pd.read_parquet(f'{RC}/layovr_val_node.parquet',columns=['reach_id','node_id','time_str','layovr_val'])
nv['reach_id']=nv.reach_id.astype('int64'); nv['layovr_val']=nv.layovr_val.where(nv.layovr_val>FILL)
nv['t']=pd.to_datetime(nv.time_str,errors='coerce',utc=True).dt.floor('h')
j=nv.dropna(subset=['t','layovr_val']).merge(ob[['reach_id','t','e_reach']].dropna(),on=['reach_id','t'],how='inner')
r10,_=st.spearmanr(j.layovr_val,j.e_reach)
print(f"  노드 layovr_val vs 동시각 reach 오차: {r10:+.3f} (n={len(j):,}) ← 원고 0.10 검증")
rv=pd.read_parquet(f'{RC}/layovr_val_reach.parquet',columns=['reach_id','time_str','layovr_val'])
rv['reach_id']=rv.reach_id.astype('int64'); rv['layovr_val']=rv.layovr_val.where(rv.layovr_val>FILL)
rv['t']=pd.to_datetime(rv.time_str,errors='coerce',utc=True).dt.floor('h')
jr=rv.dropna(subset=['t','layovr_val']).merge(ob[['reach_id','t','e_reach']].dropna(),on=['reach_id','t'],how='inner')
r06,_=st.spearmanr(jr.layovr_val,jr.e_reach)
print(f"  reach layovr_val vs reach 오차:        {r06:+.3f} (n={len(jr):,}) ← 원고 0.06 검증")
