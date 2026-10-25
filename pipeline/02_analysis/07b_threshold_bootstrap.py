#!/usr/bin/env python
"""Cluster bootstrap of the selected thresholds, parallel across reps.

The first implementation resampled reaches with a boolean scan of the full table
per reach, which dominated the runtime. Here the table is indexed by reach once
and each replicate is assembled from row positions, and replicates run across
cores."""
import os, sys, time
import numpy as np, pandas as pd, pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor

P="/project/pi_kandread_umass_edu/swot-urban/3_analysis_results"
OUT=f"{P}/round2"
_G={}

def _load():
    d=pq.read_table(f"{P}/swot_urban_nearfar_v6.5_results.parquet",
        columns=["reach_id","flow_angle","far_isa","far_building","error_node"]).to_pandas()
    d["e"]=d.error_node.abs()
    fa=d.flow_angle%360
    d["ad"]=np.minimum.reduce([fa,(fa-180).abs(),360-fa])
    d["isa"]=d.far_isa*100 if d.far_isa.max()<=1 else d.far_isa
    d=d.dropna(subset=["e","ad"])
    idx={r:g.to_numpy() for r,g in d.groupby("reach_id").indices.items()} if False else None
    groups=d.groupby("reach_id").indices
    return (d.e.to_numpy(), d.ad.to_numpy(), d.isa.to_numpy(),
            d.far_building.to_numpy(), {k:np.asarray(v) for k,v in groups.items()})

def _init():
    e,ad,isa,fb,groups=_load()
    _G.update(e=e,ad=ad,isa=isa,fb=fb,groups=groups,reaches=np.array(sorted(groups)))

def _select(rows, angles=range(5,31), heights=np.arange(4,20.1,1.0), isas=np.arange(5,40.1,2.0)):
    e,ad,isa,fb=_G["e"][rows],_G["ad"][rows],_G["isa"][rows],_G["fb"][rows]
    best=None
    for a in angles:
        m=ad<=a
        if m.sum()<500: continue
        ee,ii,bb=e[m],isa[m],fb[m]
        base=np.nanpercentile(ee,68)
        if not np.isfinite(base) or base<=0: continue
        for h in heights:
            bh=bb>h
            for i in isas:
                rm=bh|(ii>i)
                f=rm.mean()
                if f<0.02 or f>0.8: continue
                kept=ee[~rm]
                if kept.size<50: continue
                red=(base-np.nanpercentile(kept,68))/base*100
                eff=red/(f*100)
                if best is None or eff>best[-1]:
                    best=(a,float(h),float(i),float(red),float(f*100),float(eff))
    return best

def _one(seed):
    rng=np.random.default_rng(seed)
    pick=rng.choice(_G["reaches"], size=len(_G["reaches"]), replace=True)
    rows=np.concatenate([_G["groups"][r] for r in pick])
    b=_select(rows)
    return None if b is None else {"rep":int(seed),"angle":b[0],"height":b[1],"isa":b[2],
                                   "reduction_pct":b[3],"flagged_pct":b[4]}

if __name__=="__main__":
    B=int(sys.argv[1]) if len(sys.argv)>1 else 300
    W=int(os.environ.get("SLURM_CPUS_PER_TASK","8"))
    t0=time.time(); rows=[]
    print(f"bootstrap B={B} workers={W}",flush=True)
    with ProcessPoolExecutor(max_workers=W,initializer=_init) as ex:
        for n,r in enumerate(ex.map(_one,range(B),chunksize=1),1):
            if r: rows.append(r)
            if n%25==0:
                el=time.time()-t0
                print(f"  {n}/{B}  {el:.0f}s elapsed, ~{el/n*(B-n):.0f}s left",flush=True)
    out=pd.DataFrame(rows); os.makedirs(OUT,exist_ok=True)
    out.to_parquet(f"{OUT}/threshold_bootstrap.parquet",index=False)
    out.to_csv(f"{OUT}/threshold_bootstrap.csv",index=False)
    print(f"\nB={len(out)} in {time.time()-t0:.0f}s")
    for c in ("angle","height","isa","reduction_pct","flagged_pct"):
        q=out[c].quantile([.025,.5,.975])
        print(f"  {c:14s} median {q[.5]:7.1f}   95% CI [{q[.025]:.1f}, {q[.975]:.1f}]")
