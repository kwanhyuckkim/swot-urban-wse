"""2026-10-05: reproduce the V.E perturbation test (no script/output found): add Gaussian noise with the building-
height product RMSE (sigma = 4.73 m) to far-nadir building heights, re-run the knee-point selection (same knee as
26b_cv_slope.py / 39), 10 repeats; two noise variants. Manuscript: thresholds 8.0-11.5 m, median 8.5 m."""
import numpy as np, pandas as pd, pyarrow.parquet as pq
P = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results"
d = pq.read_table(f"{P}/swot_urban_nearfar_v6.5_results.parquet", columns=["flow_angle", "far_building", "error_node"]).to_pandas()
d = d.dropna(subset=["error_node"]); fa = d.flow_angle % 180; d["ang"] = np.minimum(fa, 180 - fa)
d["e"] = d.error_node.abs(); d["bld"] = d.far_building.fillna(0)
par = d[(d.ang < 13) & (d.bld >= 0)].reset_index(drop=True)
p68 = lambda x: float(np.nanpercentile(x, 68))
GRID = np.arange(20, 4.5, -0.5)


def knee(sub, col):
    base = p68(sub.e); red, rmv = [], []
    for t in GRID:
        rm = sub[col] > t; ret = sub[~rm]
        red.append((base - p68(ret.e)) / base * 100 if len(ret) else np.nan); rmv.append(rm.mean() * 100)
    red = np.array(red); rmv = np.array(rmv)
    return float(GRID[int(np.nanargmax(red / np.nanmax(red) - rmv / np.nanmax(rmv)))])


print(f"parallel n {len(par):,}; knee without noise: {knee(par, 'bld')} m")
for lab in ("A: noise on all heights, clip at 0", "B: noise only on heights > 0, clip at 0", "C: noise on heights > 0, no clip"):
    out = []
    for s in range(10):
        rng = np.random.default_rng(s); x = par.copy(); nz = x.bld > 0
        noise = rng.normal(0, 4.73, len(x))
        if lab.startswith("A"): x["b2"] = np.clip(x.bld + noise, 0, None)
        elif lab.startswith("B"): x["b2"] = np.where(nz, np.clip(x.bld + noise, 0, None), 0.0)
        else: x["b2"] = np.where(nz, x.bld + noise, 0.0)
        out.append(knee(x, "b2"))
    print(f"{lab}: {out} | min {min(out)} max {max(out)} median {np.median(out)}")
