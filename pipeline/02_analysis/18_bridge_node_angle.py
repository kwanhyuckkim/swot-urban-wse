#!/usr/bin/env python
"""
Bridges at node resolution, and the 90-degree question.

The reach-level version of this test dilutes the signal: a bridge occupies tens of
metres of a reach that runs for ten kilometres, so assigning one gauge-to-bridge
distance to every node in the reach buries the effect. Here each node gets its own
distance to the nearest waterway crossing in the National Bridge Inventory.

The hypothesis under test is that a bridge lays over onto the water under the
configuration that is safe for riverside buildings. Buildings line the bank, so their
layover reaches the channel when the channel runs parallel to the track. A bridge
crosses the channel, so its axis is rotated by about 90 degrees and its layover should
matter when the channel runs perpendicular to the track.
"""
import io
import time
import numpy as np, pandas as pd
import pyarrow.parquet as pq
import requests
from scipy import stats as st
from scipy.spatial import cKDTree

PROJ = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results"
OUT = f"{PROJ}/round2"
NODES = f"{PROJ}/swot_urban_nearfar_v6.5_results.parquet"
CACHE = f"{OUT}/nbi_waterway_bridges.parquet"
STATES = ("AL AR AZ CA CO CT DC DE FL GA IA ID IL IN KS KY LA MA MD ME MI MN MO MS "
          "MT NC ND NE NH NJ NM NV NY OH OK OR PA RI SC SD TN TX UT VA VT WA WI WV WY").split()


def dms(v, sign=1, lo=-180, hi=180):
    v = pd.to_numeric(v, errors="coerce")
    out = sign * (v // 1_000_000 + (v // 10_000) % 100 / 60 + (v % 10_000) / 100 / 3600)
    return out.where((out > lo) & (out < hi))


def load_bridges():
    import os
    if os.path.exists(CACHE):
        b = pd.read_parquet(CACHE)
        print(f"bridges from cache: {len(b):,}")
        return b
    frames = []
    for s in STATES:
        url = f"https://www.fhwa.dot.gov/bridge/nbi/2024/delimited/{s}24.txt"
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=240)
                if r.status_code == 200 and "STRUCTURE_NUMBER" in r.text[:2000]:
                    df = pd.read_csv(io.StringIO(r.text), dtype=str, low_memory=False,
                                     on_bad_lines="skip")
                    frames.append(df[["LAT_016", "LONG_017", "SERVICE_UND_042B"]])
                    break
            except Exception:
                time.sleep(2 ** attempt)
        else:
            print(f"  fetch failed: {s}")
    b = pd.concat(frames, ignore_index=True)
    b["lat"] = dms(b.LAT_016, 1, 17, 72)
    b["lon"] = dms(b.LONG_017, -1, -180, -60)
    b = b[b.SERVICE_UND_042B.isin(["5", "6", "7", "8"])].dropna(subset=["lat", "lon"])
    b = b[["lat", "lon"]].reset_index(drop=True)
    b.to_parquet(CACHE, index=False)
    print(f"bridges fetched: {len(b):,} -> {CACHE}")
    return b


def p68(x):
    return float(np.nanpercentile(x, 68)) if len(x) else np.nan


d = pq.read_table(NODES, columns=["reach_id", "lat", "lon", "flow_angle", "far_isa",
                                  "far_building", "error_node", "width_node"]).to_pandas()
d["e"] = d.error_node.abs()
fa = d.flow_angle % 360
d["ang"] = np.minimum.reduce([fa, (fa - 180).abs(), 360 - fa])
d["isa_pct"] = (d.far_isa * 100 if d.far_isa.max() <= 1 else d.far_isa).fillna(0)
d["bld"] = d.far_building.fillna(0)
d = d.dropna(subset=["e", "ang", "lat", "lon"])

b = load_bridges()
lat0 = np.radians(d.lat.mean())
kx, ky = 111.32 * np.cos(lat0), 111.32
tree = cKDTree(np.column_stack([b.lon * kx, b.lat * ky]))
dist_km, _ = tree.query(np.column_stack([d.lon * kx, d.lat * ky]))
d["bdist_m"] = dist_km * 1000
print(f"nodes {len(d):,} | node-to-bridge median {d.bdist_m.median():.0f} m | "
      f"within 100 m {100*(d.bdist_m <= 100).mean():.1f}% | within 300 m {100*(d.bdist_m <= 300).mean():.1f}%")

print("\n[1] node error by distance to the nearest bridge")
bins = [0, 50, 100, 200, 300, 500, 1000, 1e9]
rows = []
for lo, hi in zip(bins[:-1], bins[1:]):
    g = d[(d.bdist_m >= lo) & (d.bdist_m < hi)]
    if len(g) < 200:
        continue
    rows.append({"lo_m": lo, "hi_m": hi, "n": len(g), "p68_m": p68(g.e),
                 "median_m": float(g.e.median()), "isa": float(g.isa_pct.median()),
                 "bld": float(g.bld.median()), "width": float(g.width_node.median())})
t = pd.DataFrame(rows)
print(t.round(3).to_string(index=False))
r, p = st.spearmanr(d.bdist_m, d.e)
print(f"  Spearman(node-to-bridge distance, |error|) = {r:+.4f} (p={p:.2e})")

print("\n[2] the 90-degree question: bridge effect by channel-to-track angle")
NEAR = 100.0
grid = []
for lo, hi, lab in [(0, 13, "parallel (<=13)"), (13, 45, "oblique"),
                    (45, 77, "oblique high"), (77, 91, "perpendicular (>=77)")]:
    sub = d[(d.ang >= lo) & (d.ang < hi)]
    near = sub[sub.bdist_m <= NEAR]
    far = sub[sub.bdist_m > NEAR]
    if len(near) < 100 or len(far) < 100:
        continue
    mw = st.mannwhitneyu(near.e, far.e)
    grid.append({"angle_zone": lab, "n_near": len(near), "n_far": len(far),
                 "p68_near": p68(near.e), "p68_far": p68(far.e),
                 "ratio": p68(near.e) / p68(far.e), "mw_p": mw.pvalue})
g = pd.DataFrame(grid)
print(g.round(4).to_string(index=False))
print("  가설: 교량이 90도 회전한 layover 원이면 perpendicular에서 ratio가 가장 커야 함")

print("\n[3] non-urban nodes only (buildings removed as a competing explanation)")
q = d[(d.isa_pct < 5) & (d.bld < 1)]
for lo, hi, lab in [(0, 13, "parallel"), (77, 91, "perpendicular")]:
    sub = q[(q.ang >= lo) & (q.ang < hi)]
    near, far = sub[sub.bdist_m <= NEAR], sub[sub.bdist_m > NEAR]
    if len(near) < 100 or len(far) < 100:
        continue
    mw = st.mannwhitneyu(near.e, far.e)
    print(f"  {lab:14s} near {p68(near.e):.3f} (n={len(near):,}) vs far {p68(far.e):.3f} "
          f"(n={len(far):,}) | ratio {p68(near.e)/p68(far.e):.2f} | MW p={mw.pvalue:.1e}")

print("\n[4] width control (bridges favour narrow channels)")
for wlo, whi in [(0, 150), (150, 300), (300, 1e9)]:
    sub = d[(d.width_node >= wlo) & (d.width_node < whi)]
    near, far = sub[sub.bdist_m <= NEAR], sub[sub.bdist_m > NEAR]
    if len(near) < 100 or len(far) < 100:
        continue
    print(f"  width {wlo:>4}-{whi if whi < 1e8 else 'inf':<4} m: near {p68(near.e):.3f} "
          f"vs far {p68(far.e):.3f} | ratio {p68(near.e)/p68(far.e):.2f} "
          f"(n {len(near):,}/{len(far):,})")

t.to_csv(f"{OUT}/bridge_node_distance.csv", index=False)
g.to_csv(f"{OUT}/bridge_node_angle.csv", index=False)
print(f"\nsaved {OUT}/bridge_node_distance.csv, bridge_node_angle.csv")
