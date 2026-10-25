#!/usr/bin/env python3
"""2026-10-05 (supplement Figure S4): rebuild of figures_print/figS5_structure_check.png (drawing script not found) from data,
with gauge-to-bridge distances on the sphere. The 08-11 figure used 14_structure_stratified.py 179-183 (one cos(mean lat)
plane scaling), giving 34% within 100 m, Spearman -0.012 (p 0.81), Mann-Whitney p 0.37/0.82/0.72; the manuscript (V.E) now
uses great-circle distances (35%, -0.007 (p 0.88), 0.43/0.82/0.60). Panel (a) data: round2_results/structure_stratified.csv
(unchanged values). Labels kept as in the 08-11 figure; only the numbers in (b) change.
Inputs: round2_results/bridge_check.csv (440 gauge rows, 409 reaches; med_err, glat, glon),
revision1/bridge_gc_20261005/nbi_waterway_bridges.parquet (505,138 NBI waterway crossings, lat/lon).
Output: revision1/figures_print/figS4_structure_bridge_1005.png/.pdf (7.16 x 3.0 in, 600 dpi, same size as the old PNG).
Update 10-05 16h: 437 conterminous-US gauges (Alaska excluded, see below), axes widened so that no point is clipped.
Update 10-05 16h20 (user: "세 개 모두 바꾸기"): labels "amplification" -> "ratio", "(near)/(far)" -> "(<=1 km)/(>1 km)", "organize" -> "account for"."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu, spearmanr

BASE = "/Users/targetkim/target/SWOT_urbanerror"
OUT = f"{BASE}/revision1/figures_print"
R_EARTH = 6371008.8  # m, mean radius

s = pd.read_csv(f"{BASE}/round2_results/structure_stratified.csv")
far = s[s.group.str.startswith("far")].iloc[0]    # no structure within 1 km
near = s[s.group.str.startswith("near")].iloc[0]  # structure within 1 km

g = pd.read_csv(f"{BASE}/round2_results/bridge_check.csv", dtype={"site_no": str})
nbi = pd.read_parquet(f"{BASE}/revision1/bridge_gc_20261005/nbi_waterway_bridges.parquet").dropna(subset=["lat", "lon"])
# 2026-10-05 user decision "알래스카 게이지 3개 빼기 (437개)": the NBI pull covers the 48 conterminous states + DC only
# (14_structure_stratified.py STATES; parquet lat 24.6-49.4), so the 3 Alaska gauges (15024800, 15266300, 15266110) have no valid distance.
g = g[g.glat < 50].copy()
assert len(g) == 437, len(g)


def xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


chord, _ = cKDTree(xyz(nbi.lat.values, nbi.lon.values)).query(xyz(g.glat.values, g.glon.values), k=1)
g["dist_gc_m"] = 2 * np.arcsin(chord / 2) * R_EARTH
share100 = (g.dist_gc_m <= 100).mean()
rho, p_rho = spearmanr(g.dist_gc_m, g.med_err)
mw = {}
for cut in (100, 250, 500):
    a, b = g[g.dist_gc_m <= cut].med_err, g[g.dist_gc_m > cut].med_err
    mw[cut] = mannwhitneyu(a, b).pvalue
    print(f"cut {cut} m: within n={len(a)} median {a.median():.3f} | beyond n={len(b)} median {b.median():.3f} | MW p={mw[cut]:.3f}")
out = ((g.dist_gc_m < 3) | (g.dist_gc_m > 8e4) | (g.med_err < 0.002) | (g.med_err > 10)).sum(); assert out == 0, out
print(f"NBI crossings {len(nbi):,}; gauges {len(g)} ({g.reach_id.nunique()} reaches); within 100 m {int((g.dist_gc_m <= 100).sum())}/{len(g)} "
      f"= {100 * share100:.1f}%; median distance {g.dist_gc_m.median():.0f} m; Spearman {rho:+.4f} (p={p_rho:.3f})")
print(f"(a) far/near: amplification {far.angle_ratio_both:.3f}/{near.angle_ratio_both:.3f}; mask {far.mask_reduction_pct:.2f}/{near.mask_reduction_pct:.2f}%")

# font sizes measured on the 08-11 PNG (ink widths solved against DejaVu Sans at 600 dpi)
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8.3, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 7.0, "axes.linewidth": 0.7, "figure.facecolor": "white"})
fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.16, 3.0))

# (a) zone hierarchy by structure proximity (values from structure_stratified.csv)
zones = ["Neither", "Near", "Far", "Both"]
vf = [far.p68_par_neither, far.p68_par_near, far.p68_par_far, far.p68_par_both]
vn = [near.p68_par_neither, near.p68_par_near, near.p68_par_far, near.p68_par_both]
x = np.arange(4); w = 0.36
b1 = axa.bar(x - w / 2, vf, w, color="#1f78b4", edgecolor="black", lw=0.6, label=f"No structure within 1 km (n = {int(far.reaches)})")
b2 = axa.bar(x + w / 2, vn, w, color="#d95f02", edgecolor="black", lw=0.6, label=f"Structure within 1 km (n = {int(near.reaches)} reaches)")
for bars, vals in ((b1, vf), (b2, vn)):
    for r_, v in zip(bars, vals):
        axa.text(r_.get_x() + r_.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=6.8)
axa.set_xticks(x); axa.set_xticklabels(zones)
axa.set_ylim(0, 2.0); axa.set_yticks(np.arange(0, 2.01, 0.25))
axa.set_ylabel("Parallel-zone 68th-pct error (m)")
axa.set_title("(a) Class hierarchy by structure proximity")  # 10-05: "class" as in V.E and the S4 caption
axa.legend(loc="upper left", frameon=True, handlelength=1.6)  # keeps the 1.71 label clear of the frame
axa.text(0.03, 0.665, f"Parallel/perpendicular ratio:\n{near.angle_ratio_both:.2f} (\N{LESS-THAN OR EQUAL TO}1 km) vs {far.angle_ratio_both:.2f} (>1 km)\n"
         f"Mask reduction: {near.mask_reduction_pct:.1f}% vs {far.mask_reduction_pct:.1f}%",
         transform=axa.transAxes, va="top", ha="left", fontsize=6.8,
         bbox=dict(boxstyle="square,pad=0.4", facecolor="white", edgecolor="0.6", lw=0.6))

# (b) gauge-to-nearest-bridge distance (great circle) vs reach median |error|
axb.scatter(g.dist_gc_m, g.med_err, s=9, color="#4c72b0", alpha=0.45, lw=0, zorder=2)
edges = [0, 50, 100, 250, 500, 1000, 2500, 10000, np.inf]
pos_lo = [10, 50, 100, 250, 500, 1000, 2500, 10000]; pos_hi = [50, 100, 250, 500, 1000, 2500, 10000, 30000]
g["bin"] = pd.cut(g.dist_gc_m, edges)
med = g.groupby("bin", observed=False).med_err.median().values
xc = np.sqrt(np.array(pos_lo) * np.array(pos_hi))
axb.plot(xc, med, "-o", color="#c62828", lw=1.4, ms=4.5, zorder=4, label="bin median")
print("bin medians:", np.round(med, 3), "counts:", g.groupby("bin", observed=False).size().values)
for cut in (100, 250, 500):
    axb.axvline(cut, ls=":", color="0.4", lw=0.8, zorder=1)
    axb.text(cut * 1.07, 9.6, f"p={mw[cut]:.2f}", rotation=90, ha="left", va="top", fontsize=6.5, color="0.45")
axb.set_xscale("log"); axb.set_yscale("log")
axb.set_ylim(0.002, 10.0); axb.set_xlim(3, 8e4)  # bottom 0.002 m keeps the text box below the lowest point (0.014 m)  # every point inside (user: "모든 점이 보이게 축 넓히기")
axb.set_xlabel("Gauge-to-nearest-bridge distance (m, NBI)")
axb.set_ylabel("Reach median |error| (m)")
axb.set_title("(b) Bridge proximity does not account for error")
axb.legend(loc="upper right", frameon=True)
axb.text(0.04, 0.04, f"{len(nbi):,} waterway crossings\n{100 * share100:.0f}% of gauges within 100 m\n"
         f"Spearman \N{GREEK SMALL LETTER RHO} = {rho:+.3f} (p = {p_rho:.2f})".replace("+", "").replace("-", "\N{MINUS SIGN}"),
         transform=axb.transAxes, va="bottom", ha="left", fontsize=6.8,
         bbox=dict(boxstyle="square,pad=0.4", facecolor="white", edgecolor="0.6", lw=0.6))
# axes frames placed where the 08-11 PNG has them (spines at x 324/2120 and 2476/4271 px, y 128/1551 px of 4296 x 1800)
fig.subplots_adjust(left=0.0755, right=0.9941, bottom=0.1386, top=0.9292, wspace=0.1982)
fig.savefig(f"{OUT}/figS4_structure_bridge_1005.png", dpi=600, facecolor="white")
fig.savefig(f"{OUT}/figS4_structure_bridge_1005.pdf", facecolor="white")
print("saved", f"{OUT}/figS4_structure_bridge_1005.png")
