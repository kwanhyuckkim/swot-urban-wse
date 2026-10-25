#!/usr/bin/env python3
"""figS5(변위) 재구축: 원본 export의 3중 박스플롯(Δr·w·ratio by height bin)을
인쇄 규격으로 재현하고, ratio=2 파선 대신 라벨을 단 Δr/w=1 선을 그린다.
Δr = far_building / tanθ, tanθ = |xtrk_dist_node| / sat_altitude.
모집단: parallel(≤10°) & far_building ≥ 5 m (n=17,669, Fig 8과 동일)."""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PROJ = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results"
OUT = f"{PROJ}/figures_print"

plt.rcParams.update({
    "font.size": 6.6, "axes.titlesize": 8.0, "axes.labelsize": 7.4,
    "xtick.labelsize": 6.6, "ytick.labelsize": 6.6, "legend.fontsize": 6.4,
    "axes.linewidth": 0.7, "figure.facecolor": "white"})

C_DR, C_W, C_R = "#f4a582", "#92b4d4", "#a6dba0"

d = pq.read_table(f"{PROJ}/swot_urban_nearfar_v6.5_results.parquet",
                  columns=["flow_angle", "far_building", "xtrk_dist_node",
                           "sat_altitude", "width_node", "error_node"]).to_pandas()
d = d.dropna(subset=["error_node"])
fa = d.flow_angle % 180
d["ang"] = np.minimum(fa, 180 - fa)
sub = d[(d.ang <= 10) & (d.far_building >= 5)].copy()
sub["dr"] = sub.far_building / (sub.xtrk_dist_node.abs() / sub.sat_altitude)
sub["ratio"] = sub.dr / sub.width_node
assert len(sub) == 17669, len(sub)

edges = [5, 6, 7, 8, 9, 10, 11, 12, 13, 1e9]
labs = ["5-6", "6-7", "7-8", "8-9", "9-10", "10-11", "11-12", "12-13", "13+"]

fig, ax = plt.subplots(figsize=(7.16, 3.35))
fig.subplots_adjust(left=0.075, right=0.925, top=0.90, bottom=0.185)
axr = ax.twinx()

W = 0.24
share1 = (sub.ratio >= 1).mean() * 100
for k, (lo, hi, lab) in enumerate(zip(edges[:-1], edges[1:], labs)):
    g = sub[(sub.far_building >= lo) & (sub.far_building < hi)]
    for off, vals, colr, axx in [(-W - 0.03, g.dr, C_DR, ax),
                                 (0.0, g.width_node, C_W, ax),
                                 (W + 0.03, g.ratio, C_R, axr)]:
        bp = axx.boxplot([vals.dropna()], positions=[k + off], widths=W,
                         whis=(5, 95), showfliers=False, patch_artist=True,
                         medianprops=dict(color="k", lw=1.0),
                         boxprops=dict(facecolor=colr, lw=0.6),
                         whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
ax.set_xticks(range(len(labs)))
ax.set_xticklabels([f"{l}\n(n={len(sub[(sub.far_building >= lo) & (sub.far_building < hi)]):,})"
                    for lo, hi, l in zip(edges[:-1], edges[1:], labs)],
                   fontsize=6.4)
ax.set_xlim(-0.65, len(labs) - 0.35)
ax.set_xlabel("Far building height (m), 1 m bins")
ax.set_ylabel("Distance (m)")
ax.set_ylim(0, 1300)
axr.set_ylabel("Layover/width ratio (Δr/w)", color="#1a7a2e")
axr.set_ylim(0, 9.0)
axr.tick_params(axis="y", labelcolor="#1a7a2e")
axr.axhline(1.0, color="0.35", ls="--", lw=1.0, zorder=1)
axr.text(len(labs) - 0.42, 1.08, "Δr/w = 1. Full channel within layover reach",
         fontsize=6.4, color="0.25", ha="right", va="bottom",
         bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.8))
ax.set_title("Layover displacement, channel width, and their ratio by building height",
             fontweight="bold")
ax.legend(handles=[Patch(facecolor=C_DR, label="Layover displacement Δr (left)"),
                   Patch(facecolor=C_W, label="River width w (left)"),
                   Patch(facecolor=C_R, label="Ratio Δr/w (right)")],
          loc="upper left", framealpha=0.95, edgecolor="gray", borderpad=0.4,
          handlelength=1.2)
print(f"n={len(sub):,}  ratio>=1: {share1:.1f}%  "
      f"ratio 중앙(5-6)={sub[(sub.far_building < 6)].ratio.median():.2f} "
      f"(13+)={sub[(sub.far_building >= 13)].ratio.median():.2f}")
fig.savefig(f"{OUT}/figS5_displacement_print.png", dpi=600, facecolor="white")
fig.savefig(f"{OUT}/figS5_displacement_print.pdf", facecolor="white")
print("figS5_displacement_print saved")
