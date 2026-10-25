#!/usr/bin/env python3
"""figS4(->리넘버 후 figS3) 지역 분해 재구축: 기존 디자인 유지 + (b) y상한 확장으로
Low. Miss. 라벨이 레전드에 안 가리게. 데이터: supp_figS4_data/*.csv (13_region_geometry 산출).
검증 목표: Spearman(exposure, reduction)=+0.74(p=0.001), (parallel, reduction)=-0.37(p=0.16)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

BASE = "/Users/targetkim/target/SWOT_urbanerror"
OUT = f"{BASE}/figures_print"

plt.rcParams.update({
    "font.size": 6.6, "axes.titlesize": 8.0, "axes.labelsize": 7.4,
    "xtick.labelsize": 6.4, "ytick.labelsize": 6.6, "legend.fontsize": 6.2,
    "axes.linewidth": 0.7, "figure.facecolor": "white"})

g = pd.read_csv(f"{BASE}/supp_figS4_data/figS4_region_geometry.csv")
z = pd.read_csv(f"{BASE}/supp_figS4_data/figS4_zone_long.csv")
g = g.sort_values("exposure_flagged_pct", ascending=False).reset_index(drop=True)

rho_e, p_e = spearmanr(g.exposure_flagged_pct, g.reduction_pct)
rho_a, p_a = spearmanr(g.parallel_pct, g.reduction_pct)
print(f"exposure ρ={rho_e:.2f} (p={p_e:.3f}) | alignment ρ={rho_a:.2f} (p={p_a:.2f})")

fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.16, 3.5),
                               gridspec_kw=dict(width_ratios=[0.92, 1.18],
                                                wspace=0.28))
fig.subplots_adjust(left=0.075, right=0.90, top=0.92, bottom=0.20)

# ═══ (a) 존 구성 적층바 ═══
SHORT = {"Lower Colorado": "Low. Colo.",
         "Upper Colorado": "Up. Colo.", "Lower Mississippi": "Low. Miss.", "Upper Mississippi": "Up. Miss.",
         "New England": "New Eng.", "Great Lakes": "Gr. Lakes",
         "Mid-Atlantic": "Mid-Atl.", "Arkansas-White-Red": "Ark.-W.-Red",
         "South Atlantic-Gulf": "S. Atl.-Gulf", "Pacific Northwest": "Pac. NW",
         "Texas-Gulf": "Tex.-Gulf", "California": "Calif."}
ZC = {"Near": "#6baed6", "Far": "#fd8d3c", "Both": "#c23b22"}
order = g.region.tolist()
zp = z.pivot_table(index="region", columns="zone", values="share_pct").reindex(order)
x = np.arange(len(order))
bot = np.zeros(len(order))
for zone in ("Near", "Far", "Both"):
    axa.bar(x, zp[zone].values, bottom=bot, color=ZC[zone], width=0.72,
            label=zone, zorder=3)
    bot += zp[zone].values
axa.scatter(x, g.parallel_pct, facecolor="white", edgecolor="black", s=13,
            lw=0.8, zorder=4, label="parallel share")
axa.set_xticks(x)
axa.set_xticklabels([SHORT.get(r, r) for r in order], rotation=55,
                    ha="right", fontsize=6.4)
axa.set_ylabel("share of node observations (%)")
axa.set_title("(a) Urbanized-zone composition by region", fontweight="bold")
axa.legend(loc="upper right", ncol=2, framealpha=0.95, edgecolor="gray",
           borderpad=0.4, handlelength=1.2, columnspacing=0.8)
axa.text(0.98, 0.50, "remainder of each region is the Neither zone;\n"
         "regions ordered by exposure \N{RIGHTWARDS ARROW}",
         transform=axa.transAxes, ha="right", fontsize=5.8, color="0.4")

# ═══ (b) 노출 vs 감소 ═══
resp = g.response_p68_ratio
ok = resp.notna()
sc = axb.scatter(g[ok].exposure_flagged_pct, g[ok].reduction_pct,
                 s=8 + 240 * g[ok].n_obs / g.n_obs.max(),
                 c=resp[ok], cmap="RdYlBu_r", vmin=0.75, vmax=3.0,
                 edgecolor="k", lw=0.5, zorder=4)
axb.scatter(g[~ok].exposure_flagged_pct, g[~ok].reduction_pct, s=16,
            facecolor="white", edgecolor="0.35", lw=0.8, zorder=5)
axb.axhline(0, color="0.6", ls=":", lw=0.7, zorder=2)
axb.set_xlabel("exposure: parallel observations flagged (%)")
axb.set_ylabel("68th-percentile error reduction (%)")
axb.set_title("(b) Reduction follows exposure", fontweight="bold")
axb.set_ylim(-13, 45)          # 상단 헤드룸 확장: Low. Miss. 라벨이 레전드 아래로
axb.set_xlim(-5, 72)
LAB_OFF = {"Lower Mississippi": (2, -10), "New England": (5, 4),
           "Ohio": (-6, 6), "Great Lakes": (10, 5), "Upper Mississippi": (6, -8),
           "Mid-Atlantic": (8, -3), "Arkansas-White-Red": (0, 7),
           "South Atlantic-Gulf": (-19, 2), "Pacific Northwest": (15, 6),
           "Texas-Gulf": (20, -6), "California": (-13, -4), "Missouri": (-15, -3)}
for _, r in g.iterrows():
    if r.n_flagged < 100 and r.exposure_flagged_pct < 1:
        continue
    dx, dy = LAB_OFF.get(r.region, (5, 3))
    axb.annotate(SHORT.get(r.region, r.region),
                 (r.exposure_flagged_pct, r.reduction_pct),
                 xytext=(dx, dy), textcoords="offset points",
                 fontsize=6.2, ha="center", zorder=6)
axb.annotate("Colorado \N{MULTIPLICATION SIGN}2, Tennessee, Alaska\n"
             "(highest alignment, near-zero far-bank\nexposure; mask flags <0.4%)",
             xy=(0.5, 0.0), xytext=(20, -9.5), fontsize=5.9, color="0.35",
             arrowprops=dict(arrowstyle="-", lw=0.6, color="0.5"), zorder=5)
axb.text(0.975, 0.975,
         f"exposure: \N{GREEK SMALL LETTER RHO} = {rho_e:.2f} (p = {p_e:.3f})\n"
         f"alignment: \N{GREEK SMALL LETTER RHO} = {rho_a:.2f} (p = {p_a:.2f})\n"
         "marker size \N{PROPORTIONAL TO} observations",
         transform=axb.transAxes, va="top", ha="right", fontsize=6.0, zorder=6,
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                   edgecolor="gray", lw=0.6, alpha=0.95))
cb = fig.colorbar(sc, ax=axb, fraction=0.045, pad=0.02)
cb.set_label("response: p68(flagged) / p68(unflagged)", fontsize=6.6)
cb.ax.tick_params(labelsize=6.0)

fig.savefig(f"{OUT}/figS4_region_decomposition.png", dpi=600, facecolor="white")
fig.savefig(f"{OUT}/figS4_region_decomposition.pdf", facecolor="white")
print("figS4_region_decomposition saved")
