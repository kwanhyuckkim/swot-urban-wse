#!/usr/bin/env python3
"""figS1_thresholds: RSE S1/S2 디자인 언어(분위 밴드+중앙값+p68+bin별 n)를
세 임계 변수에 적용한 dose-response 3패널. Figure 9(a,b,c)와 1:1 대응.

(a) Flow angle, 전체 관측, 2° bin, 0-60°  -> θ* = 13°
(b) Far-nadir building height, parallel(<13°), 1 m bin -> θ* = 8 m
(c) Far-nadir ISA, parallel(<13°) & ISA>0, 5% bin -> θ* = 19%

각 패널: 5-95% 밴드, 25-75% 밴드, Median 선, 68th percentile 선,
T=0.18 m 대시, θ* 점선, bin별 n 라벨(회전).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

PROJ = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results"
OUT = f"{PROJ}/figures_print"
T = 0.1832

plt.rcParams.update({
    "font.size": 6.2, "axes.titlesize": 7.4, "axes.labelsize": 7.0,
    "xtick.labelsize": 6.2, "ytick.labelsize": 6.2, "legend.fontsize": 5.4,
    "axes.linewidth": 0.7, "xtick.direction": "in", "ytick.direction": "in",
    "figure.facecolor": "white", "axes.facecolor": "white",
})

C_B95 = "#cfe0f0"   # 5-95% band
C_B50 = "#8fb8dc"   # 25-75% band
C_MED = "#1f4e9c"   # median
C_P68 = "#c62828"   # 68th percentile


def load():
    cols = ["flow_angle", "far_building", "far_isa", "error_node",
            "sample_geometry_category"]
    df = pd.read_parquet(f"{PROJ}/swot_urban_nearfar_v6.5_results.parquet",
                         columns=cols, engine="pyarrow")
    df["e"] = df.error_node.abs()
    df["angle_dev"] = df.flow_angle.apply(lambda a: min(a % 180, 180 - a % 180))
    df["far_isa_pct"] = df.far_isa * 100 if df.far_isa.max() <= 1.0 else df.far_isa
    return df.dropna(subset=["e"])


def panel(ax, sub, var, edges, theta, theta_label, xlab, title, min_n=100,
          n_fs=4.0, n_step=1, show_n=True, ymax=2.0, theta_y=0.845):
    cent, q25, q50, q75, p68, ns = [], [], [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        g = sub[(sub[var] >= lo) & (sub[var] < hi)]
        if len(g) < min_n:
            continue
        cent.append((lo + hi) / 2)
        qs = np.quantile(g.e, [0.25, 0.50, 0.75, 0.68])
        q25.append(qs[0]); q50.append(qs[1]); q75.append(qs[2]); p68.append(qs[3])
        ns.append(len(g))
    cent = np.array(cent)
    ax.fill_between(cent, q25, q75, color=C_B50, alpha=0.8, lw=0,
                    label="25-75% (IQR)", zorder=3)
    q50a = np.array(q50)
    q50m = np.where(q50a > ymax, np.nan, q50a)
    ax.plot(cent, q50m, color=C_MED, lw=1.1, marker="o", ms=1.7,
            label="Median", zorder=5)
    for x, v in [(x, v) for x, v in zip(cent, q50a) if v > ymax]:
        ax.plot(x, ymax * 0.96, marker="^", color=C_MED, ms=3.4,
                markeredgecolor="white", markeredgewidth=0.4, zorder=7)
        ax.annotate(f"{v:.1f}", (x, ymax * 0.905), rotation=90, fontsize=6.4,
                    color=C_MED, fontweight="bold", ha="center", va="top",
                    zorder=8)
    p68a = np.array(p68)
    overflow = [(x, v) for x, v in zip(cent, p68a) if v > ymax]
    p68m = np.where(p68a > ymax, np.nan, p68a)
    ax.plot(cent, p68m, color=C_P68, lw=1.3, marker="s", ms=1.7,
            label="68th percentile", zorder=6)
    for x, v in overflow:
        ax.plot(x, ymax * 0.96, marker="^", color=C_P68, ms=3.4,
                markeredgecolor="white", markeredgewidth=0.4, zorder=7)
        ax.annotate(f"{v:.1f}", (x, ymax * 0.905), rotation=90, fontsize=6.4,
                    color=C_P68, fontweight="bold", ha="center", va="top",
                    zorder=8)
    ax.axhline(T, color="k", ls="--", lw=0.9, zorder=4)
    ax.axvline(theta, color="#333333", ls=":", lw=1.1, alpha=0.9, zorder=4)
    ax.set_ylim(0, ymax)
    ax.set_xlabel(xlab)
    ax.set_title(title, fontweight="bold")
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(theta, theta_y, theta_label, transform=tr, fontsize=6.6,
            fontweight="bold", color="#333333", ha="left", va="top",
            rotation=0, zorder=7,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.75))
    ax.text(0.985, T + 0.02 * ymax, "T = 0.18 m",
            transform=blended_transform_factory(ax.transAxes, ax.transData),
            fontsize=5.8, ha="right", va="bottom", color="k", zorder=7)
    if show_n:
        over_x = {x for x, _ in overflow}
        for k, (x, n) in enumerate(zip(cent, ns)):
            if k % n_step or x in over_x:
                continue
            ax.text(x, 0.985, f"n={n:,}", rotation=90, fontsize=n_fs,
                    ha="center", va="top", color="0.25", zorder=7,
                    transform=blended_transform_factory(ax.transData,
                                                        ax.transAxes))
    return cent, np.array(p68), np.array(q50), ns


def main():
    df = load()
    gc = df.sample_geometry_category.astype(str).str.lower()
    zmap = {"na": "Neither", "near": "Near", "far": "Far", "near/far": "Both"}
    df["zone"] = gc.map(zmap)

    fig = plt.figure(figsize=(7.16, 6.05))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.25], hspace=0.80,
                          wspace=0.10, left=0.068, right=0.99, top=0.945,
                          bottom=0.085)

    # 1행: 각도: (a) Far∪Both 결정 패널, (b) Near, (c) Neither 대조
    groups = [("Far-side urban (Far ∪ Both)", df.zone.isin(["Far", "Both"])),
              ("Near only", df.zone == "Near"),
              ("Neither (control)", df.zone == "Neither")]
    letters = ["a", "b", "c"]
    for k, (gname, gmask) in enumerate(groups):
        ax = fig.add_subplot(gs[0, 2*k:2*k+2])
        sub = df[gmask]
        n60 = int((sub.angle_dev < 60).sum())
        cent, p68z, medz, nsz = panel(
            ax, sub, "angle_dev", np.arange(0, 62, 2), 13, "",
            "", f"({letters[k]}) {gname} (n = {n60:,})",
            show_n=False, ymax=2.0)
        if k == 0:
            ax.set_ylabel(r"$|Relative\ WSE\ Error|$ (m)")
            ax.legend(loc="upper right", framealpha=0.92, edgecolor="gray",
                      borderpad=0.35, handlelength=1.3, labelspacing=0.3,
                      fontsize=5.6)
            tr = blended_transform_factory(ax.transData, ax.transAxes)
            ax.text(14.0, 0.985, "θ* = 13\N{DEGREE SIGN}", transform=tr,
                    fontsize=6.2, fontweight="bold", color="#333333",
                    ha="left", va="top")
            print(f"(a) Far∪Both p68 by bin: " + " ".join(
                f"{c:.0f}°:{v:.2f}" for c, v in zip(cent[:9], p68z[:9])))
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("")
    fig.text(0.53, 0.520, "Flow-angle deviation from track (\N{DEGREE SIGN}), "
             "2\N{DEGREE SIGN} bins", ha="center", fontsize=7.0)

    # 2행: 높이·ISA (parallel<13°)
    axb = fig.add_subplot(gs[1, 0:3])
    b = df[(df.angle_dev < 13) & (df.far_building >= 0)]
    cent, p68b, medb, nsb = panel(
        axb, b, "far_building", np.arange(0, 21, 1), 8,
        " θ* = 8 m", "Far-nadir building height (m), 1 m bins",
        f"(d) Far-Nadir Building Height (parallel < 13\N{DEGREE SIGN}; "
        f"n = {len(b):,})", n_step=2, n_fs=6.0, ymax=1.75, theta_y=0.58)
    axb.set_ylabel(r"$|Relative\ WSE\ Error|$ (m)")
    print(f"(d) subset n={len(b):,} p68 7-8m={p68b[7]:.2f} 8-9m={p68b[8]:.2f}")

    axc = fig.add_subplot(gs[1, 3:6])
    c = df[(df.angle_dev < 13) & (df.far_isa_pct > 0)]
    cent, p68c, medc, nsc = panel(
        axc, c, "far_isa_pct", np.arange(0, 85, 5), 19,
        " θ* = 19 %", "Far-nadir ISA (%), 5% bins",
        f"(e) Far-Nadir ISA (parallel < 13\N{DEGREE SIGN}; n = {len(c):,})",
        n_step=2, n_fs=6.0, ymax=2.2, theta_y=0.50)
    axc.set_yticklabels([])
    print(f"(e) subset n={len(c):,}")

    fig.savefig(f"{OUT}/figS1_thresholds.png", dpi=600, facecolor="white")
    fig.savefig(f"{OUT}/figS1_thresholds.pdf", facecolor="white")
    print("figS1_thresholds saved")


if __name__ == "__main__":
    main()
