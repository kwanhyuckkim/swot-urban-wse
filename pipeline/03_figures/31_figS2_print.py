#!/usr/bin/env python
"""Supplementary S2, built for the first time, plus a 600 dpi re-export of S6.

The submitted supplement pairs the S2 caption, conditional far-nadir ISA error
distributions within each building height bin, with an image that actually shows
the near/far ISA comparison. The figure the caption and Section IV describe was
never exported. This script builds it as the exact mirror of the S6 figure
(notebook cell 152): same parallel zone (flow angle within 10 degrees of the
track), same population (far building height at or above 5 m, n = 17,669), same
anchor-and-band construction, with the roles of ISA and building height swapped.

S6 is re-exported unchanged at 600 dpi from the same mechanics; it is adopted
only if its population reproduces the published n = 17,669 exactly.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pyarrow.parquet as pq

PROJ = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results"
NODES = f"{PROJ}/swot_urban_nearfar_v6.5_results.parquet"
OUT = f"{PROJ}/figures_print"
T = 0.1832

plt.style.use("default")
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"


def load():
    d = pq.read_table(NODES, columns=["flow_angle", "far_isa", "far_building",
                                      "error_node"]).to_pandas()
    d["e"] = d.error_node.abs()
    fa = d.flow_angle % 360
    d["angle_dev"] = np.minimum.reduce([fa, (fa - 180).abs(), 360 - fa])
    d["isa"] = d.far_isa * 100 if d.far_isa.max() <= 1 else d.far_isa
    d["bld"] = d.far_building
    d = d.dropna(subset=["e", "flow_angle"])
    par_isa = d[d.angle_dev <= 10].dropna(subset=["isa"])
    print(f"parallel zone with ISA: n={len(par_isa):,} (image2 shows 55,328)")
    s = d.dropna(subset=["isa", "bld"])
    s = s[(s.angle_dev <= 10) & (s.bld >= 5)]
    print(f"parallel & building >= 5 m: n={len(s):,} (published 17,669)")
    return s


def get_bin(value, edges, labels):
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return labels[i]
    return None


def conditional(ax, df, x_col, cond_col, edges, cond_clip, tol,
                fill_color, line_color, cond_name,
                offset=1, quantiles=(0, 5, 25, 50, 75, 95, 100), min_n=30):
    """One anchored-band conditional panel, the cell-152 construction."""
    labels = [f"{int(edges[i])}-{int(edges[i+1])}" if np.isfinite(edges[i + 1])
              else f"{int(edges[i])}+" for i in range(len(edges) - 1)]
    df = df.copy()
    df["xbin"] = df[x_col].apply(lambda v: get_bin(v, edges, labels))
    df = df.dropna(subset=["xbin", cond_col])

    centers = [(edges[i] + edges[i + 1]) / 2 if np.isfinite(edges[i + 1])
               else edges[i] + 1 for i in range(len(edges) - 1)]
    n_per = np.array([len(df[df.xbin == lb]["e"].dropna()) for lb in labels])
    valid_bins = n_per >= min_n

    anchors = np.percentile(df[cond_col].dropna(), quantiles)
    lines = []
    for q in anchors:
        lo = max(cond_clip[0], q - tol)
        hi = min(cond_clip[1], q + tol)
        sub = df[(df[cond_col] >= lo) & (df[cond_col] <= hi)]
        if len(sub) < 50:
            continue
        ys = []
        for i, lb in enumerate(labels):
            if not valid_bins[i]:
                ys.append(np.nan)
                continue
            bd = sub[sub.xbin == lb]["e"].dropna()
            ys.append(np.percentile(bd, 68) if len(bd) >= 10 else np.nan)
        lines.append(ys)
    lines = np.array(lines)
    print(f"  anchors kept: {len(lines)}/{len(anchors)}")

    with np.errstate(all="ignore"):
        med = np.nanmedian(lines, axis=0) + offset
        q25 = np.nanpercentile(lines, 25, axis=0) + offset
        q75 = np.nanpercentile(lines, 75, axis=0) + offset
        q05 = np.nanpercentile(lines, 5, axis=0) + offset
        q95 = np.nanpercentile(lines, 95, axis=0) + offset
        q00 = np.nanmin(lines, axis=0) + offset
        q99 = np.nanmax(lines, axis=0) + offset
    vm = ~np.isnan(med - offset)
    xv = np.array(centers)[vm]

    ax.fill_between(xv, q00[vm], q99[vm], alpha=0.15, color=fill_color,
                    label=f"{cond_name} 0-100%")
    ax.fill_between(xv, q05[vm], q95[vm], alpha=0.25, color=fill_color,
                    label=f"{cond_name} 5-95%")
    ax.fill_between(xv, q25[vm], q75[vm], alpha=0.45, color=fill_color,
                    label=f"{cond_name} 25-75%")
    ax.plot(xv, med[vm], "o-", color=line_color, linewidth=2.5, markersize=8,
            label=f"{cond_name} Median")

    if valid_bins[-1]:
        ax.text(centers[-1], offset * 1.5, labels[-1], ha="center", va="top",
                fontsize=14, color="black", fontweight="bold")
    for x, y, n in zip(xv, med[vm], n_per[vm]):
        ax.text(x, y * 1.1, f"n={n:,}", ha="center", va="bottom", fontsize=11,
                color="black", fontweight="bold", rotation=90)

    ax.axhline(y=T + offset, color="black", linestyle="--", linewidth=2, alpha=0.8)
    x_mid = (edges[0] + edges[-2]) / 2
    ax.text(x_mid, (T + offset) * 0.95, f"T={T}m", fontsize=12,
            fontweight="bold", va="top", ha="center")

    band_top = np.nanmax(q99 - offset)
    ymax = max(5, band_top * 1.15)
    ticks = [t for t in (0, 1, 2, 4, 8, 16) if t <= ymax]

    ax.set_yscale("log")
    ax.set_ylim(offset * 0.9, ymax + offset)

    def fmt(v, _):
        o = v - offset
        for t in ticks:
            if abs(o - t) < 0.01:
                return str(t)
        return ""
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_yticks([t + offset for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(edges[0] - 1, edges[-2] + 3)
    ax.set_ylabel(r"68th Percentile $|Relative\ WSE\ Error|$ (m)",
                  color="black", fontsize=13)
    ax.tick_params(colors="black", labelsize=11)
    ax.grid(True, alpha=0.3, color="gray")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    for sp in ax.spines.values():
        sp.set_color("black")


def figS2(s):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), facecolor="white")
    conditional(ax, s, x_col="isa", cond_col="bld",
                edges=np.array([0, 5, 10, 15, 20, 25, 30, 40, 60, np.inf]),
                cond_clip=(5, s.bld.max()), tol=1.0,
                fill_color="steelblue", line_color="navy",
                cond_name="Far Building")
    ax.set_xlabel("Far ISA (%)", color="black", fontsize=13)
    ax.set_title(f"Far ISA (by Far Building Height, n={len(s):,})",
                 fontsize=14, fontweight="bold", pad=5)
    plt.tight_layout()
    fig.savefig(f"{OUT}/figS2_isa_by_height.png", dpi=600, facecolor="white",
                bbox_inches="tight")
    fig.savefig(f"{OUT}/figS2_isa_by_height.pdf", facecolor="white",
                bbox_inches="tight")
    plt.close(fig)
    print("figS2 saved")


def figS6(s):
    if len(s) != 17669:
        print(f"figS6 SKIPPED: population {len(s):,} != 17,669; keeping the "
              "original image3")
        return
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), facecolor="white")
    conditional(ax, s, x_col="bld", cond_col="isa",
                edges=np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, np.inf]),
                cond_clip=(0, 100), tol=5.0,
                fill_color="peru", line_color="darkorange",
                cond_name="Far ISA")
    ax.set_xlabel("Far Building Height (m)", color="black", fontsize=13)
    ax.set_title(f"Far Building Height (by Far ISA, n={len(s):,})",
                 fontsize=14, fontweight="bold", pad=5)
    plt.tight_layout()
    fig.savefig(f"{OUT}/figS6_height_by_isa.png", dpi=600, facecolor="white",
                bbox_inches="tight")
    fig.savefig(f"{OUT}/figS6_height_by_isa.pdf", facecolor="white",
                bbox_inches="tight")
    plt.close(fig)
    print("figS6 saved")


if __name__ == "__main__":
    s = load()
    for a in (sys.argv[1:] or ["figS2", "figS6"]):
        globals()[a](s)
