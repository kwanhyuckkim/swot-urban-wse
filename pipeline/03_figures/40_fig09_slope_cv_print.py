#!/usr/bin/env python3
"""fig09_slope_cv (2026-10-04, from 39): panel (d) reads the slope-rule CV/bootstrap (26b/27b) via
CV_CSV/BOOT_CSV; CV mean and oracle mean are computed from the CV CSV. Panels (a)-(c) unchanged.
fig09_slope_angle (2026-10-04): stage-1 angle chosen WITHOUT the building/ISA thresholds,
by the submitted rule (cell 245): sum over the five category curves of the slope of cumulative
68th-percentile reduction between consecutive 2-degree grid points; take the right end of the
steepest step. The full-mask curve and its marker are removed from (a). Derived from fig09_allparallel (2026-10-03): panels (b) and (c) on the SAME population (all parallel
observations, zeros included); (a) without the uncited "All Observations" curve; no "Total dataset"
curves in (b)/(c); (d) retitled; fonts slightly larger. Derived from fig09_orig: 원 제출 Figure 9(cell 245 'Masking Optimization v2') 디자인을
인쇄 규격(7.16in, 600dpi)으로 이식 + 리뷰어 대응 요소 추가.

원본 유지: 범주명(All Observations/Far-Nadir Only/...), 청=감소·적=제거 이중축,
  θ* 세로선+교차점 도트+점선 참조선, 축변 굵은 %·cm 주석, 서브셋/전체 이중 곡선,
  (a) 모노스페이스 수치 박스, knee-point 선택(b·c), x축 반전(b·c).
추가(리뷰어 약속): (a) full-mask 감소 곡선(θ*=13°에서 최대 12.3%), 플래토 음영
  5-19°/3-10 m/4-23%, (d) reach-cluster 부트스트랩 검증 패널.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJ = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results"
import os
OUT = os.environ.get("FIG_OUT", ".")
R2 = f"{PROJ}/round2"

plt.rcParams.update({
    "font.size": 8.0, "axes.titlesize": 9.6, "axes.labelsize": 9.0,
    "xtick.labelsize": 8.0, "ytick.labelsize": 8.0, "legend.fontsize": 7.4,
    "axes.linewidth": 0.7, "xtick.direction": "in", "ytick.direction": "in",
    "figure.facecolor": "white", "axes.facecolor": "white",
})

MIN_RATIO = 0.50
CAT_ORDER = ["All", "Far", "Near", "Both", "Neither"]
COLORS = {"All": "#000000", "Far": "#D62728", "Near": "#2CA02C",
          "Both": "#1F77B4", "Neither": "#7F7F7F"}
LABELS = {"All": "All Observations", "Far": "Far-Nadir Only",
          "Near": "Near-Nadir Only", "Both": "Both Geometries",
          "Neither": "Neither Geometry"}


def load():
    cols = ["flow_angle", "far_building", "far_isa", "error_node",
            "near_isa", "near_ratio", "far_ratio"]
    df = pd.read_parquet(f"{PROJ}/swot_urban_nearfar_v6.5_results.parquet",
                         columns=cols, engine="pyarrow")
    df["abs_error"] = df.error_node.abs()
    df["flow_angle_parallel"] = df.flow_angle.apply(
        lambda a: min(a % 180, 180 - a % 180))
    df["far_isa_pct"] = df.far_isa * 100 if df.far_isa.max() <= 1.0 else df.far_isa
    return df


def main():
    df = load()
    abs_err = df.abs_error.values
    flow_par = df.flow_angle_parallel.values
    far_bldg = df.far_building.values
    far_isa = df.far_isa_pct.values

    has_far = (df.far_isa.fillna(0) > 0) & (df.far_ratio.fillna(0) >= MIN_RATIO)
    has_near = (df.near_isa.fillna(0) > 0) & (df.near_ratio.fillna(0) >= MIN_RATIO)
    CAT = {"All": np.ones(len(df), bool),
           "Far": (has_far & ~has_near).values,
           "Near": (has_near & ~has_far).values,
           "Both": (has_far & has_near).values,
           "Neither": (~has_far & ~has_near).values}

    n_total = len(df)
    p68_total = np.quantile(abs_err, 0.68)
    baselines = {n: np.quantile(abs_err[CAT[n]], 0.68) for n in CAT_ORDER}
    print(f"total n={n_total:,}  p68={p68_total:.4f}")

    # (a) flow-angle sweep, 원본 2° 그리드
    angle_thresholds = np.arange(1, 32, 2)
    results_angle = {n: [] for n in CAT_ORDER}
    for at in angle_thresholds:
        remove = flow_par < at
        for name in CAT_ORDER:
            m = CAT[name]
            ret = m & ~remove
            p68r = np.quantile(abs_err[ret], 0.68) if ret.sum() else np.nan
            results_angle[name].append(dict(
                threshold=at,
                pct_reduction=(baselines[name] - p68r) / baselines[name] * 100,
                pct_removed=(m & remove).sum() / m.sum() * 100))
    slopes_a = []
    for i in range(len(angle_thresholds) - 1):
        dx = float(angle_thresholds[i + 1] - angle_thresholds[i])
        slopes_a.append(sum((results_angle[n][i + 1]["pct_reduction"] - results_angle[n][i]["pct_reduction"]) / dx
                            for n in CAT_ORDER))
    opt_step_a = int(np.argmax(slopes_a))
    opt_idx_a = opt_step_a + 1
    OPT_A = float(angle_thresholds[opt_idx_a])
    print(f"(a) slope rule: max slope sum {slopes_a[opt_step_a]:.2f} %/deg over "
          f"{angle_thresholds[opt_step_a]}°->{OPT_A:.0f}°  => theta* = {OPT_A:.0f}°")
    print("    slope sums by step:", {f"{angle_thresholds[i]}-{angle_thresholds[i+1]}": round(s, 2) for i, s in enumerate(slopes_a)})

    # (b) building-height sweep
    is_par_bldg = (flow_par < OPT_A) & (far_bldg >= 0)
    n_sub_b = int(is_par_bldg.sum())
    p68_sub_b = np.quantile(abs_err[is_par_bldg], 0.68)
    bldg_thresholds = np.arange(20, 4.5, -0.5)
    res_b = []
    for ht in bldg_thresholds:
        remove = is_par_bldg & (far_bldg > ht)
        ret = is_par_bldg & ~remove
        p68s = np.quantile(abs_err[ret], 0.68) if ret.sum() else np.nan
        res_b.append(dict(
            threshold=ht,
            red_sub=(p68_sub_b - p68s) / p68_sub_b * 100,
            rm_sub=remove.sum() / n_sub_b * 100,
            red_tot=(p68_total - np.quantile(abs_err[~remove], 0.68)) / p68_total * 100,
            rm_tot=remove.sum() / n_total * 100))
    red_bn = np.array([r["red_sub"] for r in res_b])
    rm_bn = np.array([r["rm_sub"] for r in res_b])
    knee_b = red_bn / np.nanmax(red_bn) - rm_bn / rm_bn.max()
    oib = int(np.argmax(knee_b))
    print(f"(b) subset n={n_sub_b:,} p68={p68_sub_b:.4f}  θ*={res_b[oib]['threshold']}m"
          f"  red={res_b[oib]['red_sub']:.1f}% rm={res_b[oib]['rm_sub']:.1f}%")

    # (c) ISA sweep
    is_par_isa = (flow_par < OPT_A) & (far_isa >= 0)   # 2026-10-03: all parallel, zeros included
    n_sub_c = int(is_par_isa.sum())
    p68_sub_c = np.quantile(abs_err[is_par_isa], 0.68)
    isa_thresholds = np.arange(63, 4, -2)
    res_c = []
    for it in isa_thresholds:
        remove = is_par_isa & (far_isa > it)
        ret = is_par_isa & ~remove
        p68s = np.quantile(abs_err[ret], 0.68) if ret.sum() else np.nan
        res_c.append(dict(
            threshold=it,
            red_sub=(p68_sub_c - p68s) / p68_sub_c * 100,
            rm_sub=remove.sum() / n_sub_c * 100,
            red_tot=(p68_total - np.quantile(abs_err[~remove], 0.68)) / p68_total * 100,
            rm_tot=remove.sum() / n_total * 100))
    red_cn = np.array([r["red_sub"] for r in res_c])
    rm_cn = np.array([r["rm_sub"] for r in res_c])
    knee_c = red_cn / np.nanmax(red_cn) - rm_cn / rm_cn.max()
    oic = int(np.argmax(knee_c))
    print(f"(c) subset n={n_sub_c:,} p68={p68_sub_c:.4f}  θ*={res_c[oic]['threshold']}%"
          f"  red={res_c[oic]['red_sub']:.1f}% rm={res_c[oic]['rm_sub']:.1f}%")
    print(f"(b) exact: red {res_b[oib]['red_sub']:.4f}% rm {res_b[oib]['rm_sub']:.4f}% drop {p68_sub_b*res_b[oib]['red_sub']:.3f} cm")
    print(f"(c) exact: red {res_c[oic]['red_sub']:.4f}% rm {res_c[oic]['rm_sub']:.4f}% drop {p68_sub_c*res_c[oic]['red_sub']:.3f} cm")


    fig, axes = plt.subplots(2, 2, figsize=(7.16, 7.5))

    # (a)
    ax = axes[0][0]
    xa = [r["threshold"] for r in results_angle["All"]]
    for name in CAT_ORDER:
        ax.plot(xa, [r["pct_reduction"] for r in results_angle[name]],
                color=COLORS[name], ls="-", marker="o", ms=1.9, lw=1.0,
                label=LABELS[name], zorder=4)
    ax.axvline(OPT_A, color="#444444", ls=":", lw=1.1, alpha=0.85, zorder=3)
    ax.set_xlabel("Flow Angle Threshold (°)")
    ax.set_ylabel("Cumul. 68th %ile Error Reduction (%)", color="blue")
    ax.tick_params(axis="y", labelcolor="blue")
    ax.set_title("(a) Flow Angle Masking", fontweight="bold")
    ax.grid(True, alpha=0.25, lw=0.4)
    ax2 = ax.twinx()
    vals_rm = [r["pct_removed"] for r in results_angle["All"]]
    ax2.plot(xa, vals_rm, color="red", ls="--", lw=0.9, alpha=0.6,
             label="Removed, all obs. (%)")
    ax2.set_ylabel("Observations Removed (%)", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax3 = ax.twinx()
    mids = angle_thresholds[1:].astype(float)  # slope of each 2-deg step drawn at the step's upper end (the cutoff that realizes it)
    ax3.plot(mids, slopes_a, color="k", ls="--", lw=1.0, zorder=5)
    ax3.plot(mids[opt_step_a], slopes_a[opt_step_a], "D", color="k", ms=3.4,
             markeredgecolor="white", markeredgewidth=0.5, zorder=6)
    ax3.set_yticks([]); ax3.spines["right"].set_visible(False)
    ax3.set_ylim(0, max(slopes_a) * 1.6)
    for name in CAT_ORDER:
        ax.plot(OPT_A, results_angle[name][opt_idx_a]["pct_reduction"], "o",
                color=COLORS[name], ms=3.4, markeredgecolor="white",
                markeredgewidth=0.5, zorder=6)
    rm_all = results_angle["All"][opt_idx_a]["pct_removed"]
    ax2.plot(OPT_A, rm_all, "s", color="red", ms=3.0, markeredgecolor="white",
             markeredgewidth=0.5, zorder=6)
    head = [f"θ* = {OPT_A:.0f}°", f"Max summed slope: {angle_thresholds[opt_step_a]}°→{OPT_A:.0f}°",
            f"Removed (All): {rm_all:.1f}%"]
    ax.text(0.02, 0.985, "\n".join(head), transform=ax.transAxes, fontsize=7.4,
            va="top", ha="left", family="monospace", fontweight="bold", zorder=7,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFFFF0",
                      edgecolor="gray", lw=0.5, alpha=0.94))
    SIDE = {"Neither": "L", "Near": "L", "All": "R", "Far": "R", "Both": "R"}
    for name in CAT_ORDER:
        v = results_angle[name][opt_idx_a]["pct_reduction"]
        txt = f"{v:.1f}% ({baselines[name] * v:.1f} cm)"
        if SIDE[name] == "L":
            ax.text(OPT_A - 0.7, v, txt, fontsize=7.2, fontweight="bold",
                    color=COLORS[name], ha="right", va="center", zorder=7,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.5,
                              alpha=0.75))
        else:
            ax.text(OPT_A + 0.7, v, txt, fontsize=7.2, fontweight="bold",
                    color=COLORS[name], ha="left", va="center", zorder=7,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.5,
                              alpha=0.75))
    l1, b1 = ax.get_legend_handles_labels()
    l2, b2 = ax2.get_legend_handles_labels()
    l1.append(Line2D([0], [0], color="k", ls="--", lw=1.0))
    b1.append("Summed slope, five categories")
    ax.legend(l1 + l2, b1 + b2, loc="upper center", bbox_to_anchor=(0.5, -0.19),
              fontsize=6.6, ncol=3, framealpha=0.95, edgecolor="gray",
              borderpad=0.35, handlelength=1.3, labelspacing=0.35,
              columnspacing=0.7, handletextpad=0.4)

    # (b)/(c) 공통
    def panel_bc(ax, res, opt_i, n_sub, p68_sub, span, xlab, title, unit, leg):
        xs = [r["threshold"] for r in res]
        ax.axvspan(*span, color="#1f4e9c", alpha=0.08, zorder=1)
        ax.plot(xs, [r["red_sub"] for r in res], "b-o", ms=1.9, lw=1.1,
                label=f"Parallel obs. (n={n_sub:,})", zorder=4)
        ax.set_xlabel(xlab)
        ax.set_ylabel("Cumul. 68th %ile Error Reduction (%)", color="blue")
        ax.tick_params(axis="y", labelcolor="blue")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, alpha=0.25, lw=0.4)
        ax.invert_xaxis()
        axr = ax.twinx()
        axr.plot(xs, [r["rm_sub"] for r in res], "r-s", ms=1.7, lw=0.95,
                 label="Obs. removed (%)", zorder=4)
        axr.set_ylabel("Observations Removed (%)", color="red")
        axr.tick_params(axis="y", labelcolor="red")
        r = res[opt_i]
        ox = r["threshold"]
        ax.axvline(ox, color="#444444", ls=":", lw=1.1, alpha=0.85, zorder=3)
        ax.plot(ox, r["red_sub"], "o", color="blue", ms=3.4,
                markeredgecolor="white", markeredgewidth=0.5, zorder=6)
        axr.plot(ox, r["rm_sub"], "s", color="red", ms=3.0,
                 markeredgecolor="white", markeredgewidth=0.5, zorder=6)
        xl = ax.get_xlim()          # inverted: xl[0] > xl[1]
        ax.plot([xl[0], ox], [r["red_sub"]] * 2, color="blue", ls=":",
                lw=0.6, alpha=0.4, zorder=2)
        axr.plot([ox, xl[1]], [r["rm_sub"]] * 2, color="red", ls=":",
                 lw=0.6, alpha=0.4, zorder=2)
        drop_cm = (round(p68_sub, 3) - round(p68_sub * (1 - r['red_sub'] / 100), 3)) * 100  # from the 3-dp metre values quoted in the text
        ax.annotate(f"{r['red_sub']:.1f}%\n({drop_cm:.1f} cm)",
                    xy=(0.015, r["red_sub"]),
                    xycoords=("axes fraction", "data"), fontsize=8.0,
                    color="blue", fontweight="bold", va="bottom", ha="left",
                    zorder=7)
        axr.annotate(f"{r['rm_sub']:.1f}%", xy=(0.985, r["rm_sub"]),
                     xycoords=("axes fraction", "data"), fontsize=8.0,
                     color="red", fontweight="bold", va="bottom", ha="right",
                     zorder=7)
        tr = blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(ox - 0.022 * abs(xl[0] - xl[1]), 0.845,
                f"θ* = {ox:.0f} {unit}", fontsize=8.4, fontweight="bold",
                color="#333333", ha="right", va="top", transform=tr, zorder=7)
        la, ba = ax.get_legend_handles_labels()
        lb, bb = axr.get_legend_handles_labels()
        ax.legend(la + lb, ba + bb, loc="upper center",
                  bbox_to_anchor=(0.5, -0.19), ncol=2, fontsize=7.0,
                  framealpha=0.95, edgecolor="gray", borderpad=0.4,
                  handlelength=1.5, labelspacing=0.4, columnspacing=1.0)

    panel_bc(axes[0][1], res_b, oib, n_sub_b, p68_sub_b, (3, 10),
             "Far-Nadir Building Height Threshold (m)",
             f"(b) Building Height Masking (Parallel < {OPT_A:.0f}°)", "m",
             dict(loc="center left", bbox_to_anchor=(0.015, 0.60)))
    panel_bc(axes[1][0], res_c, oic, n_sub_c, p68_sub_c, (4, 23),
             "Far-Nadir ISA Threshold (%)",
             f"(c) ISA Masking (Parallel < {OPT_A:.0f}°)", "%",
             dict(loc="lower right", bbox_to_anchor=(0.985, 0.03)))

    # (d) 검증
    axd = axes[1][1]
    boot = pd.read_csv(os.environ.get("BOOT_CSV", f"{R2}/threshold_bootstrap_slope.csv"))
    cv = pd.read_csv(os.environ.get("CV_CSV", f"{R2}/crossval_slope.csv"))
    CV_MEAN, ORC_MEAN, POOLED = cv.held.mean(), cv.orc.mean(), 12.3
    print(f"(d) CV mean {CV_MEAN:.2f} sd {cv.held.std(ddof=1):.2f}  oracle {ORC_MEAN:.2f}  "
          f"boot median {boot.reduction_pct.median():.2f}")
    axd.hist(boot.reduction_pct, bins=24, color="#9ecae1", edgecolor="white",
             lw=0.4)
    lo, hi = boot.reduction_pct.quantile([.025, .975])
    axd.axvspan(lo, hi, color="#1f4e9c", alpha=0.10)
    for v, lab, colr, side in [(POOLED, f"pooled in-sample {POOLED:.1f}%", "#c62828", "left"),
                               (CV_MEAN, f"cross-validated {CV_MEAN:.1f}%", "#1f4e9c", "right"),
                               (ORC_MEAN, f"oracle ceiling {ORC_MEAN:.1f}%", "#555555", "right")]:
        axd.axvline(v, color=colr, lw=1.0, ls="--" if v != POOLED else "-")
        _ty = 0.985 if v != ORC_MEAN else 0.02
        _va = "top" if v != ORC_MEAN else "bottom"
        _tr = blended_transform_factory(axd.transData, axd.transAxes)
        axd.text(v, _ty, lab, rotation=90, va=_va, transform=_tr,
                 ha=side, fontsize=7.4, color=colr,
                 bbox=dict(facecolor="white", edgecolor="none", pad=0.4,
                           alpha=0.75))
    axd.text(0.46, 0.80, f"95% CI [{lo:.1f}, {hi:.1f}]",
             transform=axd.transAxes, ha="left", va="top", fontsize=7.6,
             bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.8))
    axd.set_title("(d) Reach Bootstrap, Full Mask (B = 300)",
                  fontweight="bold")
    axd.set_xlabel("Full-Mask 68th %ile Error Reduction (%)")
    axd.set_ylabel("Replicates")

    fig.subplots_adjust(left=0.085, right=0.925, top=0.975, bottom=0.15,
                        wspace=0.46, hspace=0.56)
    fig.savefig(f"{OUT}/fig09_slope_cv_print.png", dpi=600, facecolor="white")
    fig.savefig(f"{OUT}/fig09_slope_cv_print.pdf", facecolor="white")
    print("fig09_allparallel_print saved")


if __name__ == "__main__":
    main()
