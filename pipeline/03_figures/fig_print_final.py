#!/usr/bin/env python
"""
Print-ready figure set for TGRS-2026-02661, drawn at final IEEE size.

Every figure is laid out at its true print width (3.5 in single column, 7.16 in
double column) with fonts of 8 pt or larger, so what you see is what the reviewer
gets. Saved without bbox cropping so the physical size is exact; 600 dpi PNG + PDF.

  fig03  two-pass zone-flip illustration from a real study reach (2 col)
  fig04  error by reach ISA category (1 col)
  fig05  quality-flag ladder + geometric screen, silent bits as one line (2 col)
  fig06  CDFs by flow-angle class and zone + embedded stats table, 2x2 (2 col)
  fig07  far/near ISA in the parallel zone, 2x2 (2 col)
  fig08  building height x ISA grouped boxes, far side (2 col)
  fig09  mask optimization, 2x2 with a validation panel (2 col)
  figS1  2-degree flow-angle bins (2 col)
  figS3  near-nadir counterpart of fig08 (2 col)
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pyarrow.parquet as pq

PROJ = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results"
WORK = "/work/pi_kandread_umass_edu/swot-urban"
NODES = f"{PROJ}/swot_urban_nearfar_v6.5_results.parquet"
OBS = f"{WORK}/2_urban_reaches/all_final_shape.parquet"
NETP = f"{PROJ}/reviewer_checks/obstruction_network_distance.parquet"
R2 = f"{PROJ}/round2"
OUT = f"{PROJ}/figures_print"
COL, TWO = 3.5, 7.16
T = 0.1832
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.linewidth": 0.7, "lines.linewidth": 1.2, "patch.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.02, "figure.constrained_layout.w_pad": 0.02,
})


def p68(x):
    return float(np.nanpercentile(x, 68)) if len(x) else np.nan


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=600)
    fig.savefig(f"{OUT}/{name}.pdf")
    plt.close(fig)
    print("saved", name)


def load_nodes(extra=()):
    cols = ["reach_id", "time_str", "flow_angle", "far_isa", "near_isa",
            "far_building", "near_building", "error_node",
            "sample_geometry_category"] + list(extra)
    d = pq.read_table(NODES, columns=cols).to_pandas()
    d["reach_id"] = d.reach_id.astype("int64")
    d["e"] = d.error_node.abs()
    fa = d.flow_angle % 360
    d["angle_dev"] = np.minimum.reduce([fa, (fa - 180).abs(), 360 - fa])
    for c in ("far_isa", "near_isa"):
        d[c + "_pct"] = d[c] * 100 if d[c].max() <= 1 else d[c]
    d["visit"] = pd.to_datetime(d.time_str, errors="coerce", utc=True).dt.floor("h")
    return d.dropna(subset=["e"])


# ══════════════════════════════════ fig03
def fig03():
    from shapely.geometry import LineString
    d = load_nodes(extra=("node_id", "lat", "lon", "sat_lat", "sat_lon"))
    d = d.dropna(subset=["lat", "lon", "sat_lat", "sat_lon"])
    g = (d.groupby(["reach_id", "visit"])
         .agg(fi=("far_isa_pct", "mean"), ni=("near_isa_pct", "mean"),
              fb=("far_building", "mean"), nb=("near_building", "mean"),
              n=("e", "size"), slat=("sat_lat", "median"), slon=("sat_lon", "median"))
         .reset_index())
    best = None
    for rid, gr in g.groupby("reach_id"):
        A = gr[(gr.fi >= 25) & (gr.ni <= 8)]
        B = gr[(gr.ni >= 25) & (gr.fi <= 8)]
        if len(A) and len(B):
            score = min(A.fi.max(), B.ni.max()) + 0.01 * min(A.n.max(), B.n.max())
            if best is None or score > best[0]:
                best = (score, rid, A.sort_values("fi").iloc[-1], B.sort_values("ni").iloc[-1])
    if best is None:
        print("fig03: no reach met the strict flip criterion")
        return
    _, rid, va, vb = best
    print(f"fig03 reach {rid}: far-visit {va.visit.date()} far {va.fi:.0f}%/near {va.ni:.0f}%,"
          f" near-visit {vb.visit.date()} near {vb.ni:.0f}%/far {vb.fi:.0f}%")
    fig, axes = plt.subplots(1, 2, figsize=(TWO, 2.85))
    for ax, vis, tag, pl in ((axes[0], va, "far", "a"), (axes[1], vb, "near", "b")):
        nod = d[(d.reach_id == rid) & (d.visit == vis.visit)].sort_values("node_id")
        lon0, lat0 = nod.lon.mean(), nod.lat.mean()
        kx = np.cos(np.radians(lat0)) * 111.32
        x = ((nod.lon - lon0) * kx).values
        y = ((nod.lat - lat0) * 111.32).values
        sx = (vis.slon - lon0) * kx
        sy = (vis.slat - lat0) * 111.32
        sd = np.hypot(sx, sy)
        ux, uy = sx / sd, sy / sd
        line = LineString(np.column_stack([x, y]))
        wkm = 0.85
        offA = line.offset_curve(wkm)
        offB = line.offset_curve(-wkm)
        cA = np.array(offA.coords).mean(axis=0)
        near_off, far_off = (offA, offB) if np.dot(cA - [x.mean(), y.mean()],
                                                   [ux, uy]) > 0 else (offB, offA)
        urban_off = far_off if tag == "far" else near_off
        for off, colr in ((near_off, "#74add1"), (far_off, "#f4a582")):
            oc = np.array(off.coords)
            poly = np.vstack([np.column_stack([x, y]), oc[::-1]])
            ax.add_patch(mpatches.Polygon(poly, closed=True, facecolor=colr,
                                          alpha=0.45, edgecolor="none"))
        ouc = np.array(urban_off.coords)
        upoly = np.vstack([np.column_stack([x, y]), ouc[::-1]])
        ax.add_patch(mpatches.Polygon(upoly, closed=True, facecolor="none",
                                      edgecolor="#7f0000", hatch="////", lw=0.0,
                                      alpha=0.85))
        ax.plot(x, y, color="#08306b", lw=1.8, solid_capstyle="round")
        r0 = 1.6, 2.5
        ax.annotate("", xytext=(ux * r0[1], uy * r0[1]), xy=(ux * r0[0], uy * r0[0]),
                    arrowprops=dict(arrowstyle="-|>", lw=1.4, color="k"))
        if ux >= 0:
            tx, ty = ux * 2.85, uy * 2.85
        else:
            tx, ty = ux * 2.05, uy * 2.05 + 0.78
        ax.text(tx, ty, "to\nsatellite", ha="center", va="center",
                fontsize=7.3, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.8, alpha=0.75))
        mid = len(x) // 2
        for off, lab, colr in ((near_off, "near-nadir", "#08519c"),
                               (far_off, "far-nadir", "#a50f15")):
            oc = np.array(off.coords)
            om = oc[len(oc) // 2]
            away = om - [x[mid], y[mid]]
            away = away / np.hypot(*away)
            ax.text(om[0] + away[0] * 0.72, om[1] + away[1] * 0.72, lab,
                    fontsize=7.2, color=colr, ha="center", fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.7))
        zisa = vis.fi if tag == "far" else vis.ni
        zbld = vis.fb if tag == "far" else vis.nb
        om2 = ouc[int(len(ouc) * 0.22)]
        aw2 = om2 - [x[int(len(x) * 0.22)], y[int(len(y) * 0.22)]]
        aw2 = aw2 / np.hypot(*aw2)
        ax.annotate(f"urban corridor\nISA {zisa:.0f}%, {zbld:.1f} m",
                    xy=(om2[0], om2[1]),
                    xytext=(om2[0] + aw2[0] * 1.15, om2[1] + aw2[1] * 1.15 - 0.6),
                    fontsize=7.0, ha="center",
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.8, alpha=0.75),
                    arrowprops=dict(arrowstyle="-", lw=0.7, color="#7f0000"))
        ax.set_title(f"({pl}) same urban corridor, {tag}-nadir zone\n"
                     f"({vis.visit.date()})", fontsize=8.3)
        ax.set_aspect("equal")
        ax.set_xlim(-3.1, 3.1)
        ax.set_ylim(-3.4, 3.4)
        ax.set_xlabel("east-west (km)")
    axes[0].set_ylabel("north-south (km)")
    save(fig, "fig03_zone_flip")


# ══════════════════════════════════ fig04
def fig04():
    ob = pq.read_table(OBS, columns=["reach_id", "wse_relative_m",
                                     "wse_usgs_relative_m"]).to_pandas()
    ob["reach_id"] = ob.reach_id.astype("int64")
    ob["e"] = (ob.wse_relative_m - ob.wse_usgs_relative_m).abs()
    isa = pd.read_parquet(NETP, columns=["reach_id", "isa"])
    isa["reach_id"] = isa.reach_id.astype("int64")
    if isa.isa.max() <= 1.5:
        isa["isa"] = isa.isa * 100
    ob = ob.merge(isa, on="reach_id", how="left")
    from scipy import stats as st
    r, p = st.spearmanr(ob.isa, ob.e, nan_policy="omit")
    bins = [0, 5, 10, 20, 25, 30, 40, 50, 101]
    labs = ["0-5", "5-10", "10-20", "20-25", "25-30", "30-40", "40-50", "50+"]
    ob["bin"] = pd.cut(ob.isa, bins, labels=labs, right=False)
    counts = ob.bin.value_counts().reindex(labs)
    print("fig04 bins:", counts.to_dict(), f"| Spearman {r:+.3f} (p={p:.3f})",
          "| target n: 0-5=8331, 25-30=286, 50+=139")
    fig, ax = plt.subplots(figsize=(COL, 2.65))
    data, used = [], []
    for lb in labs:
        gsub = ob[ob.bin == lb]
        if len(gsub) < 20:
            continue
        data.append(gsub.e.values)
        used.append((lb, len(gsub), p68(gsub.e)))
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.62,
                    medianprops=dict(color="black", lw=1.2))
    cm = plt.cm.OrRd
    for i, patch in enumerate(bp["boxes"]):
        patch.set(facecolor=cm(0.2 + 0.6 * i / max(len(data) - 1, 1)), alpha=0.9)
    for i, (_, n, q) in enumerate(used, 1):
        ax.plot(i, q, "x", color="k", ms=5, mew=1.4)
    ax.axhline(T, color="#ca0020", ls="--", lw=1.0)
    ax.text(len(data) + 0.42, T, "T", color="#ca0020", fontsize=8, va="center")
    ax.set_ylim(0, 2.7)
    ax.set_xticklabels([f"{u[0]}\n{u[1]:,}" for u in used], rotation=0, fontsize=6.6)
    ax.set_xlabel("Reach ISA category (%)")
    ax.set_ylabel("|Relative WSE Error| (m)")
    ax.text(0.02, 0.97, f"Spearman ρ = {r:.3f} (p = {p:.2f})\nx = 68th percentile",
            transform=ax.transAxes, va="top", fontsize=7)
    save(fig, "fig04_isa_categories")


# ══════════════════════════════════ fig05
def fig05():
    ob = pq.read_table(OBS, columns=["reach_id", "time_str", "reach_q_b",
                                     "wse_relative_m", "wse_usgs_relative_m"]).to_pandas()
    ob["reach_id"] = ob.reach_id.astype("int64")
    ob["e"] = (ob.wse_relative_m - ob.wse_usgs_relative_m).abs()
    ob["visit"] = pd.to_datetime(ob.time_str, errors="coerce", utc=True).dt.floor("h")
    d = load_nodes()
    d["flag"] = (d.angle_dev <= 13) & ((d.far_building.fillna(0) > 8)
                                       | (d.far_isa_pct.fillna(0) > 19))
    vis = d.groupby(["reach_id", "visit"]).agg(nflag=("flag", "sum"),
                                               npar=("flag", "size")).reset_index()
    ob = ob.merge(vis, on=["reach_id", "visit"], how="left")
    frac = ob.nflag / ob.npar.replace(0, np.nan)
    screened = ob[frac.fillna(0) > 0.25]
    FIRING = {"water_frac": 8, "geo_suspect": 4, "class_suspect": 2,
              "class_degraded": 262144, "near_range": 16384, "far_range": 8192,
              "geo_degraded": 524288, "few_wse": 2048}
    SILENT = "bright_land, few_area, partial_obs, lake_flag, min_fit_pts, no_area, no_wse, no_pixels"
    q = ob.reach_q_b.astype("int64").values
    err = ob.e.values
    base = (err >= T).mean() * 100
    rows = [("All observations", err, len(ob), base, "base")]
    for k, b in FIRING.items():
        m = (q & b) > 0
        rows.append((k, err[m], int(m.sum()), (err[m] >= T).mean() * 100, "flag"))
    rows.append(("Geometric screen\n(>25% of nodes, this work)", screened.e.values,
                 len(screened), 100 * (screened.e >= T).mean(), "screen"))
    fig, ax = plt.subplots(figsize=(TWO, 4.15))
    ypos = np.arange(len(rows))[::-1] + 1.0
    for y, (name, ev, n, hi, kind) in zip(ypos, rows):
        colr = {"base": "#9e9e9e", "flag": "#4C72B0", "screen": "#2E7D32"}[kind]
        bp = ax.boxplot([ev], positions=[y], vert=False, widths=0.6,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=1.3))
        bp["boxes"][0].set(facecolor=colr, alpha=0.8)
        bx = dict(facecolor="white", edgecolor="none", pad=1.0)
        ax.text(2.98, y, f"{n:,}", va="center", fontsize=7.6, bbox=bx)
        ax.text(3.58, y, f"{hi:.0f}%", va="center", fontsize=7.9, fontweight="bold",
                bbox=bx, color="#c62828" if hi > base + 2 else "black")
    ax.text(0.05, 0.52, "never activate (n = 0):", fontsize=7.0, style="italic",
            color="#555555")
    ax.text(0.05, 0.18, SILENT, fontsize=6.7, style="italic", color="#555555")
    ax.axvline(T, color="#ca0020", ls="--", lw=1.0)
    ax.text(T + 0.02, len(rows) + 0.62, f"T = {T:.2f} m", color="#ca0020", fontsize=7.5)
    ax.set_yticks(ypos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8, fontweight="bold")
    ax.set_xlim(0, 4.05)
    ax.set_ylim(0.25, len(rows) + 0.9)
    ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5])
    ax.set_xlabel("|Relative WSE Error| (m)")
    ax.text(2.92, len(rows) + 0.62, "n", fontsize=7.5, style="italic")
    ax.text(3.52, len(rows) + 0.62, "share > T", fontsize=7.5, style="italic")
    print(f"fig05 screen row: n={len(screened):,} share {100*(screened.e>=T).mean():.1f}%")
    save(fig, "fig05_flags_screen")


# ══════════════════════════════════ fig06
def fig06():
    d = load_nodes()
    gc = d.sample_geometry_category.astype(str).str.lower()
    zmap = {"na": "Neither", "near": "Near", "far": "Far", "near/far": "Both"}
    d["zone"] = gc.map(zmap)
    d["ac"] = np.where(d.angle_dev <= 10, "Parallel",
                       np.where(d.angle_dev >= 80, "Perpendicular", "Other"))
    ZC = {"Neither": "#8d8d8d", "Near": "#2f7ed8", "Far": "#d84b40", "Both": "#e0a13c"}
    fig, axes = plt.subplots(2, 2, figsize=(TWO, 5.1))
    stats = {}
    for ax, ac in zip((axes[0][0], axes[0][1], axes[1][0]),
                      ("Parallel", "Other", "Perpendicular")):
        sub = d[d.ac == ac]
        for z in ("Neither", "Near", "Far", "Both"):
            gz = sub[sub.zone == z]
            if len(gz) < 50:
                continue
            xs = np.sort(gz.e.values)
            ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=ZC[z], lw=1.3, label=z)
            stats[(ac, z)] = (p68(gz.e), float(gz.e.median()),
                              100 * (gz.e >= T).mean(), len(gz))
        ax.set_xscale("log")
        ax.set_xlim(3e-3, 30)
        ax.axhline(0.68, color="k", ls=":", lw=0.8)
        ax.axvline(T, color="#ca0020", ls="--", lw=0.9)
        ax.set_title(ac)
        ax.set_xlabel("|Relative WSE Error| (m)")
    axes[0][0].set_ylabel("CDF"); axes[1][0].set_ylabel("CDF")
    axes[0][0].legend(loc="upper left", fontsize=7)
    axd = axes[1][1]
    axd.axis("off")
    collab = ["Parallel", "Other", "Perpendicular"]
    cells, rowlab = [], []
    for z in ("Neither", "Near", "Far", "Both"):
        rowlab.append(z)
        cells.append([f"{stats[(a, z)][0]:.2f} m / {stats[(a, z)][2]:.0f}%\n"
                      f"(n = {stats[(a, z)][3]:,})" for a in collab])
    tb = axd.table(cellText=cells, rowLabels=rowlab, colLabels=collab,
                   cellLoc="center", loc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(7.2)
    tb.scale(1, 2.1)
    for (ri, ci), cell in tb.get_celld().items():
        cell.set_linewidth(0.5)
        if ri == 0 or ci == -1:
            cell.set_text_props(fontweight="bold")
    axd.set_title("68th-pct error / share > T", fontsize=8)
    print("fig06 check: Both par", stats.get(("Parallel", "Both")),
          "| Far par", stats.get(("Parallel", "Far")),
          "| Both perp", stats.get(("Perpendicular", "Both")),
          "| Far perp", stats.get(("Perpendicular", "Far")))
    save(fig, "fig06_cdf_2x2")


# ══════════════════════════════════ fig07
def fig07():
    d = load_nodes()
    par = d[d.angle_dev <= 10]
    bins = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 100]
    labs = ["0-5", "5-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70",
            "70-80", "80-100"]
    fig, axes = plt.subplots(2, 2, figsize=(TWO, 5.3))
    for row, (colc, cmap, ttl) in enumerate([("far_isa_pct", plt.cm.Oranges, "Far-nadir ISA"),
                                             ("near_isa_pct", plt.cm.Greens, "Near-nadir ISA")]):
        sub = par.copy()
        sub[colc] = sub[colc].fillna(0)
        sub["bin"] = pd.cut(sub[colc], bins, labels=labs, right=False)
        axc, axb = axes[row]
        data, used = [], []
        for i, lb in enumerate(labs):
            gz = sub[sub.bin == lb]
            if len(gz) < 30:
                continue
            c = cmap(0.25 + 0.7 * i / (len(labs) - 1))
            xs = np.sort(gz.e.values)
            axc.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=c, lw=1.1,
                     label=f"{lb}%")
            data.append(gz.e.values); used.append((lb, c, p68(gz.e), len(gz)))
        axc.set_xscale("log")
        axc.set_xlim(3e-3, 40)
        axc.axhline(0.68, color="k", ls=":", lw=0.8)
        axc.axvline(T, color="#ca0020", ls="--", lw=0.9)
        axc.set_ylabel("CDF")
        axc.set_title(f"{ttl}: CDF by bin", fontsize=8.5)
        axc.legend(fontsize=6.2, ncol=2, loc="lower right", handlelength=1.1,
                   columnspacing=0.8, borderpad=0.3, labelspacing=0.25)
        bp = axb.boxplot(data, patch_artist=True, showfliers=False, widths=0.62,
                         medianprops=dict(color="black", lw=1.1))
        for patch, (_, c, _, _) in zip(bp["boxes"], used):
            patch.set(facecolor=c, alpha=0.9)
        for i, (_, _, q, n) in enumerate(used, 1):
            axb.plot(i, q, "x", color="k", ms=4.5, mew=1.3)
            axb.text(i, 55, f"{n:,}", ha="center", va="center", fontsize=5.4)
        axb.axhline(T, color="#ca0020", ls="--", lw=0.9)
        axb.set_yscale("log")
        axb.set_ylim(1e-2, 100)
        axb.set_xticklabels([u[0] for u in used], rotation=38, ha="right", fontsize=6.8)
        axb.set_ylabel("|Error| (m)")
        axb.set_title(f"{ttl}: by bin (x = 68th pct)", fontsize=8.5)
        if row == 1:
            axc.set_xlabel("|Relative WSE Error| (m)")
            axb.set_xlabel(f"{ttl} (%)")
    save(fig, "fig07_parallel_2x2")


# ══════════════════════════════════ fig08 / figS3 shared
def grouped_panel(par, side, cmap, name):
    hb = [0, 5, 7, 9, 11, 13, 100]
    hl = ["0-5", "5-7", "7-9", "9-11", "11-13", ">13"]
    ib = [0, 5, 15, 30, 45, 100]
    il = ["0-5", "5-15", "15-30", "30-45", ">45"]
    par = par.copy()
    par["hbin"] = pd.cut(par[f"{side}_building"].fillna(0), hb, labels=hl, right=False)
    par["ibin"] = pd.cut(par[f"{side}_isa_pct"].fillna(0), ib, labels=il, right=False)
    if side == "far":
        g05 = par[par.hbin == "0-5"]
        lo = p68(g05[g05.ibin == "0-5"].e)
        hi = p68(g05[g05.ibin == "30-45"].e)
        print(f"{name} check 0-5 m bin: ISA 0-5% p68 {lo:.2f} m -> 30-45% {hi:.2f} m "
              "(text: 0.50 -> 0.97)")
    fig, axes = plt.subplots(2, 1, figsize=(TWO, 5.1))
    for ax, outer, inner, olabs, ilabs, xl, lt in [
            (axes[0], "hbin", "ibin", hl, il,
             f"{side.capitalize()}-nadir building height (m)", f"{side} ISA (%)"),
            (axes[1], "ibin", "hbin", il, hl,
             f"{side.capitalize()}-nadir ISA (%)", f"{side} height (m)")]:
        pos, data, cols, ticks, tlab = 0, [], [], [], []
        for obv in olabs:
            grp = par[par[outer] == obv]
            start = pos
            for k, ibv in enumerate(ilabs):
                gz = grp[grp[inner] == ibv]
                if len(gz) < 20:
                    continue
                data.append(gz.e.values)
                cols.append(cmap(0.28 + 0.65 * k / (len(ilabs) - 1)))
                pos += 1
            if pos > start:
                ticks.append((start + pos - 1) / 2)
                tlab.append(obv)
                pos += 0.8
        bp = ax.boxplot(data, positions=np.arange(len(data)) if False else None,
                        patch_artist=True, showfliers=False, widths=0.72,
                        medianprops=dict(color="black", lw=1.0))
        # boxplot default positions 1..N; rebuild with explicit positions
        ax.cla()
        posit, pos, gaps = [], 0, []
        for obv in olabs:
            grp = par[par[outer] == obv]
            start = pos
            for k, ibv in enumerate(ilabs):
                gz = grp[grp[inner] == ibv]
                if len(gz) < 20:
                    continue
                posit.append(pos); pos += 1
            if pos > start:
                gaps.append((start, pos)); pos += 0.8
        bp = ax.boxplot(data, positions=posit, patch_artist=True, showfliers=False,
                        widths=0.72, medianprops=dict(color="black", lw=1.0))
        for patch, c in zip(bp["boxes"], cols):
            patch.set(facecolor=c, alpha=0.9)
        ticks = [(a + b - 1) / 2 for a, b in gaps]
        ax.set_yscale("log")
        ax.set_ylim(1e-2, 25)
        ax.axhline(T, color="#ca0020", ls="--", lw=0.9)
        ax.set_xticks(ticks)
        ax.set_xticklabels(tlab, fontsize=7.5)
        ax.set_xlabel(xl, fontweight="bold")
        ax.set_ylabel("|Error| (m)")
        ax.legend(handles=[mpatches.Patch(color=cmap(0.28 + 0.65 * k / (len(ilabs) - 1)),
                                          label=ilabs[k]) for k in range(len(ilabs))],
                  fontsize=6.4, ncol=len(ilabs), loc="upper left", title=lt,
                  title_fontsize=6.6, handlelength=1.0, columnspacing=0.7,
                  borderpad=0.3)
    save(fig, name)


def fig08():
    d = load_nodes()
    grouped_panel(d[d.angle_dev <= 10], "far", plt.cm.OrRd, "fig08_far_grouped")


def figS3():
    d = load_nodes()
    par = d[(d.angle_dev <= 10) & (d.near_building.fillna(0) >= 5)]
    print(f"figS3 n={len(par):,} (target 17,368)")
    grouped_panel(par, "near", plt.cm.Greens, "figS3_near_grouped")


# ══════════════════════════════════ fig09
def fig09():
    d = load_nodes()
    gc = d.sample_geometry_category.astype(str).str.lower()
    cats = {"All": np.ones(len(d), bool)}
    for raw, name in {"far": "Far", "near": "Near", "near/far": "Both",
                      "na": "Neither"}.items():
        m = (gc == raw).values
        if m.sum() > 500:
            cats[name] = m
    angles = np.arange(1, 31)
    fig, axes = plt.subplots(2, 2, figsize=(TWO, 5.4))
    axa = axes[0][0]
    CC = {"All": "#377eb8", "Far": "#e41a1c", "Near": "#4daf4a", "Both": "#ff7f00",
          "Neither": "#984ea3"}
    for name, m in cats.items():
        sub = d[m]
        base = p68(sub.e)
        red = np.array([(base - p68(sub[sub.angle_dev > a].e)) / base * 100
                        for a in angles])
        axa.plot(angles, red, lw=1.0, alpha=0.9, label=name, color=CC[name])
    rm = np.array([(d.angle_dev <= a).mean() * 100 for a in angles])
    axa2 = axa.twinx()
    axa2.plot(angles, rm, color="#c62828", lw=1.2, ls="-.")
    axa2.set_ylabel("Removed (%)", color="#c62828", fontsize=8)
    axa2.tick_params(axis="y", colors="#c62828", labelsize=7)
    sens = pd.read_csv(f"{R2}/threshold_sensitivity.csv")
    sa = sens[sens.vary == "angle_deg"].sort_values("value")
    axa.plot(sa.value, sa.reduction_pct, "k--", lw=1.2)
    amax = sa.loc[sa.reduction_pct.idxmax()]
    print(f"fig09(a) full-mask curve argmax: {amax.value:.0f}° ({amax.reduction_pct:.2f}%)")
    axa.axvspan(5, 19, color="#1f4e9c", alpha=0.10)
    axa.axvline(13, color="k", lw=1.0, ls=":")
    axa.set_title("(a) stage 1 - flow angle sweep", fontsize=8.2)
    axa.set_xlabel("Cutoff (deg)")
    axa.set_ylabel("p68 reduction (%)", color="#1f4e9c", fontsize=8)
    axa.tick_params(axis="y", colors="#1f4e9c", labelsize=7)
    axa.legend(fontsize=6.2, loc="upper left", ncol=2, columnspacing=0.7,
               handlelength=1.1, borderpad=0.3)
    axa.text(13.3, axa.get_ylim()[1] * 0.9, "θ* = 13°", fontsize=7.5, fontweight="bold",
             bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.8))

    par = d[d.angle_dev <= 13]
    for ax, colc, thr, span, ttl, unit, floor in [
            (axes[0][1], "far_building", 8, (3, 10), "(b) stage 2 - far-nadir building height", "m", 5),
            (axes[1][0], "far_isa_pct", 19, (4, 23), "(c) stage 2 - far-nadir ISA", "%", 0)]:
        sub = par[par[colc].fillna(0) > floor]
        base = p68(sub.e)
        vals = (np.arange(5.5, 20.5, 0.5) if unit == "m" else np.arange(3, 50.0, 1.0))
        keep_n = np.array([(sub[colc] <= v).sum() for v in vals])
        vals = vals[keep_n >= 200]
        red = np.array([(base - p68(sub[sub[colc] <= v].e)) / base * 100 for v in vals])
        rm = np.array([(sub[colc] > v).mean() * 100 for v in vals])
        ax.plot(vals, red, color="#1f4e9c", lw=1.3)
        ax2 = ax.twinx()
        ax2.plot(vals, rm, color="#c62828", lw=1.2, ls="-.")
        ax2.set_ylabel("Removed (%)", color="#c62828", fontsize=8)
        ax2.tick_params(axis="y", colors="#c62828", labelsize=7)
        ax.axvspan(*span, color="#1f4e9c", alpha=0.10)
        ax.axvline(thr, color="k", lw=1.0, ls=":")
        ax.invert_xaxis()
        ax.text(0.985, 0.03, "stricter →", transform=ax.transAxes, fontsize=6,
                color="0.35", ha="right")
        ax.set_title(f"{ttl}; θ* = {thr:g} {unit}", fontsize=8.2)
        ax.set_xlabel(f"Threshold ({unit})")
        ax.set_ylabel("p68 reduction (%)", color="#1f4e9c", fontsize=8)
        ax.tick_params(axis="y", colors="#1f4e9c", labelsize=7)

    axd = axes[1][1]
    boot = pd.read_csv(f"{R2}/threshold_bootstrap.csv")
    axd.hist(boot.reduction_pct, bins=24, color="#9ecae1", edgecolor="white", lw=0.4)
    lo, hi = boot.reduction_pct.quantile([.025, .975])
    axd.axvspan(lo, hi, color="#1f4e9c", alpha=0.10)
    for v, lab, colr in [(12.3, "in-sample 12.3", "#c62828"),
                         (5.1, "held-out CV 5.1", "#1f4e9c"),
                         (14.7, "oracle 14.7", "#555555"),
                         (0.07, "random 0.1", "#999999")]:
        axd.axvline(v, color=colr, lw=1.1, ls="--" if v != 12.3 else "-")
        axd.text(v, axd.get_ylim()[1] * 0.99, lab, rotation=90, va="top", ha="right",
                 fontsize=6.2, color=colr,
                 bbox=dict(facecolor="white", edgecolor="none", pad=0.4, alpha=0.75))
    axd.set_title("(d) validation - bootstrap of achieved reduction (B = 300)", fontsize=8.2)
    axd.set_xlabel("68th-pct error reduction (%)")
    axd.set_ylabel("Replicates")
    axd.text(0.30, 0.97, f"95% CI [{lo:.1f}, {hi:.1f}]", transform=axd.transAxes,
             ha="left", va="top", fontsize=7,
             bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.8))
    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([0], [0], color="#1f4e9c", lw=1.3,
               label="68th-pct error reduction (left axis)"),
        Line2D([0], [0], color="#c62828", lw=1.2, ls="-.",
               label="observations removed (right axis)"),
        Line2D([0], [0], color="k", lw=1.2, ls="--",
               label="full-mask reduction, angle cutoff varied ((a) only)"),
        Line2D([0], [0], color="k", lw=1.0, ls=":",
               label="selected threshold θ*"),
        mpatches.Patch(facecolor="#1f4e9c", alpha=0.10,
                       label="within one point of maximum")],
        loc="outside upper center", ncol=3, fontsize=6.2, frameon=True,
        edgecolor="gray", borderpad=0.4, handlelength=1.7, columnspacing=1.0)
    save(fig, "fig09_mask_2x2")


# ══════════════════════════════════ figS1
def figS1():
    d = load_nodes()
    gc = d.sample_geometry_category.astype(str).str.lower()
    zmap = {"na": "Neither", "near": "Near", "far": "Far", "near/far": "Both"}
    d["zone"] = gc.map(zmap)
    ZC = {"Neither": "#8d8d8d", "Near": "#2f7ed8", "Far": "#d84b40", "Both": "#e0a13c"}
    fig, ax = plt.subplots(figsize=(TWO, 3.15))
    for z, colr in ZC.items():
        sub = d[d.zone == z]
        cent, val, ns = [], [], 0
        for a0 in range(0, 40, 2):
            gz = sub[(sub.angle_dev >= a0) & (sub.angle_dev < a0 + 2)]
            if len(gz) < 100:
                continue
            cent.append(a0 + 1); val.append(p68(gz.e)); ns += len(gz)
        ax.plot(cent, val, marker="o", ms=2.8, lw=1.2, color=colr,
                label=f"{z} (n = {ns:,})")
    ax.axhline(T, color="#ca0020", ls="--", lw=0.9)
    ax.text(39.4, T * 1.06, "T", color="#ca0020", fontsize=7.5, ha="right")
    ax.axvspan(0, 10, color="#1f4e9c", alpha=0.08, zorder=1)
    ax.axvline(13, color="#444444", ls=":", lw=1.1, alpha=0.9)
    ax.set_yscale("log")
    ax.text(5, 0.968, "steep-rise range (<10\N{DEGREE SIGN})", transform=ax.get_xaxis_transform(),
            fontsize=6.8, color="#1f4e9c", ha="center", va="top")
    ax.text(13.4, 0.968, "\u03b8* = 13\N{DEGREE SIGN}", transform=ax.get_xaxis_transform(),
            fontsize=6.8, fontweight="bold", color="#333333", ha="left", va="top")
    ax.set_xlabel("Flow-angle deviation from track (deg), 2° bins")
    ax.set_ylabel("68th-pct |Error| (m)")
    ax.legend(fontsize=7, ncol=2)
    save(fig, "figS1_angle2deg")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for a in sys.argv[1:]:
            globals()[a]()
    else:
        for f in (fig03, fig04, fig05, fig06, fig07, fig08, fig09, figS1, figS3):
            try:
                f()
            except Exception as ex:
                print(f"{f.__name__} FAILED: {type(ex).__name__}: {ex}")
