#!/usr/bin/env python
"""
Round-2 figure rebuilds for TGRS-2026-02661, print-ready at double-column width.

Outputs (PNG 300 dpi + PDF) under figures_round2/:
  fig05_quality_flags_v2   16 bit flags + the geometric screen on the same axis
  fig07_parallel_isa_2x2   far-nadir (top) and near-nadir (bottom) ISA, CDF + box
  fig09_mask_v2            threshold curves with plateau shading and CV/bootstrap note
  figS1_angle_2deg         p68 versus flow-angle deviation in 2 degree bins, by zone
  figS3_near_counterpart   near-nadir counterpart of Figure 8, identical binning
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

PROJ = "/project/pi_kandread_umass_edu/swot-urban/3_analysis_results"
WORK = "/work/pi_kandread_umass_edu/swot-urban"
NODES = f"{PROJ}/swot_urban_nearfar_v6.5_results.parquet"
OBS = f"{WORK}/2_urban_reaches/all_final_shape.parquet"
OUT = f"{PROJ}/figures_round2"
T = 0.1832
ANGLE, HEIGHT, ISA = 13.0, 8.0, 19.0
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 14,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11,
                     "figure.dpi": 110})


def p68(x):
    return float(np.nanpercentile(x, 68)) if len(x) else np.nan


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def load_nodes():
    d = pq.read_table(NODES, columns=[
        "reach_id", "time_str", "flow_angle", "far_isa", "near_isa", "far_building",
        "near_building", "far_ratio", "error_node", "sample_geometry_category"]).to_pandas()
    d["reach_id"] = d.reach_id.astype("int64")
    d["e"] = d.error_node.abs()
    fa = d.flow_angle % 360
    d["angle_dev"] = np.minimum.reduce([fa, (fa - 180).abs(), 360 - fa])
    for c in ("far_isa", "near_isa"):
        d[c + "_pct"] = d[c] * 100 if d[c].max() <= 1 else d[c]
    d["visit"] = pd.to_datetime(d.time_str, errors="coerce", utc=True).dt.floor("h")
    return d.dropna(subset=["e"])


# ══════════════════════════════════ fig05 v2
def fig05():
    ob = pq.read_table(OBS, columns=["reach_id", "time_str", "reach_q_b",
                                     "wse_relative_m", "wse_usgs_relative_m"]).to_pandas()
    ob["reach_id"] = ob.reach_id.astype("int64")
    ob["e"] = (ob.wse_relative_m - ob.wse_usgs_relative_m).abs()
    ob["visit"] = pd.to_datetime(ob.time_str, errors="coerce", utc=True).dt.floor("h")
    d = load_nodes()
    d["flag"] = (d.angle_dev <= ANGLE) & ((d.far_building.fillna(0) > HEIGHT)
                                          | (d.far_isa_pct.fillna(0) > ISA))
    vis = d.groupby(["reach_id", "visit"]).agg(nflag=("flag", "sum"),
                                               npar=("flag", "size")).reset_index()
    ob = ob.merge(vis, on=["reach_id", "visit"], how="left")
    frac = ob.nflag / ob.npar.replace(0, np.nan)
    for cut in (0.0, 0.25, 0.5):
        sc = ob[frac.fillna(0) > cut]
        print(f"fig05 screen frac>{cut}: n {len(sc):,} ({len(sc)/len(ob)*100:.1f}%), "
              f"share>T {100*(sc.e >= T).mean():.1f}%")
    screened = ob[frac.fillna(0) > 0.25]

    BITS = {"class_suspect": 2, "geo_suspect": 4, "water_frac": 8, "bright_land": 128,
            "few_area": 1024, "few_wse": 2048, "far_range": 8192, "near_range": 16384,
            "partial_obs": 32768, "class_degraded": 262144, "geo_degraded": 524288,
            "lake_flag": 4194304, "min_fit_pts": 33554432, "no_area": 67108864,
            "no_wse": 134217728, "no_pixels": 268435456}
    q = ob.reach_q_b.astype("int64").values
    err = ob.e.values
    base = (err >= T).mean() * 100
    recs = []
    for k, b in BITS.items():
        m = (q & b) > 0
        n = int(m.sum())
        recs.append({"disp": k, "n": n, "err": err[m],
                     "hi": (err[m] >= T).mean() * 100 if n else np.nan})
    recs.sort(key=lambda r: r["n"], reverse=True)
    rows = ([{"disp": "All observations", "n": len(ob), "err": err, "hi": base, "kind": "base"}]
            + [{**r, "kind": "flag"} for r in recs]
            + [{"disp": "Geometric screen\n(>25% of nodes, this work)", "n": len(screened),
                "err": screened.e.values, "hi": 100 * (screened.e >= T).mean(),
                "kind": "screen"}])
    fig, ax = plt.subplots(figsize=(12, 10.5))
    ypos = np.arange(len(rows))[::-1]
    for y, r in zip(ypos, rows):
        col = {"base": "#9e9e9e", "flag": "#4C72B0", "screen": "#2E7D32"}[r["kind"]]
        if r["err"].size >= 5:
            bp = ax.boxplot([r["err"]], positions=[y], vert=False, widths=0.62,
                            patch_artist=True, showfliers=False,
                            medianprops=dict(color="black", linewidth=2.2))
            bp["boxes"][0].set(facecolor=col, alpha=0.75)
        ax.text(4.02, y, f"n={r['n']:,}", va="center", fontsize=12)
        ax.text(4.82, y, "—" if np.isnan(r["hi"]) else f"{r['hi']:.0f}%",
                va="center", fontsize=12, fontweight="bold",
                color="#c62828" if (not np.isnan(r["hi"]) and r["hi"] > base + 2) else "black")
    ax.axvline(T, color="#ca0020", ls="--", lw=2)
    ax.text(T, len(rows) - 0.2, f" T = {T:.2f} m", color="#ca0020", fontsize=12,
            fontweight="bold")
    ax.set_yticks(ypos)
    ax.set_yticklabels([r["disp"] for r in rows], fontsize=13, fontweight="bold")
    ax.set_xlim(0, 5.45)
    ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])
    ax.set_xlabel("|Relative WSE Error| (m)", fontweight="bold")
    ax.text(4.02, len(rows) + 0.15, "activations", fontsize=11, style="italic")
    ax.text(4.82, len(rows) + 0.15, "share > T", fontsize=11, style="italic")
    save(fig, "fig05_quality_flags_v2")


# ══════════════════════════════════ fig07 2x2
def fig07():
    d = load_nodes()
    par = d[d.angle_dev <= 10]
    print(f"fig07: parallel(<=10 deg) n={len(par):,}")
    bins = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 100]
    labs = ["0-5", "5-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70",
            "70-80", "80-100"]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10))
    for row, (col, cmap, ttl) in enumerate([("far_isa_pct", plt.cm.Oranges, "Far-nadir ISA"),
                                            ("near_isa_pct", plt.cm.Greens, "Near-nadir ISA")]):
        sub = par.copy()
        sub[col] = sub[col].fillna(0)
        sub["bin"] = pd.cut(sub[col], bins, labels=labs, right=False)
        axc, axb = axes[row]
        data, used = [], []
        for i, lb in enumerate(labs):
            g = sub[sub.bin == lb]
            if len(g) < 30:
                continue
            c = cmap(0.25 + 0.7 * i / (len(labs) - 1))
            x = np.sort(g.e.values)
            axc.plot(x, np.arange(1, len(x) + 1) / len(x), color=c, lw=2,
                     label=f"{lb}% ({len(g):,}; p68 {p68(g.e):.2f} m)")
            data.append(g.e.values); used.append((lb, c))
        axc.set_xscale("log")
        axc.set_xlim(1e-3, 60)
        axc.axhline(0.68, color="k", ls=":", lw=1.5)
        axc.axvline(T, color="#ca0020", ls="--", lw=1.5)
        axc.set_xlabel("|Relative WSE Error| (m)"); axc.set_ylabel("CDF")
        axc.set_title(f"{ttl}: CDF by ISA bin", fontweight="bold")
        axc.legend(fontsize=8.5, loc="lower right", framealpha=0.9)
        bp = axb.boxplot(data, patch_artist=True, showfliers=False,
                         medianprops=dict(color="black", lw=2))
        for patch, (_, c) in zip(bp["boxes"], used):
            patch.set(facecolor=c, alpha=0.85)
        for i, arr in enumerate(data, 1):
            axb.plot(i, p68(arr), marker="x", color="k", ms=9, mew=2.2)
        axb.axhline(T, color="#ca0020", ls="--", lw=1.5)
        axb.set_yscale("log")
        axb.set_ylim(1e-3, 30)
        axb.set_xticklabels([u[0] for u in used], rotation=45)
        axb.set_xlabel(f"{ttl} (%)"); axb.set_ylabel("|Relative WSE Error| (m)")
        axb.set_title(f"{ttl}: distribution by bin (x = 68th pct)", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig07_parallel_isa_2x2")


# ══════════════════════════════════ fig09 v2
def fig09():
    d = load_nodes()
    cats = {"All": np.ones(len(d), bool)}
    gc = d.sample_geometry_category.astype(str).str.lower()
    print("fig09 raw categories:", gc.value_counts().to_dict())
    MAPC = {"far": "Far", "near": "Near", "near/far": "Both", "both": "Both",
            "na": "Neither", "none": "Neither", "neither": "Neither"}
    for raw, name in MAPC.items():
        m = (gc == raw)
        if m.sum() > 500:
            if name in cats:
                cats[name] = cats[name] | m.values
            else:
                cats[name] = m.values
    print("fig09 categories:", {k: int(v.sum()) for k, v in cats.items()})
    angles = np.arange(1, 31)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    axa = axes[0]
    slopes_sum = np.zeros(len(angles))
    for name, m in cats.items():
        sub = d[m]
        base = p68(sub.e)
        red = np.array([(base - p68(sub[sub.angle_dev > a].e)) / base * 100 for a in angles])
        slopes_sum += np.gradient(red, angles)  # smoothed before plotting
        axa.plot(angles, red, lw=1.8, alpha=0.85, label=name)
    rm = np.array([(d.angle_dev <= a).mean() * 100 for a in angles])
    axa.set_xlabel("Flow-angle cutoff (deg)")
    axa.set_ylabel("68th-pct error reduction (%)", color="#1f4e9c")
    axa.tick_params(axis="y", colors="#1f4e9c")
    axa2 = axa.twinx()
    axa2.plot(angles, rm, color="#c62828", lw=2.2, ls="-.")
    axa2.set_ylabel("Observations removed (%)", color="#c62828")
    axa2.tick_params(axis="y", colors="#c62828")
    axa3 = axa.twinx()
    axa3.spines.right.set_visible(False)
    ker = np.ones(5) / 5
    smooth = np.convolve(slopes_sum, ker, mode="same")
    axa3.plot(angles, smooth, "k--", lw=2, label="summed slope")
    axa3.set_yticks([])
    axa.axvspan(5, 19, color="#1f4e9c", alpha=0.10)
    axa.axvline(ANGLE, color="k", lw=1.5, ls=":")
    axa.annotate("θ* = 13°", (13, axa.get_ylim()[1] * 0.92), fontsize=12,
                 fontweight="bold", ha="left")
    axa.set_title("(a) Flow angle (shaded: within 1 pt of best)", fontweight="bold")
    axa.legend(fontsize=9, loc="center right")

    par = d[d.angle_dev <= ANGLE]
    for ax, col, thr, span, lab, unit in [
            (axes[1], "far_building", HEIGHT, (3, 10), "(b) Far-nadir building height", "m"),
            (axes[2], "far_isa_pct", ISA, (4, 23), "(c) Far-nadir ISA", "%")]:
        sub = par[par[col].fillna(0) > (5 if col == "far_building" else 0)]
        base = p68(sub.e)
        vals = (np.arange(5.5, 20.5, 0.5) if col == "far_building"
                else np.arange(3, 50.0, 1.0))
        keep_n = np.array([(sub[col] <= v).sum() for v in vals])
        ok = keep_n >= 200
        vals = vals[ok]
        red = np.array([(base - p68(sub[sub[col] <= v].e)) / base * 100 for v in vals])
        rm = np.array([(sub[col] > v).mean() * 100 for v in vals])
        ax.plot(vals, red, color="#1f4e9c", lw=2.2)
        ax.set_ylabel("68th-pct error reduction (%)", color="#1f4e9c")
        ax.tick_params(axis="y", colors="#1f4e9c")
        ax2 = ax.twinx()
        ax2.plot(vals, rm, color="#c62828", lw=2.2, ls="-.")
        ax2.set_ylabel("Observations removed (%)", color="#c62828")
        ax2.tick_params(axis="y", colors="#c62828")
        ax.axvspan(*span, color="#1f4e9c", alpha=0.10)
        ax.axvline(thr, color="k", lw=1.5, ls=":")
        ax.annotate(f"θ* = {thr:g} {unit}", (thr, ax.get_ylim()[1] * 0.92), fontsize=12,
                    fontweight="bold")
        ax.invert_xaxis()
        ax.set_xlabel(f"{lab.split(') ')[1]} threshold ({unit})")
        ax.set_title(lab + " (shaded: plateau)", fontweight="bold")
    fig.suptitle("Held-out reduction 5.1% (random 0.1%, oracle 14.7%); "
                 "cluster bootstrap 95% CI on reduction: 2.5-14.4%",
                 fontsize=12, y=1.02, style="italic")
    fig.tight_layout()
    save(fig, "fig09_mask_v2")


# ══════════════════════════════════ figS1 2 deg bins
def figS1():
    d = load_nodes()
    gc = d.sample_geometry_category.astype(str).str.lower()
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, col in [("Neither", "#8d8d8d"), ("Near", "#2f7ed8"), ("Far", "#d84b40"),
                      ("Both", "#e0a13c")]:
        MAPS1 = {"Neither": ("na",), "Near": ("near",),
                 "Far": ("far",), "Both": ("near/far",)}
        sub = d[gc.isin(MAPS1[name])]
        cent, val, ns = [], [], []
        for a0 in range(0, 40, 2):
            g = sub[(sub.angle_dev >= a0) & (sub.angle_dev < a0 + 2)]
            if len(g) < 100:
                continue
            cent.append(a0 + 1); val.append(p68(g.e)); ns.append(len(g))
        ax.plot(cent, val, marker="o", ms=5, lw=2, color=col,
                label=f"{name} (n={sum(ns):,})")
    ax.axhline(T, color="#ca0020", ls="--", lw=1.5)
    ax.text(30.4, T * 1.05, f"T = {T:.2f} m", color="#ca0020", fontsize=11)
    ax.axvline(10, color="k", ls=":", lw=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("Flow-angle deviation from track (deg), 2° bins")
    ax.set_ylabel("68th-percentile |Relative WSE Error| (m)")
    ax.legend()
    save(fig, "figS1_angle_2deg")


# ══════════════════════════════════ figS3 near counterpart
def figS3():
    d = load_nodes()
    for adeg in (10, 13):
        n10 = ((d.angle_dev <= adeg) & (d.near_building.fillna(0) >= 5)).sum()
        print(f"figS3 candidate angle<={adeg}: n={n10:,}")
    par = d[(d.angle_dev <= 10) & (d.near_building.fillna(0) >= 5)].copy()
    print(f"figS3: parallel(<=10) & near_building>=5 m n={len(par):,}")
    hb = [5, 7, 9, 11, 13, 100]
    hl = ["5-7", "7-9", "9-11", "11-13", ">13"]
    ib = [0, 5, 15, 30, 45, 100]
    il = ["0-5", "5-15", "15-30", "30-45", ">45"]
    par["hbin"] = pd.cut(par.near_building, hb, labels=hl, right=False)
    par["ibin"] = pd.cut(par.near_isa_pct.fillna(0), ib, labels=il, right=False)
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 10))
    cmapi = plt.cm.Greens
    for ax, outer, inner, olabs, ilabs, xl in [
            (axes[0], "hbin", "ibin", hl, il, "Near-nadir building height (m)"),
            (axes[1], "ibin", "hbin", il, hl, "Near-nadir ISA (%)")]:
        pos, data, cols, ticks, tlab = 0, [], [], [], []
        for ob in olabs:
            grp = par[par[outer] == ob]
            start = pos
            for k, ibv in enumerate(ilabs):
                g = grp[grp[inner] == ibv]
                if len(g) < 20:
                    continue
                data.append(g.e.values)
                cols.append(cmapi(0.3 + 0.65 * k / (len(ilabs) - 1)))
                pos += 1
            if pos > start:
                ticks.append((start + pos - 1) / 2)
                tlab.append(ob)
                pos += 1
        bp = ax.boxplot(data, positions=range(len(data)), patch_artist=True,
                        showfliers=False, widths=0.75,
                        medianprops=dict(color="black", lw=1.8))
        pos = 0
        for patch, c in zip(bp["boxes"], cols):
            patch.set(facecolor=c, alpha=0.9)
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 20)
        ax.axhline(T, color="#ca0020", ls="--", lw=1.4)
        ax.set_xticks(ticks)
        ax.set_xticklabels(tlab)
        ax.set_xlabel(xl, fontweight="bold")
        ax.set_ylabel("|Relative WSE Error| (m)")
        import matplotlib.patches as mpatches
        ax.legend(handles=[mpatches.Patch(color=cmapi(0.3 + 0.65 * k / (len(ilabs) - 1)),
                                          label=ilabs[k]) for k in range(len(ilabs))],
                  fontsize=9, ncol=len(ilabs), loc="upper left",
                  title=("Near ISA (%)" if outer == "hbin" else "Near height (m)"),
                  title_fontsize=9)
    fig.tight_layout()
    save(fig, "figS3_near_counterpart")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        globals()[sys.argv[1]]()
        raise SystemExit
    fig05(); fig07(); fig09(); figS1(); figS3()
    print("all figures done ->", OUT)
